from dataclasses import dataclass
import os
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
import torch
from datasets import load_dataset
from transformers import AutoModelForImageTextToText, Qwen2VLForConditionalGeneration, Qwen2VLProcessor
import wandb
from qwen_vl_utils import process_vision_info, vision_process

os.environ["CUDA_VISIBLE_DEVICES"]="0"

@dataclass
class Config:
    dataset_id:str
    model_id:str
    train_percent:int
    validation_percent:int
    test_percent:int

cfg = Config(
    dataset_id="HuggingFaceM4/ChartQA",
    model_id="Qwen/Qwen2-VL-7B-Instruct",
    train_percent=10,
    validation_percent=10,
    test_percent=10
)

print(torch.cuda.is_available())
print(torch.cuda.device_count())

system_message = """You are a Vision Language Model specializing in interpreting visual data from chart images.
Your task is to analyze the provided chart image and respond to queries with concise answers, usually a single word, number, or short phrase.
The charts include a variety of types (e.g., line charts, bar charts) and contain colors, labels, and text.
Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."""

def format_data(sample):
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_message
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": sample["image"],
                },
                {
                    "type": "text",
                    "text": sample['query'],
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": sample["label"][0]
                }
            ],
        },
    ]

train_dataset, eval_dataset, test_dataset = load_dataset(cfg.dataset_id, split=[f'train[:{cfg.train_percent}%]', f'val[:{cfg.validation_percent}%]', f'test[:{cfg.test_percent}%]'])

train_dataset = [format_data(sample) for sample in train_dataset]
eval_dataset = [format_data(sample) for sample in eval_dataset]
test_dataset = [format_data(sample) for sample in test_dataset]



model = Qwen2VLForConditionalGeneration.from_pretrained(
    cfg.model_id,
    device_map="cuda",
    dtype=torch.bfloat16
)

processor = Qwen2VLProcessor.from_pretrained(cfg.model_id)

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=8,
    bias="none",
    target_modules=["q_proj","v_proj"],
    task_type="CAUSAL_LM",
)

peft_model=get_peft_model(model, peft_config)
peft_model.print_trainable_parameters()

training_args = SFTConfig(
    output_dir = "qwen2-7b-instruct-trl-sft-ChartQA",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    learning_rate=2e-4,
    lr_scheduler_type="constant",
    logging_steps=1,
    eval_steps=10,
    eval_strategy="steps",
    save_strategy="steps",
    save_steps=20,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    load_best_model_at_end=True,
    bf16=False,
    fp16=True,
    tf32=False,
    max_grad_norm=0.3, 
    warmup_ratio=0.03,
    push_to_hub=True,
    report_to=None,
    gradient_checkpointing_kwargs={"use_reentrant":False},
    dataset_text_field="",
    dataset_kwargs={"skip_prepare_dataset":True},
    max_length=1024
)

training_args.remove_unused_columns=False

wandb.init(
    project="weavhacks2",
    name="qwen2-7b-instruct-trl-sft-ChartQA",
    config=training_args
)


# Create a data collator to encode text and image pairs
def collate_fn(examples):
    # Get the texts and images, and apply the chat template
    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]  # Prepare texts for processing
    image_inputs = [process_vision_info(example)[0] for example in examples]  # Process the images to extract inputs

    # Tokenize the texts and process the images
    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)  # Encode texts and images into tensors

    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels = batch["input_ids"].clone()  # Clone input IDs for labels
    labels[labels == processor.tokenizer.pad_token_id] = -100  # Mask padding tokens in labels

    # Ignore the image token index in the loss computation (model specific)
    if isinstance(processor, Qwen2VLProcessor):  # Check if the processor is Qwen2VLProcessor
        image_tokens = [151652, 151653, 151655]  # Specific image token IDs for Qwen2VLProcessor
    else:
        image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]  # Convert image token to ID

    # Mask image token IDs in the labels
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100  # Mask image token IDs in labels

    batch["labels"] = labels  # Add labels to the batch

    return batch  # Return the prepared batch
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
    peft_config=peft_config,
    processing_class=processor
)

trainer.train()
trainer.save_model(training_args.output_dir)