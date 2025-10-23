from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

dataset = load_dataset('trl-lib/ultrafeedback-prompt',split='train')
def reward_num_unique_letters(completions, **kwargs):
    completion_contents = [completion[0]["content"] for completion in completions]
    return [float(len(set(content))) for content in completion_contents]

training_args = GRPOConfig(output_dir="Qwen3-0.5B-GRPO",
    use_vllm=True,
    vllm_mode="colocate")

trainer = GRPOTrainer(
    model="Qwen/Qwen3-0.6B",
    reward_funcs = reward_num_unique_letters,
    args = training_args,
    train_dataset=dataset,

)

trainer.train()