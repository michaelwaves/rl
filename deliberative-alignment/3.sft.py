
from trl import SFTTrainer
from datasets import load_dataset
from dataclasses import dataclass
@dataclass
class Config:
    model:str
    dataset:str

cfg = Config(
    model="Qwen/Qwen3-0.6B",
    dataset="datasets/synthetic_scheming_sft.jsonl"
)
trainer = SFTTrainer(
    model=cfg.model,
    train_dataset = load_dataset("json",data_files=cfg.dataset, split="train")
)

trainer.train()