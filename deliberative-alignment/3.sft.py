
from trl import SFTTrainer
from datasets import load_dataset
from dataclasses import dataclass


@dataclass
class Config:
    model: str = "Qwen/Qwen3-0.6B"
    dataset: str = "datasets/synthetic_scheming_sft.jsonl"


cfg = Config()

trainer = SFTTrainer(
    model=cfg.model,
    train_dataset=load_dataset("json", data_files=cfg.dataset, split="train")
)

trainer.train()
