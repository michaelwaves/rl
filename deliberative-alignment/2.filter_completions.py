from openai import AsyncOpenAI
from jinja2 import Environment, FileSystemLoader
from datasets import load_dataset
import asyncio
from dataclasses import dataclass
from tqdm.asyncio import tqdm
import json


@dataclass
class Config:
    max_concurrency = 5
    max_samples = 3
    input_file = "datasets/synthetic_scheming_sft_generated.jsonl"
    output_file = "datasets/synthetic_scheming_sft_filtered.jsonl"


cfg = Config()

env = Environment(
    loader=FileSystemLoader("templates")
)

template = env.get_template("antischeming_reward_model.jinja")

with open("prompts/basic_scheming.txt", "r") as f:
    guidelines = f.read()

dataset = load_dataset(
    "json", data_files=cfg.input_file, split="train")
dataset = dataset.select(range(cfg.max_samples))

sem = asyncio.Semaphore(5)

client = AsyncOpenAI()


async def process_record(prompt, completion):
    async with sem:
        prompt = template.render(
            guidelines=guidelines,
            prompt=prompt,
            completion=completion
        )
        print(prompt)
        try:
            completion = await client.responses.create(
                input=prompt,
                model="gpt-5-mini",
                max_output_tokens=2000,
                reasoning={"effort": 'medium'}
            )
            completion_text = completion.output_text.strip()

        except Exception as e:
            completion_text = f"[Error: {e}]"

        return {"prompt": prompt, "completion": completion_text}


async def main():
    tasks = [process_record(record["prompt"], record["completion"])
             for record in dataset]
    with open(cfg.output_file, 'a', encoding='utf-8') as f:
        for result in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating SFT data"):
            res = await result
            f.write(json.dumps(res, ensure_ascii=False)+"\n")

    print(f"✅ Save {len(tasks)} prompt/completion pairs to {cfg.output_file}")

asyncio.run(main())
