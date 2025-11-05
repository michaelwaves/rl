from attr import dataclass
from jinja2 import Environment, FileSystemLoader
from openai import AsyncOpenAI
import json
import asyncio
from tqdm.asyncio import tqdm


@dataclass
class Config:
    max_concurrency = 5
    max_samples = 3
    output_file = "datasets/synthetic_scheming_sft_generated.jsonl"


cfg = Config()

env = Environment(
    loader=FileSystemLoader("templates")
)

completion_template = env.get_template("antischeming_completion.jinja")

with open("prompts/basic_scheming.txt", "r") as f:
    guidelines = f.read()

trajectories = []
with open("datasets/synthetic_scheming.jsonl", "r") as f:
    for record in f:
        trajectories.append(json.loads(record)["trace"])
trajectories = trajectories[:cfg.max_samples]


client = AsyncOpenAI()

sem = asyncio.Semaphore(cfg.max_concurrency)


async def process_trajectory(trajectory):
    async with sem:
        prompt = completion_template.render(
            guidelines=guidelines,
            conversation=trajectory
        )
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
    tasks = [process_trajectory(trajectory)
             for trajectory in trajectories]
    with open(cfg.output_file, 'a', encoding='utf-8') as f:
        for result in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating SFT data"):
            res = await result
            f.write(json.dumps(res, ensure_ascii=False)+"\n")

    print(f"✅ Save {len(tasks)} prompt/completion pairs to {cfg.output_file}")

asyncio.run(main())
