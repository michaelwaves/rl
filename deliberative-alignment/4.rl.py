from datasets import load_dataset
from jinja2 import Environment, FileSystemLoader
from trl import GRPOTrainer, GRPOConfig
from openai import AsyncOpenAI
import re
import asyncio


def reward_num_unique_letter(completions, **kwargs) -> list[float]:
    completion_contents = [completion[0]["content"]
                           for completion in completions]
    rewards = [float(len(set(content))) for content in completion_contents]
    return rewards


async def reward_better_monitoring(completions, prompts, **kwargs):
    completion_contents = [completion[0]["content"]
                           for completion in completions]

    # maybe we could also parse out score here and compare it to label we have?
    # some kind of weighted reward between process and outcome?

    tasks = [score_completion(c) for c in completion_contents]
    results = await asyncio.gather(tasks)
    return results


# GRPO Trainer might accept synchronous reward function only
def reward_better_monitoring_sync(completions, prompts, **kwargs):
    return asyncio.run(reward_better_monitoring(completions, prompts, **kwargs))


async def score_completion(completion) -> int:
    env = Environment(
        loader=FileSystemLoader("templates")
    )
    template = env.get_template("antischeming_reward_model.jinja")
    with open("prompts/basic_scheming.txt", "r") as f:
        guidelines = f.read()
    prompt = template.render(
        prompt="",
        completion=completion,
        guidelines=guidelines
    )
    client = AsyncOpenAI()
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

    pattern = r'(?i)SCORE\s*:\s*(-?\d+(?:\.\d+)?)'
    matches = re.findall(pattern, completion_text)

    return matches[0]


def main():
