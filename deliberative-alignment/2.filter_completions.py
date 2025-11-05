from openai import AsyncOpenAI
from jinja2 import Environment, FileSystemLoader
from datasets import load_dataset
env = Environment(
    loader=FileSystemLoader("templates")
)

template = env.get_template("antischeming_reward_model.jinja")
print(template.render(prompt="testPrompt"))

dataset = load_dataset(
    "json", data_files="datasets/synthetic_scheming.json", split="train")

trajectory = dataset[0]["trace"]
answer = dataset.
client = AsyncOpenAI()
