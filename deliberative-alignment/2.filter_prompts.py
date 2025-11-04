from openai import AsyncOpenAI
from jinja2 import Environment, FileSystemLoader
env = Environment(
    loader=FileSystemLoader("templates")
)

template = env.get_template("antischeming_reward_model.jinja")
print(template.render(prompt="testPrompt"))
client = AsyncOpenAI()
