from jinja2 import Environment, FileSystemLoader

env = Environment(
    loader=FileSystemLoader("templates")
)

completion_template = env.get_template("antischeming_completion.jinja")
