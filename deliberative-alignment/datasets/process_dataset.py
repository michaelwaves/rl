import json

with open("prompts/basic_scheming.txt",'r') as f:
    system_prompt = f.read()
data = []
with open("datasets/synthetic_scheming.jsonl",'r') as f:
    for line in f:
        data.append(json.loads(line))

messages = []

def ensure_str(x):
        if isinstance(x, list):
            return "\n".join(map(str, x))
        return str(x)
for d in data:
    messages.append(
        {"messages":[
            {"role":"system","content":system_prompt},
            {"role":"user", "content":ensure_str(d["trace"])},
            {"role": "assistant", "content": ensure_str(d["metadata"]["misalignment_evidence"])}
            ]})

print(messages[0])

with open("datasets/synthetic_scheming_sft.jsonl",'w') as f:
    for m in messages:
        f.write(json.dumps(m)+ "\n")