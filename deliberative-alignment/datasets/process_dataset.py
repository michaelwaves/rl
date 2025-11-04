import json

with open("prompts/basic_scheming.txt",'r') as f:
    system_prompt = f.read()
data = []
with open("datasets/synthetic_scheming.jsonl",'r') as f:
    for line in f:
        data.append(json.loads(line))

messages = []
for d in data:
    messages.append(
        {"messages":[
            {"role":"system","content":system_prompt},
            {"role":"user", "content":d["trace"]},
            {"role": "assistant", "content": d["metadata"]["misalignment_evidence"]}
            ]})

print(messages[0])