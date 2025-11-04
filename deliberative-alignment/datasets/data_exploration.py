import json

data = []
with open('datasets/synthetic_scheming.jsonl',"r") as f:
    for line in f: 
        data.append(json.loads(line))

print(f"Loaded {len(data)} records")
print("First record keys:",data[0].keys())
print("First trace object", data[0]["trace"][10])
print("Metadata:", data[0]["metadata"].keys())
print(data[0]["metadata"]["misalignment_evidence"])


