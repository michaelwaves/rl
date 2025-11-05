import json

data = []
with open('datasets/synthetic_scheming.jsonl', "r") as f:
    for line in f:
        data.append(json.loads(line))

print(f"Loaded {len(data)} records")
print("First record keys:", data[0].keys())
print("First trace object", data[0]["trace"][10])
print("Metadata:", data[0]["metadata"].keys())
print(data[0]["metadata"]["misalignment_evidence"])


trace_lengths = [len(record["trace"]) for record in data]
max_len = max(trace_lengths)
avg_len = sum(trace_lengths)/len(trace_lengths)

# Compute character length of each trace
trace_char_lengths = []
for record in data:
    # Convert each element of the trace to string and join them
    trace_str = "".join(str(step) for step in record["trace"])
    trace_char_lengths.append(len(trace_str))

max_chars = max(trace_char_lengths)
min_chars = min(trace_char_lengths)
avg_chars = sum(trace_char_lengths) / len(trace_char_lengths)

print(f"Max trace length: {max_len}")
print(f"Average trace length: {avg_len:2f}")
print(f"Max chars in trace: {max_chars}")
print(f"Avg chars in trace: {avg_chars}")
