Everything is built around the synthetic scheming dataset datasets/synthetic_scheming.jsonl
https://github.com/richcooper95/black-box-scheming-monitors/blob/main/gemini_synthetic_transcripts/scheming/synthetic_scheming.jsonl

This dataset has trajectories, in record["trace"] and scheming scores in metadata["misalignment_evidence"]

## Deliberative Alignment

SFT + RL 
https://arxiv.org/pdf/2412.16339
Similar to constitutional AI, but the model actually sees the text of the spec
Original templates from paper are in templates original_*jinja

4 step process
1. Generate Completions Data
Starting from (prompt) => (prompt, CoT, output)
Model has access to policy in templates/antischeming_completion.jinja and prompts/basic_scheming.txt

2. Filter Completions Data
Use a "reward model" (in this case GPT 5 mini with medium reasoning) to go over the prompt, CoT, output tuples and choose the best ones. Drop the spec from the prompts here

3. SFT
Train on the high quality (prompt, completion) pairs

4. RL
Use reward model with access to safety policies to do RL. A monitor for the monitor


## Issues
The transcripts are really long and the model sometimes gets confused. Like the triangle arbitrage Helios transcript, supposed to be a 9 but model ranks it 3
The completions are sort of ok
Not sure if the antischeming_* jinjas I wrote make sense

## Current state

1.generate_completions.py takes the traces from synthetic_scheming.jsonl and makes completions (monitor scores). outputs to synthetic_scheming_sft_generated.jsonl
2.filter_completions.py, outputs scores of how well the monitor did in synthetic_scheming_sft_generated.jsonl to synthetic_scheming_sft_filtered.jsonl

only did 3 samples of the above so far

3.sft.py works with Rich's original synthetic_scheming.jsonl prompts and completions (426 samples). takes 1 hr on 2 L40s with Qwen 3 0.6B. The training data is in datasets/synthetic_scheming.jsonl

4. RL tbd