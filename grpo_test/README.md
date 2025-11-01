## Running with slurm

### Option 1: srun and tmux

```bash
srun --gres=gpu:1 --time=08:00:00 --mem=32G --pty tmux new -s vllm
```
Then run the training script
```bash
python train_grpo.py
```

### Option 2: run tmux in a slurm allocation

```bash
salloc --gres=gpu:1 --time=08:00:00 --mem=32G
tmux new -s vllm
```