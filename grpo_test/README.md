## Running with slurm


#see node name and general resources (gpus)
```sh
sinfo -p compute -o "%N %G"

#see running jobs
squeue

#enter tmux
tmux

#enter interactive shell for 10 minutes. 3 ways
#sinteractive: helper script, might not exist
sinteractive --gres=gpu:2 -t  00:10:00
#srun runs command and then exits
srun --gres=gpu:2 -t 00:10:00 --pty bash
#salloc also works, automatically creates session
salloc --gres=gpu:2 -t 00:10:00

#enter the same salloc job:
sinfo #find job id
srun --jobid=<JOB_ID> --pty bash

#to check, run nvtop
#to exit (kills session) run exit

#to ssh and connect to like a vllm server running on a compute node, jump through the control node
ssh -L 8888:localhost:8000 -J m.yu@mats m.yu@l40-worker
```