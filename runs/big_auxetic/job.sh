#!/bin/bash


# Submit the job
sbatch --job-name=big \
        --account=pi-depablo \
        --partition=depablo-gpu \
        --output=job_output.txt \
        --nodes=1 \
        --ntasks-per-node=16 \
        --cpus-per-gpu=16 \
        --gres=gpu:1 \
        --time=12:00:00 \
        --wrap="python main.py"
    


