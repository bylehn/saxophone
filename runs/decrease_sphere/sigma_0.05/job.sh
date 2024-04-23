#!/bin/bash


sbatch --job-name=${dir_name} \
        --account=pi-depablo \
        --partition=caslake \
        --output=job_output_%j.txt \
        --nodes=1 \
        --tasks=1 \
        --mem-per-cpu=16G \
        --time=12:00:00 \
        --wrap="python main.py"


