#!/bin/bash


# Submit the job
sbatch --job-name=big \
        --account=pi-depablo \
        --partition=caslake \
        --output=job_output.txt \
        --nodes=1 \
        --tasks=1 \
        --mem-per-cpu=32G \
        --time=12:00:00 \
        --wrap="python main.py"
    


