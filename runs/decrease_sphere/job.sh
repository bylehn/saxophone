#!/bin/bash

# Array of poisson ratios to test

cd k_0001/

# Submit the job
sbatch --job-name=${dir_name} \
        --account=pi-depablo \
        --partition=caslake \
        --output=job_output_%j.txt \
        --nodes=1 \
        --tasks=1 \
        --mem-per-cpu=32G \
        --time=12:00:00 \
        --wrap="python ../main.py"

# Change back to the original directory
cd ..


