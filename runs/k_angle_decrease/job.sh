#!/bin/bash

    # Change to the newly created directory
dir=k_001_pert_1

mkdir -p $dir
cd $dir

# Submit the job
sbatch --job-name=k_0.01 \
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

