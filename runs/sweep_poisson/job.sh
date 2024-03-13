#!/bin/bash

# Array of poisson ratios to test
poisson=(-1.0 -0.9 -0.8 -0.7 -0.6 -0.5 -0.4 -0.3 -0.2 -0.1 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

for ratio in "${poisson[@]}"; do
    # Replace negative sign with 'minus_' to avoid mkdir command error
    dir_name=$(echo ${ratio} | sed 's/-/minus_/')
    
    # Create a directory with the modified name
    mkdir -p ${dir_name}
    
    # Change to the newly created directory
    cd ${dir_name}
    
    # Submit the job
    sbatch --job-name=${dir_name} \
           --account=pi-depablo \
           --partition=caslake \
           --output=job_output_%j.txt \
           --nodes=1 \
           --tasks=1 \
           --mem-per-cpu=16G \
           --time=12:00:00 \
           --wrap="python ../main.py ${ratio}"
    
    # Change back to the original directory
    cd ..
done

