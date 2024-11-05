#!/bin/bash

# Array of poisson ratios to test
#poisson=(-1.0 -0.8 -0.6 -0.4 -0.2 -0.1 0.1 0.2 0.4 0.6 0.8 1.0)
poisson=(-1.0 1.0)

for ratio in "${poisson[@]}"; do
    # Replace negative sign with 'minus_' to avoid mkdir command error
    dir_name=$(echo ${ratio} | sed 's/-/minus_/')
    
    # Create a directory with the modified name
    mkdir -p ${dir_name}
    
    # Change to the newly created directory
    cd ${dir_name}
    
    # Submit the job
    sbatch --job-name=${dir_name} \
            --account=pi-ranganathanr \
            --partition=gpu \
            --output=job_output.txt \
            --nodes=1 \
            --ntasks-per-node=16 \
            --cpus-per-gpu=16 \
            --gres=gpu:1 \
            --qos=debug \
            --time=00:30:00 \
           --wrap="python ../main.py ${ratio}"
    
    # Change back to the original directory
    cd ..
done

