#!/bin/bash

# Array of Poisson ratios to test
poisson_ratios=(-0.4 -0.2 0.2 0.4)
#poisson_ratios=(0.0)

for ratio in "${poisson_ratios[@]}"; do
    # Create directory name, replacing negative sign
    dir_name=$(echo ${ratio} | sed 's/-/minus_/')
    
    # Create and enter directory
    mkdir -p ${dir_name}
    cd ${dir_name}
    
    # Submit job
    sbatch --job-name=${dir_name} \
           --account=pi-ranganathanr \
           --partition=gpu \
           --output=job_output.txt \
           --nodes=1 \
           --ntasks-per-node=16 \
           --cpus-per-gpu=16 \
           --gres=gpu:1 \
           --time=36:00:00 \
           --wrap="python ../main.py ${ratio}"
    
    cd ..
done

