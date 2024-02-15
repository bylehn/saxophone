#!/bin/bash

# Array of perturbation iis to test
perturbation=(0.1 0.2 0.5 0.8 1.0 1.3 1.5 1.8 2.0 2.3 2.5)

for ii in "${perturbation[@]}"; do
    # Replace negative sign with 'minus_' to avoid mkdir command error
    dir_name=$(echo ${ii} | sed 's/-/minus_/')

    # Check if the directory exists, if not, create it
    if [ ! -d "${dir_name}" ]; then
        mkdir -p ${dir_name}
    fi
    
    # Change to the newly created directory
    cd ${dir_name}
    
    # Submit the job
    sbatch --job-name=${dir_name} \
           --account=pi-depablo \
           --partition=caslake \
           --output=job_output_%j.txt \
           --nodes=1 \
           --tasks=4 \
           --mem-per-cpu=16G \
           --time=12:00:00 \
           --wrap="python ../main.py ${ii}"
    
    # Change back to the original directory
    cd ..
done

