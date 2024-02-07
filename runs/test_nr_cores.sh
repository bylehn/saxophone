#!/bin/bash

# Array of cores to test (e.g., 1, 2, 4, 8 cores)
cores_to_test=(1 2 4 8 16 32 48)

for cores in "${cores_to_test[@]}"
do
    # Submit a job for each core count
    sbatch --job-name=test_${cores} \
	       --account=pi-depablo \
           --partition=caslake \
           --output=python_test_${cores}_cores_%j.out \
           --nodes=1 \
           --tasks=$cores \
           --time=00:10:00 \
           --wrap="python ../main.py"
done

