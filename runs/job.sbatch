#!/bin/bash

#SBATCH --job-name=test_${cores} \
--account=pi-depablo \
           --partition=caslake \
           --output=python_test_${cores}_cores_%j.out \
           --nodes=1 \
           --cpus-per-task=$cores \
           --time=00:10:00 \
           --wrap="python ../main.py'"