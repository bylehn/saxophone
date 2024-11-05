# %%
import jax.numpy as np
import numpy as onp
from jax import random
from jax.config import config; config.update("jax_enable_x64", True); config.update("jax_debug_nans", False)
from jax_md import space
from jax import random, grad
from jax import jit, vmap
from jax import lax
import networkx as nx
import sys
sys.path.insert(0, '/scratch/midway3/bylehn/auxetic_networks_jaxmd/')  # Adds the parent directory to sys.path
import saxophone.visualize as visualize
import saxophone.utils as utils
import saxophone.simulation as simulation
import gc
import time


# %%


poisson_target = onp.float64(sys.argv[1])
opt_steps = 200
dw = 0.1
w_c = 2.0
k_angle = 0.01
perturbation = 1.0
num_of_runs = 10
size = 10
results = []
for run in range(num_of_runs):
    try:
        poisson_distance, bandgap_distance, exit_flag, R_temp, k_temp, system, shift, displacement, result = simulation.generate_auxetic_acoustic_adaptive(
            run, size, k_angle, perturbation, w_c, dw, poisson_target, opt_steps)
        
        # Check if results are valid
        if np.isnan(poisson_distance) or np.isnan(bandgap_distance):
            print(f"Run {run}: NaN encountered, skipping")
            continue
            
        results.append([run, poisson_distance, bandgap_distance, exit_flag])
        
    except Exception as e:
        print(f"Run {run} failed with error: {str(e)}")
        continue
    finally:
        gc.collect()

# Only save valid results
valid_results = [r for r in results if not np.any(np.isnan(r))]
if valid_results:
    results = np.array(valid_results)
    onp.savetxt('results.txt', results, fmt='%i %.5f %.5f %i')  # Note: Added format for bandgap_distance
