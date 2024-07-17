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
from memory_profiler import profile


# %%


poisson_target = onp.float64(sys.argv[1])
opt_steps = 200
dw = 0.1
w_c = 2.0
k_angle = 0.01
perturbation = 1.0
num_of_runs = 1
size = 10

results = []

@profile
def run_simulation():
    for run in range(num_of_runs):
        poisson_distance, bandgap_distance, exit_flag, R_temp, k_temp, system, shift, displacement, result = simulation.generate_auxetic_acoustic_adaptive(
            run, size, k_angle, perturbation, w_c, dw, poisson_target, opt_steps
        )
        results.append([run, poisson_distance, bandgap_distance, exit_flag])
        
        # Save individual results to file to avoid memory accumulation
        onp.savetxt(f'results_{run}.txt', np.array([run, poisson_distance, bandgap_distance, exit_flag]), fmt='%s')
 

    # Optionally, save the accumulated results at the end
    onp.savetxt('results_all.txt', np.array(results), fmt='%s')

run_simulation()


# %%
