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

import time


# %%


poisson_target = onp.float64(sys.argv[1])
opt_steps = 200
frequency_closed = 2.0
width_closed = 0.2
frequency_opened = 1.6
width_opened = 0.2
k_angle = 0.01
perturbation = 1.0
num_of_runs = 10
size = 10
results = []
for run in range(num_of_runs):
    poisson_distance, bandgap_distance, exit_flag, R_temp, k_temp, system, shift, displacement, result = simulation.generate_auxetic_acoustic_shift(run,
                                                                                                                                                    size,
                                                                                                                                                    k_angle,
                                                                                                                                                    perturbation,
                                                                                                                                                    frequency_closed,
                                                                                                                                                    width_closed,
                                                                                                                                                    frequency_opened,
                                                                                                                                                    width_opened,
                                                                                                                                                    poisson_target,
                                                                                                                                                    opt_steps)
    results.append([run, poisson_distance, bandgap_distance, exit_flag])

results=np.array(results)
onp.savetxt('results.txt',np.array(results),fmt='%i %.5f %i')

