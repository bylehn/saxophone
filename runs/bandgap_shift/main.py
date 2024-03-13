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
import jaxnets.visualize as visualize
import jaxnets.visualize as visualize
import jaxnets.utils as utils
import jaxnets.simulation as simulation

import time


# %%


poisson_target = onp.float64(sys.argv[1])
dw = 0.1
w_c = 2.0
perturbation = 2.0
num_of_runs = 5
frequency_closed = 1.9
width_closed = 0.1
frequency_opened = 2.1
width_opened = 0.1
results=[]
for run in range(num_of_runs):
    poisson_distance, bandgap_distance, exit_flag, R_temp, k_temp, system, shift, displacement, result = simulation.generate_auxetic_acoustic_shift(run, poisson_target, perturbation, frequency_closed, width_closed, frequency_opened, width_opened)
    results.append([run, poisson_distance, bandgap_distance, exit_flag])



results=np.array(results)
np.savetxt('results.txt',np.array(results),fmt='%i %.5f %i')


# %%
