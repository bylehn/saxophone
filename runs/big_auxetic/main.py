# %%
import jax.numpy as np
import numpy as onp
from jax import random
import jax
jax.config.update("jax_enable_x64", True); jax.config.update("jax_debug_nans", False)
from jax_md import space
from jax import random, grad
from jax import jit, vmap
from jax import lax
import networkx as nx
import sys
sys.path.insert(0, '/home/fabian/Documents/network/saxophone')  # Adds the parent directory to sys.path
import saxophone.visualize as visualize
import saxophone.utils as utils
import saxophone.simulation as simulation

import time


# %%


dw = 0.1
w_c = 2.0
perturbation = 2.0
k_angle = 10.0
opt_steps = 500
num_of_runs = 1
size = 20
results=[]
for run in range(num_of_runs):
    poisson, exit_flag, R_temp, k_temp, system, shift, displacement, evolution_log = simulation.generate_auxetic(run, size, k_angle, perturbation, opt_steps, output_evolution = True)
    results.append([run, poisson, exit_flag])



results=np.array(results)
onp.savetxt('results.txt',np.array(results),fmt='%i %.5f %i')


# %%
