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
import saxophone.visualize as visualize
import saxophone.utils as utils
import saxophone.simulation as simulation

import time


# %%


#perturbation = onp.float64(sys.argv[1])
number_of_nodes_per_side = 4
perturbation = 0.2 * number_of_nodes_per_side
opt_steps = 200
k_angle = 0.1
num_of_runs = 1000 
results=[]

for run in range(num_of_runs):
    poisson, exit_flag, R_temp, k_temp, system, shift, displacement = simulation.generate_auxetic(run, number_of_nodes_per_side, k_angle, perturbation, opt_steps)
    results.append([run,poisson, exit_flag])



results=np.array(results)
np.savetxt('results.txt',np.array(results),fmt='%i %.5f %i')


# %%
