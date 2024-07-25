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


perturbation = onp.float64(sys.argv[1])
number_of_nodes_per_side = 10
w_c = 2.0
dw = 0.2
k_angle = 1.0
poisson_target = -0.3
opt_steps = 200
num_of_runs = 5 
results=[]
for run in range(num_of_runs):
    poisson_distance, bandgap_distance, exit_flag, R_temp, k_temp, system, shift, displacement, result = simulation.generate_auxetic_acoustic_adaptive(46, number_of_nodes_per_side, k_angle, perturbation, w_c, dw, poisson_target, opt_steps, output_evolution = False)
    results.append([run,poisson_distance, bandgap_distance, exit_flag])



results=np.array(results)
np.savetxt('results.txt',np.array(results),fmt='%i %.5f %i')


# %%
