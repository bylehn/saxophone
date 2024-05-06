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
import jaxnets.visualize as visualize
import jaxnets.utils as utils
import jaxnets.simulation as simulation

import time


# %%


perturbation = onp.float64(sys.argv[1])
number_of_nodes_per_side = 8
w_c = 2.0
dw = 0.2
opt_steps = 200

num_of_runs = 5 
results=[]
for run in range(num_of_runs):
    bandgap_contrast, exit_flag, R_temp, k_temp, system, shift, displacement = simulation.generate_acoustic(run, number_of_nodes_per_side, perturbation, w_c, dw, opt_steps)
    results.append([run,bandgap_contrast, exit_flag])



results=np.array(results)
np.savetxt('results.txt',np.array(results),fmt='%i %.5f %i')


# %%
