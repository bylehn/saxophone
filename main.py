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
import jaxnets.visualize as visualize
import jaxnets.utils as utils
import jaxnets.simulation as simulation

import time

# %%

perturbation = onp.float64(sys.argv[1])

num_of_runs = 5 
results=[]
for run in range(num_of_runs):
    bandgap_contrast, exit_flag, R_temp, k_temp, system, shift, displacement = simulation.generate_acoustic(run, perturbation)
    results.append([run,bandgap_contrast, exit_flag])



results=np.array(results)
np.savetxt('results.txt',np.array(results),fmt='%i %.5f %i')


# %%
