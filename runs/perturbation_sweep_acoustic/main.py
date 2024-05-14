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

perturbation = onp.float64(sys.argv[1])

start = 0
end = 25

results=[]
for run in range(start, end):
    bandgap_contrast, exit_flag, R_temp, k_temp, system, shift, displacement = simulation.generate_acoustic(run, perturbation)
    results.append([run, perturbation, bandgap_contrast, exit_flag])



results = np.array(results)
onp.savetxt('results.txt', results ,fmt='%i %.5f %i')


# %%
