# %%
import jax.numpy as np
import numpy as onp
from jax.config import config; config.update("jax_enable_x64", True); config.update("jax_debug_nans", False)
import sys
sys.path.insert(0, '/scratch/midway3/bylehn/auxetic_networks_jaxmd/')  # Adds the parent directory to sys.path
import jaxnets.simulation as simulation

# %%

k_angle = 0.01
perturbation = 1.0
w_c = 2.0
dw = 0.2
num_of_runs = 50
opt_steps = 200
size = 10
results=[]
for run in range(num_of_runs):
    result = simulation.generate_acoustic(run, size, k_angle, perturbation, w_c, dw, opt_steps)
    bandgap_contrast, exit_flag, R_temp, k_temp, system, shift, displacement = result
    results.append([run, bandgap_contrast, exit_flag])



results=np.array(results)
onp.savetxt('results.txt',np.array(results),fmt='%i %.5f %i')


# %%
