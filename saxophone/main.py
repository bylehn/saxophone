# %%
import jax.numpy as np
import numpy as onp
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
import gc
import time
from memory_profiler import profile
from timeit import default_timer as timer


# %%
# Input parameters
#poisson_target = onp.float64(sys.argv[1])
opt_steps = 3

k_angle = 0.01
perturbation = 1.0
num_of_runs = 1
size = 10

min_steps = 50
write_every = 1
delta_perturbation = 0.1
nr_trials=500
ageing_rate=0.1
success_frac=0.05


# Acoustic parameters
frequency_opened = 1.0
frequency_closed = 1.0
width_opened = 0.35
width_closed = 0.35
dw = 0.1
w_c = 2.0
k_fit = 2.0/(width_opened**2)


# Auxetic parameters
poisson_target = -1.0



# %%
# Initialize network

system = utils.System(size, k_angle, 2.0, 0.35)
system.initialize(random_seed=22)
system.acoustic_parameters(frequency_opened, width_opened, nr_trials, ageing_rate, success_frac)
system.auxetic_parameters(perturbation, delta_perturbation, min_steps, write_every)

# %%
# Minimizing the initial configuration
_, R ,_  = simulation.simulate_minimize_penalty(system)

system.X = R
displacement = system.displacement
system.create_spring_constants()
system.calculate_initial_angles_method(displacement)


# %%

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

start = timer()
run_simulation()
end = timer()

print(f"Time elapsed: {end - start} s")

# %%
