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

import visualize
import utils
import simulation

import time

# %%
def generate_auxetic(run, perturbation):
    steps = 50
    write_every = 1
    delta_perturbation = 0.1
    number_of_nodes_per_side = 7
    nr_trials=500
    dw=0.2
    w_c=2.0
    ageing_rate=0.1
    success_frac=0.05
    k_fit = 50
    poisson_factor=40
    system = utils.System(number_of_nodes_per_side, 26+run, 2.0, 0.2, 1e-1)
    system.initialize()
    system.acoustic_parameters(w_c, dw, nr_trials, ageing_rate, success_frac)
    system.auxetic_parameters(perturbation, delta_perturbation, steps, write_every)
    displacement = system.displacement
    shift = system.shift
    R = system.X
    k_bond = system.spring_constants
    auxetic_function = simulation.simulate_auxetic_NOMM_wrapper(R, k_bond, system,shift,displacement)
    grad_auxetic_NOMM = jit(grad(auxetic_function, argnums=0))
    grad_auxetic_NOMM_k = jit(grad(auxetic_function, argnums=1))
    opt_steps = 20
    R_temp = R
    k_temp = k_bond
    poisson = -10
    exit_flag=0
    """
    0: max steps reached
    1: gradients exceeded
    2: max k_temp exceeded
    
    """
    for i in range(opt_steps):

        #evaluate gradients for bond stiffness and positions
        gradients_R = grad_auxetic_NOMM(R_temp, k_temp)
        gradients_k = grad_auxetic_NOMM_k(R_temp, k_temp)

        #evaluate maximum gradients
        gradient_max_k = np.max(np.abs(gradients_k))
        gradient_max_R = np.max(np.abs(gradients_R))

        #check if gradients exceed a threshold
        if np.maximum(gradient_max_k,gradient_max_R)>0.1:
            print(i, gradient_max_k, gradient_max_R)
            exit_flag = 1
            break

        #check if k_temp has exceeded a threshold
        if np.max(k_temp)>10:
            exit_flag = 2
            break

        #update k and R
        k_temp = utils.update_kbonds(gradients_k, k_temp, learning_rate = 0.02)
        R_temp = utils.update_R(gradients_R, R_temp,0.01)

        #evaluate new fitness for reporting
        poisson, log, R_init, R_final = simulation.simulate_auxetic_NOMM(R_temp,
                                                                k_temp,
                                                                system,
                                                                shift,
                                                                displacement)
        print(i, gradient_max_k, gradient_max_R,  poisson)
    np.savez(str(run), R_temp = R_temp, k_temp = k_temp, poisson = poisson, exit_flag = exit_flag)
    return poisson, exit_flag, R_temp, k_temp, system, shift, displacement

# %%
num_of_runs = 1
results=[]

start_time = time.time()
for run in range(num_of_runs):
    poisson, exit_flag, R_temp, k_temp, system, shift, displacement = generate_auxetic(run, 1)
    results.append([poisson, exit_flag])

execution_time = time.time() - start_time
print(f"Execution Time: {execution_time} seconds")
# %%
