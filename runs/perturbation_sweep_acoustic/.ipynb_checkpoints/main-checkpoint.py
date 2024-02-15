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
import jaxnets.utils as utils
import jaxnets.simulation as simulation

import time



# %%
def generate_acoustic(run, perturbation):

    #parameters
    steps = 50
    write_every = 1
    delta_perturbation = 0.1
    number_of_nodes_per_side = 10
    nr_trials=500
    dw=0.2
    w_c=2.0
    ageing_rate=0.1
    success_frac=0.05
    k_fit = 50
    poisson_factor = 0.0
    system = utils.System(number_of_nodes_per_side, 26+run, 2.0, 0.2, 1e-1)
    system.initialize()
    system.acoustic_parameters(w_c, dw, nr_trials, ageing_rate, success_frac)
    system.auxetic_parameters(perturbation, delta_perturbation, steps, write_every)
    displacement = system.displacement
    shift = system.shift
    R = system.X
    k_bond = system.spring_constants

 

    
    opt_steps = 200
    R_temp = R
    k_temp = k_bond
    
    exit_flag=0
    
    """
    0: max steps reached
    1: gradients exceeded
    2: max k_temp exceeded
    3: converged
    
    """
    
    bandgap_contrast = 0

    result = simulation.forbidden_states_compression_NOMM(R_temp, k_temp, system, shift, displacement)

    forbidden_states_init = result.forbidden_states_init

    print('initial forbidden states: ', forbidden_states_init) 




    #initialize the grad functions
    acoustic_function = simulation.acoustic_compression_nomm_wrapper(system, shift, displacement, k_fit, poisson_factor)
    
    grad_acoustic_R = jit(grad(acoustic_function, argnums=0))
    grad_acoustic_k = jit(grad(acoustic_function, argnums=1))
    

    prev_gradient_max_k = 0
    prev_gradient_max_R = 0
    
    for i in range(opt_steps):
        gradients_k = grad_acoustic_k(R_temp, k_temp)
        gradients_R = grad_acoustic_R(R_temp, k_temp)
        
        #evaluate maximum gradients
        gradient_max_k = np.max(np.abs(gradients_k))
        gradient_max_R = np.max(np.abs(gradients_R))
        
        #calculate difference in maximum gradients
        diff_gradient_max_k = gradient_max_k - prev_gradient_max_k
        diff_gradient_max_R = gradient_max_R - prev_gradient_max_R
    
        #check if difference in gradients exceed a threshold
        if np.maximum(diff_gradient_max_k, diff_gradient_max_R) > 10.:
            print(i, diff_gradient_max_k, diff_gradient_max_R)
            exit_flag = 1
            break
        
        prev_gradient_max_k = gradient_max_k
        prev_gradient_max_R = gradient_max_R
        #check if k_temp has exceeded a threshold
        if np.max(k_temp)>10:
            print('max k_temp',np.max(k_temp))
            exit_flag = 2
            break
    
        
        k_temp = utils.update_kbonds(gradients_k, k_temp, learning_rate = 0.02)
        R_temp = utils.update_R(gradients_R, R_temp,0.01)
    
        bandgap_contrast = simulation.acoustic_compression_nomm_wrapper(system, shift, displacement, k_fit,poisson_factor)(R_temp, k_temp)
        

        if bandgap_contrast < - 0.95*forbidden_states_init: 
            print('converged')
            exit_flag = 3
            break
        
        print(i, np.max(gradients_k),np.max(gradients_R), bandgap_contrast)

    result = simulation.forbidden_states_compression_NOMM(R_temp, k_temp, system, shift, displacement)
    np.savez(str(run), 
             R_temp = R_temp, 
             k_temp = k_temp,
             perturbation = perturbation,
             connectivity = system.E,
             surface_nodes = system.surface_nodes,
             bandgap_contrast = bandgap_contrast, 
             forbidden_states_init = result.forbidden_states_init,
             forbidden_states_final = result.forbidden_states_final,
             exit_flag = exit_flag)
    return bandgap_contrast, exit_flag, R_temp, k_temp, system, shift, displacement
# %%


perturbation = onp.float64(sys.argv[1])

num_of_runs = 5 
results=[]
for run in range(num_of_runs):
    bandgap_contrast, exit_flag, R_temp, k_temp, system, shift, displacement = generate_acoustic(run, perturbation)
    results.append([run, perturbation, bandgap_contrast, exit_flag])



results = np.array(results)
onp.savetxt('results.txt', results ,fmt='%i %.5f %i')


# %%
