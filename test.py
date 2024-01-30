
import jax.numpy as np
import numpy as onp
from jax import random
from jax.config import config; config.update("jax_enable_x64", True); config.update("jax_debug_nans", False)
from jax_md import space
from jax import random, grad
from jax import jit, vmap
from jax import lax
import networkx as nx

import utils
import simulation

import time

def time_calc(number_of_nodes_per_side):
    
    steps = 10
    write_every = 1
    perturbation = 1.0
    delta_perturbation = 0.01
    nr_trials=500
    dw=0.4
    w_c=2.0
    ageing_rate=0.1
    success_frac=0.05
    system = utils.System(number_of_nodes_per_side, 21, 2.0, 0.1, 1e-1)
    system.initialize()
    system.acoustic_parameters(w_c, dw, nr_trials, ageing_rate, success_frac)
    system.auxetic_parameters(perturbation, delta_perturbation, steps, write_every)
    displacement = system.displacement
    shift = system.shift
    R = system.X
    k_bond = system.spring_constants
    grad_auxetic = grad(simulation.simulate_auxetic_optimize, argnums=1)
    grad_acoustic = grad(simulation.acoustic_compression_grad, argnums=2)
    grad_auxetic_NOMM = grad(simulation.simulate_auxetic_optimize_NOMM, argnums=0)
    StartTime=time.time()
    poisson, log, R_init, R_final = simulation.simulate_auxetic(R,
                                                                k_bond,
                                                                system,
                                                                shift,
                                                                displacement)
    SimTime=time.time()
   
    _ = grad_auxetic(R,
                                                k_bond,
                                                system,
                                                shift,
                                                displacement)
    GradTime=time.time()

    return SimTime-StartTime, GradTime-SimTime

times=[]
for i in range(5,25,2):
    
    t1,t2=time_calc(i)
    times.append([i,t1,t2])
    print([i,t1,t2])


# Open the file in write mode ('w')
with open('output.txt', 'w') as f:
    # Write R_init array to the file
    f.write('Positions: \n')
    for sub_array in R_init:
        f.write(f'{sub_array[0]}, {sub_array[1]}\n')


