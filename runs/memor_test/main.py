import jax.numpy as np
import numpy as onp
from jax import random
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", False)
import sys
sys.path.insert(0, '/home/fabian/Documents/network/saxophone')
import saxophone.visualize as visualize
import saxophone.utils as utils
import saxophone.simulation as simulation
from memory_profiler import profile
from timeit import default_timer as timer

# Set simulation parameters
poisson_target = -1.0
opt_steps = 10
dw = 0.1
w_c = 2.0
k_angle = 0.01
perturbation = 1.0
num_of_runs = 1
size = 10

results = []

@profile
def run_simulation():
    for run in range(num_of_runs):
        # Create system and graph data outside of JIT-compiled function
        system = utils.System(size, k_angle, run, 2.0, 0.35)
        G, X, E, L, surface_mask = utils.create_delaunay_graph(system)
        
        # Extract necessary data from the graph
        graph_data = {
            'X': np.array(X),
            'E': np.array(E),
            'L': np.array(L),
            'surface_mask': np.array(surface_mask),
            'degrees': np.array([d for _, d in G.degree()])
        }
        
        poisson_distance, bandgap_distance, exit_flag, R_temp, k_temp, system, shift, displacement, result = simulation.generate_auxetic_acoustic_adaptive_memory(
            run, size, k_angle, perturbation, w_c, dw, poisson_target, opt_steps, graph_data=graph_data
        )
        
        results.append([run, poisson_distance, bandgap_distance, exit_flag])
        
        # Save individual results to file to avoid memory accumulation
        onp.savetxt(f'results_{run}.txt', onp.array([run, poisson_distance, bandgap_distance, exit_flag]), fmt='%s')
        
        # Clear any cached computations to free memory
        jax.clear_caches()
        
    # Optionally, save the accumulated results at the end
    onp.savetxt('results_all.txt', onp.array(results), fmt='%s')

if __name__ == "__main__":
    start = timer()
    run_simulation()
    end = timer()
    print(f"Time elapsed: {end - start} s")