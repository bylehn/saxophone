import jax.numpy as np
import numpy as onp
from jax.config import config
config.update("jax_enable_x64", True)
config.update("jax_debug_nans", False)
import sys
sys.path.insert(0, '/scratch/midway3/bylehn/auxetic_networks_jaxmd/')
import saxophone.simulation as simulation
import gc

if len(sys.argv) != 2:
    print("Usage: python main.py <poisson_target>")
    sys.exit(1)

# Parameters
poisson_target = onp.float64(sys.argv[1])
opt_steps = 500
size = 15
k_angle = 0.01
perturbation = 0.8
num_of_runs = 3

# Bandgap parameters
frequency_center = 2.0
bandgap_width = 0.1
frequency_closed = frequency_center - bandgap_width
frequency_opened = frequency_center + bandgap_width

# Initialize results list
results = []

# Run simulations
for run in range(num_of_runs):
    try:
        print(f"\nRun {run} starting...")
        # Create auxetic wrapper with poisson target
        auxetic_wrapper = simulation.simulate_auxetic_wrapper(
            R=None,  # Will be set by the shift function
            k_bond=None,  # Will be set by the shift function
            system=None,  # Will be set by the shift function
            shift=None,  # Will be set by the shift function
            displacement=None,  # Will be set by the shift function
            poisson_target=poisson_target
        )
        
        poisson_distance, bandgap_distance, exit_flag, R_temp, k_temp, system, shift, displacement, result = \
            simulation.generate_auxetic_acoustic_shift(
                run=run,
                number_of_nodes_per_side=size,
                k_angle=k_angle,
                perturbation=perturbation,
                frequency_closed=frequency_closed,
                width_closed=bandgap_width,
                frequency_opened=frequency_opened,
                width_opened=bandgap_width,
                poisson_target=poisson_target,
                opt_steps=opt_steps,
                initial_lr=0.02
            )
        
        if np.isnan(poisson_distance) or np.isnan(bandgap_distance):
            print(f"Run {run}: NaN encountered, skipping")
            continue
            
        results.append([run, poisson_distance, bandgap_distance, exit_flag])
        print(f"Run {run} completed: poisson_dist={poisson_distance:.3f}, "
              f"bandgap_dist={bandgap_distance:.3f}, exit_flag={exit_flag}")
        
    except Exception as e:
        print(f"Run {run} failed with error: {str(e)}")
        continue
    finally:
        gc.collect()

# Save valid results
valid_results = [r for r in results if not np.any(np.isnan(r))]
if valid_results:
    results = np.array(valid_results)
    onp.savetxt('results.txt', results, fmt='%i %.5f %.5f %i')
    print("\nResults saved to results.txt")
else:
    print("\nNo valid results to save")