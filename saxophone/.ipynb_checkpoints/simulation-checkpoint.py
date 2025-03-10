import jax.numpy as np
import numpy as onp
from jax.config import config; config.update("jax_enable_x64", True)
from jax_md import energy, minimize
from jax import jit, vmap, grad
from jax import lax
from jax import debug
import networkx as nx
import saxophone.utils as utils
import saxophone.energies as energies
from collections import namedtuple
#from memory_profiler import profile
import gc

Result_forbidden_modes = namedtuple('Result', [
    'D_init',
    'V_init',
    'C_init',
    'forbidden_states_init',
    'frequency_init',
    'R_init',
    'D_final',
    'V_final',
    'C_final',
    'forbidden_states_final',
    'frequency_final',
    'R_final',
    'log',
    'poisson'
])

def simulate_minimize_penalty(R,
                     k_bond,
                     system,
                     shift,
                     displacement
                     ):
    """
    minimizes using a System instance and is set to evaulate network's penalties only (spring constants and angle energy not included)

    system: System instance containing the state and properties of the system
    shift: shift parameter for the FIRE minimization
    perturbation: total perturbation
    delta_perturbation: perturbation step size
    displacement: displacement function
    steps: number of steps in the simulation
    write_every: frequency of writing data
    optimize: boolean to indicate whether to optimize the poisson ratio

    Returns:
    poisson: poisson ratio
    log: log dictionary
    R_init: initial positions
    R_final: final positions
    """
    #update variables according to R so that the derivative accounts for them
    system.X=R
    displacement = system.displacement
    system.create_spring_constants()
    system.calculate_initial_angles_method(displacement)

    # Get the surface nodes.
    top_indices = system.surface_nodes['top']
    bottom_indices = system.surface_nodes['bottom']
    left_indices = system.surface_nodes['left']
    right_indices = system.surface_nodes['right']

    #fix all surface nodes
    mask = np.ones(R.shape)
    mask = mask.at[left_indices].set(0)
    mask = mask.at[right_indices].set(0)
    mask = mask.at[top_indices].set(0)
    mask = mask.at[bottom_indices].set(0)

    num_iterations = 1
    cumulative_perturbation = 0.0

    log = {
            'force': onp.zeros((num_iterations*(system.steps // system.write_every),) + R.shape),
            'position': onp.zeros((num_iterations*(system.steps // system.write_every),) + R.shape)
    }

    def step_fn_generator(apply, start_idx):
        def step_fn(i, state_and_log):
            """
            Minimizes the configuration at each step.

            i: step number
            state_and_log: state and log dictionary
            """
            fire_state, log = state_and_log
            i_adjusted = i + start_idx
            log['force'] = lax.cond(i_adjusted % system.write_every == 0,
                                        lambda p: p.at[i_adjusted // system.write_every].set(fire_state.force),
                                        lambda p: p,
                                        log['force'])

            log['position'] = lax.cond(i_adjusted % system.write_every == 0,
                                            lambda p: p.at[i_adjusted // system.write_every].set(fire_state.position),
                                            lambda p: p,
                                            log['position'])

            fire_state = apply(fire_state)
            return fire_state, log

        return step_fn

    def perturb_and_minimize(i, state_log_perturb):
            R_current, log, cumulative_perturbation = state_log_perturb
            R_perturbed = R_current#.at[left_indices, 0].add(system.delta_perturbation)
            #cumulative_perturbation += system.delta_perturbation
            # Update the force function with the new positions
            force_fn = energies.constrained_force_fn(R_perturbed, energy_fn_wrapper, mask)
    
            # Reinitialize the fire state with the new positions and updated force function
            fire_init, fire_apply = minimize.fire_descent(force_fn, shift, dt_max = 0.2)
            fire_state = fire_init(R_perturbed)
    
            # Update step function generator with the new start index
    
            start_idx = i * (system.steps // system.write_every)
    
            step_fn = step_fn_generator(fire_apply, start_idx)
    
            # Perform the minimization step
            fire_state, log = lax.fori_loop(0, system.steps, step_fn, (fire_state, log))
            R_perturbed = fire_state.position
    
            return R_perturbed, log, cumulative_perturbation

    def penalty_energy(R, system, **kwargs):
        displacement = system.displacement
        crossing_penalty = np.sum(energies.bond_crossing_penalty(system, system.angle_triplets, displacement, R))
        # Bond energy (assuming that simple_spring_bond is JAX-compatible)
        node_energy = energy.soft_sphere_pair(displacement, sigma = system.soft_sphere_sigma, epsilon= system.soft_sphere_epsilon)(R, **kwargs)

        return crossing_penalty + node_energy

    def energy_fn_wrapper(R, **kwargs):
        return penalty_energy(R, system, **kwargs)

    R_init = R

    R_final, log, cumulative_perturbation = lax.fori_loop(0, num_iterations, perturb_and_minimize, (R_init, log, cumulative_perturbation))

    print("Energy hopefully reduced from ", energies.penalty_energy(R_init, system)," to ", energies.penalty_energy(R_final, system))
    
    return R_init, R_final, log


def simulate_auxetic(R,
                     k_bond,
                     system,
                     shift,
                     displacement
                     ):
    """
    Simulates the auxetic process using a System instance and is set to evaulate network with a "natural" equal width spring constants

    system: System instance containing the state and properties of the system
    shift: shift parameter for the FIRE minimization
    perturbation: total perturbation
    delta_perturbation: perturbation step size
    displacement: displacement function
    steps: number of steps in the simulation
    write_every: frequency of writing data
    optimize: boolean to indicate whether to optimize the poisson ratio

    Returns:
    poisson: poisson ratio
    log: log dictionary
    R_init: initial positions
    R_final: final positions
    """
    #update variables according to R so that the derivative accounts for them
    system.X=R
    displacement = system.displacement
    system.create_spring_constants()
    system.calculate_initial_angles_method(displacement)

    # Get the surface nodes.
    top_indices = system.surface_nodes['top']
    bottom_indices = system.surface_nodes['bottom']
    left_indices = system.surface_nodes['left']
    right_indices = system.surface_nodes['right']
    mask = np.ones(R.shape)
    mask = mask.at[left_indices].set(0)
    mask = mask.at[right_indices].set(0)
    num_iterations = onp.abs(int(onp.ceil(system.perturbation / system.delta_perturbation)))
    # Initialize the cumulative perturbation
    cumulative_perturbation = 0.0

    log = {
            'force': onp.zeros((num_iterations*(system.steps // system.write_every),) + R.shape),
            'position': onp.zeros((num_iterations*(system.steps // system.write_every),) + R.shape)
    }

    def step_fn_generator(apply, start_idx):
        def step_fn(i, state_and_log):
            """
            Minimizes the configuration at each step.

            i: step number
            state_and_log: state and log dictionary
            """
            fire_state, log = state_and_log
            i_adjusted = i + start_idx
            log['force'] = lax.cond(i_adjusted % system.write_every == 0,
                                        lambda p: p.at[i_adjusted // system.write_every].set(fire_state.force),
                                        lambda p: p,
                                        log['force'])

            log['position'] = lax.cond(i_adjusted % system.write_every == 0,
                                            lambda p: p.at[i_adjusted // system.write_every].set(fire_state.position),
                                            lambda p: p,
                                            log['position'])

            fire_state = apply(fire_state)
            return fire_state, log

        return step_fn

    def perturb_and_minimize(i, state_log_perturb):
        R_current, log, cumulative_perturbation = state_log_perturb
        R_perturbed = R_current.at[left_indices, 0].add(system.delta_perturbation)
        cumulative_perturbation += system.delta_perturbation
        # Update the force function with the new positions
        force_fn = energies.constrained_force_fn(R_perturbed, energy_fn_wrapper, mask)

        # Reinitialize the fire state with the new positions and updated force function
        fire_init, fire_apply = minimize.fire_descent(force_fn, shift, dt_max = 0.2)
        fire_state = fire_init(R_perturbed)

        # Update step function generator with the new start index

        start_idx = i * (system.steps // system.write_every)

        step_fn = step_fn_generator(fire_apply, start_idx)

        # Perform the minimization step
        fire_state, log = lax.fori_loop(0, system.steps, step_fn, (fire_state, log))
        R_perturbed = fire_state.position

        return R_perturbed, log, cumulative_perturbation

    def energy_fn(R, system, **kwargs):
        angle_energy = np.sum(energies.angle_energy(system, system.angle_triplets, displacement, R))
        # Bond energy (assuming that simple_spring_bond is JAX-compatible)
        bond_energy = energy.simple_spring_bond(displacement, system.E, length=system.distances, epsilon=k_bond[:, 0])(R, **kwargs)
        node_energy = energy.soft_sphere_pair(displacement, sigma = system.soft_sphere_sigma, epsilon= system.soft_sphere_epsilon)(R, **kwargs)

        return bond_energy + angle_energy + node_energy

    def energy_fn_wrapper(R, **kwargs):
        return energy_fn(R, system, **kwargs)

    R_init = R
    # Initial dimensions (before deformation)
    # Exclude the first and last index for horizontal edges (top and bottom)
    # as these are corners with the left and right edges
    initial_horizontal = onp.mean(R_init[right_indices[1:-1]], axis=0)[0] - onp.mean(R_init[left_indices[1:-1]], axis=0)[0]

    # Exclude the first and last index for vertical edges (left and right)
    # as these are corners with the top and bottom edges
    initial_vertical = onp.mean(R_init[top_indices[1:-1]], axis=0)[1] - onp.mean(R_init[bottom_indices[1:-1]], axis=0)[1]

    R_final, log, cumulative_perturbation = lax.fori_loop(0, num_iterations, perturb_and_minimize, (R_init, log, cumulative_perturbation))
    # Final dimensions (after deformation)
    final_horizontal = onp.mean(R_final[right_indices[1:-1]], axis=0)[0] - onp.mean(R_final[left_indices[1:-1]], axis=0)[0]
    final_vertical = onp.mean(R_final[top_indices[1:-1]], axis=0)[1] - onp.mean(R_final[bottom_indices[1:-1]], axis=0)[1]

    # Calculate the poisson ratio.
    poisson = utils.poisson_ratio(initial_horizontal, initial_vertical, final_horizontal, final_vertical)
    #fit = fitness(poisson)

    return poisson, log, R_init, R_final

#@profile
def simulate_auxetic_wrapper(R,
                     k_bond,
                     system,
                     shift,
                     displacement
                     ):
    """
    Simulates the auxetic process using a System instance.

    R: position matrix  
    k_bond: spring constant matrix
    system: System instance containing the state and properties of the system   
    shift: shift parameter for the FIRE minimization
    displacement: displacement function             

    Returns:
    poisson: poisson ratio

    """             


    def simulate_auxetic_optimize(R,
                         k_bond
                         ):
        """
        wrapped function.
    
        R: position matrix  
        k_bond: spring constant matrix
             
    
        Returns:
        poisson: poisson ratio
    
        """             
        poisson, log, R_init , R_final = simulate_auxetic(R, k_bond, system, shift, displacement)
        output = poisson + energies.penalty_energy(R_init, system) / system.penalty_scale + utils.stiffness_penalty(system, k_bond)
        # penalty added here because making a separate funtion and gradient could be memory loading...
        return output
    return simulate_auxetic_optimize

def get_bond_importance(C, V, D, D_range):
    # Create a mask for the modes within the specified range
    # Create a mask for the modes within the specified range
    mode_mask = ((D > D_range[0]) & (D < D_range[1])).astype(float)
    
    # Compute delta_E for all modes, irrespective of the mask
    delta_E_all_modes = np.dot(C.T, V)

    # Apply mask to delta_E to zero out the modes outside the range
    delta_E = delta_E_all_modes * mode_mask

    # Compute bond importance using vmap for element-wise operations
    bond_importance = vmap(lambda ec: np.mean(np.abs(ec)))(delta_E)
    bond_importance = bond_importance/np.max(bond_importance)
    bond_importance_centered = bond_importance - np.mean(bond_importance)
    bond_importance_normalized = bond_importance_centered/np.max(np.abs(bond_importance_centered))
    
    return bond_importance_normalized.reshape(-1,1)

def create_compatibility(system, R):
    N_b = system.E.shape[0]
    #mdict = dict(zip(range(system.N), system.m))
    #nx.set_node_attributes(system.G, mdict, 'Mass')

    # Initialize C with zeros
    C = np.zeros((2 * system.N, N_b))
    
    # Compute the b_vec for each edge
    b_vec = R[system.E[:, 0], :] - R[system.E[:, 1], :]
    b_vec_norm = np.linalg.norm(b_vec, axis=1, keepdims=True)
    b_vec_normalized = b_vec / b_vec_norm

    # Ensure that b_vec_normalized is correctly reshaped for broadcasting
    b_vec_normalized = b_vec_normalized.reshape(-1, 2)

    # Update C using the .at property for advanced indexing
    for i in range(N_b):
        C = C.at[2 * system.E[i, 0]:2 * system.E[i, 0] + 2, i].add(b_vec_normalized[i])
        C = C.at[2 * system.E[i, 1]:2 * system.E[i, 1] + 2, i].add(-b_vec_normalized[i])

    return C

def get_forbidden_states(C, k_bond, system):
    """
    Get the forbidden modes for a given spring constant k.

    C: compatibility matrix
    k: spring constant
    M: mass matrix
    w_c: center frequency
    dw: frequency width

    Returns:
    D: eigenvalues
    V: eigenvectors
    forbidden_states: number of forbidden states
    """
    kd = np.diag(np.squeeze(k_bond))
    K = C @ kd @ C.T
    DMAT = np.linalg.inv(system.mass) @ K
    #debug.print("DMAT: {DMAT}", DMAT=DMAT)
    D, V = np.linalg.eigh(DMAT)
    D = np.real(D)
    frequency = np.sqrt(np.abs(D))
    forbidden_states = np.sum(np.logical_and(frequency > system.frequency_center - system.frequency_width/2,
                                              frequency < system.frequency_center + system.frequency_width/2))
    V = np.real(V)
    return D, V, forbidden_states, frequency

def age_springs(k_old, system, D, V, C, D_range):
    """
    Aging algorithm for springs.

    k_old: old spring constant matrix
    X: position matrix
    C: compatibility matrix
    V: eigenvectors
    D: eigenvalues
    D_range: frequency range
    ageing_rate: ageing rate

    Returns:
    k_new: new spring constant matrix
    """
    bond_importance = get_bond_importance(C, V, D, D_range)
    k_new = k_old * (1 + 2 * system.ageing_rate * bond_importance)
    return k_new

def optimize_ageing(C, k, system, success_frac):
    """
    Optimize for acoustic bandgap.

    C: compatibility matrix
    k: spring constant matrix
    M: mass matrix
    w_c: center frequency
    dw: frequency width
    N_trials: number of trials
    ageing_rate: ageing rate
    success_frac: success fraction

    Returns:
    k: spring constant matrix
    success: success boolean
    trial: trial number
    """

    frequency_range=[system.frequency_center - system.frequency_width/2,
                     system.frequency_center + system.frequency_width/2]
    
    D_range = [x**2 for x in frequency_range]

    D, V, forbidden_states_initial = get_forbidden_states(C, k, system)
    if forbidden_states_initial == 0:
        return k,1,0
    for trial in range(1, system.nr_trials + 1):

        k = age_springs(k, system, D, V, C, D_range)

        D, V, forbidden_states = get_forbidden_states(C, k, system)
        print(trial,forbidden_states)

        if forbidden_states <= success_frac * forbidden_states_initial:

            return k, 1, trial

    return k, 0, trial

def age_springs_compressed(k_old, system, result, C_init, C_final, D_range):
    """
    Aging algorithm for springs when doing compression.

    Returns:
    k_new: new spring constant matrix
    """
    threshold = 0.05  # Define a threshold for importance increase
    penalty_factor = 0.1  # Factor to penalize increasing importance in initial state

    bond_importance_init = scale_bond_importance(get_bond_importance(C_init, result.V_init, result.D_init, D_range))
    bond_importance_final = scale_bond_importance(get_bond_importance(C_final, result.V_final, result.D_final, D_range))

    # Calculate the ratio of forbidden states (add 1 to initial to avoid division by zero)
    #forbidden_states_ratio = (result.forbidden_states_init + 1) / (result.forbidden_states_final + 1)

    # Calculate the differential importance
    differential_importance = bond_importance_final - bond_importance_init

    # Adjust the differential importance to avoid increasing initial forbidden modes
        # Strengthen or weaken springs based on differential importance
    adjustment_factors = np.where(differential_importance > threshold, 
                                  1 + system.ageing_rate * differential_importance, 
                                #   0 + 1) # Strengthen
                                  1 - penalty_factor * system.ageing_rate * differential_importance) 
    # Adjust bond importance based on the ratio
   # bond_importance_adjusted = bond_importance_final  #-  forbidden_states_ratio * bond_importance_init
    
    # Apply adjustments while maintaining a similar distribution
    #k_new = k_old * adjustment_factors
    #k_new = adjust_distribution(k_new, k_old)  # Function to maintain distribution
    k_new = k_old * (1 + 2*system.ageing_rate * differential_importance)
    return k_new

def scale_bond_importance(bond_importance):
    """
    Scales the bond importance vector. 

    Returns:
    scaled bond importance vector centered at 0 and max extents -1 to 1
    """
    #returns the vector centred at mean = 0 and max extents -1 to 1
    bi_centered = bond_importance - np.mean(bond_importance)
    return bi_centered/np.max(np.abs(bi_centered))

#@profile
def forbidden_states_compression(R,
                                 k_bond,
                                 system,
                                 shift,
                                 displacement
    ):
    """
    Get the forbidden modes when compressing the network.

    Returns:
    D_init: initial eigenvalues
    V_init: initial eigenvectors
    forbidden_states_init: initial number of forbidden states
    R_init: initial positions
    D_final: final eigenvalues
    V_final: final eigenvectors
    forbidden_states_final: final number of forbidden states
    R_final: final positions
    log: log dictionary
    poisson: fish, jk. the poisson ratio
    """
    poisson, log, R_init, R_final = simulate_auxetic(R,
                                               k_bond,
                                               system,
                                               shift,
                                               displacement
                                               )

    C_init = create_compatibility(system, R_init)

    C_final = create_compatibility(system, R_final)
    D_init, V_init, forbidden_states_init, frequency_init = get_forbidden_states(C_init, k_bond, system)
    D_final, V_final, forbidden_states_final, frequency_final = get_forbidden_states(C_final, k_bond, system)

    return Result_forbidden_modes(D_init,
                                  V_init,
                                  C_init,
                                  forbidden_states_init,
                                  frequency_init,
                                  R_init,
                                  D_final,
                                  V_final,
                                  C_final,
                                  forbidden_states_final,
                                  frequency_final,
                                  R_final,
                                  log,
                                  poisson
    )

def optimize_ageing_compression(R, system, k_bond, shift, displacement):
    """
    Optimize for acoustic bandgap when compressing the network.

    Returns:
    k_bond: spring constant matrix
    success: success boolean
    trial: trial number
    """

    frequency_range=[system.frequency_center - system.frequency_width/2,
                     system.frequency_center + system.frequency_width/2]
    
    D_range = [x**2 for x in frequency_range]

    def condition(state):
        trial, current_k_bond, optimization_successful = state
        not_reached_max_trials = lax.lt(trial, system.nr_trials + 1)
        not_optimization_successful = ~optimization_successful
        continue_condition = lax.bitwise_and(not_reached_max_trials, not_optimization_successful)
        return continue_condition

    def loop_body(state):
        trial, current_k_bond, _ = state
        result = forbidden_states_compression(R, current_k_bond, system, shift, displacement)

        optimization_successful = lax.bitwise_and(lax.ge(result.forbidden_states_init, 1),
                                                  lax.eq(result.forbidden_states_final, 0))

        C_init = create_compatibility(system, result.R_init)
        C_final = create_compatibility(system, result.R_final)
        k_bond_updated = age_springs_compressed(current_k_bond, system, result, C_init, C_final, D_range)
        # Check if any entry in k_bond is NaN
        #print(trial, result.forbidden_states_init, result.forbidden_states_final)

        # Update carry values, note that `k_bond_updated` is only used if the loop continues.
        return trial + 1, k_bond_updated, optimization_successful

    initial_state = (1, k_bond, False)
    final_trial, final_k_bond, _ = lax.while_loop(condition, loop_body, initial_state)
    forbidden_states_final = forbidden_states_compression(R, final_k_bond, system, shift, displacement).forbidden_states_final
    forbidden_states_init = forbidden_states_compression(R, final_k_bond, system, shift, displacement).forbidden_states_init


    return final_k_bond, final_trial, forbidden_states_init, forbidden_states_final

#@profile
def acoustic_compression_wrapper(system, shift, displacement, k_fit):
    def acoustic_compression_grad(R, k_bond):
        """
        This function might not be needed since we can just use the forbidden_states_compression, but to 
        retain functionality of other functions, we keep it for now.
        """
        def gap_objective(frequency, frequency_center, k_fit):
            
            return np.sum(np.exp(-0.5*k_fit * (frequency - frequency_center)**2))

        result = forbidden_states_compression(R, k_bond, system, shift, displacement)
        # Fitness energy for the initial state with a penalty for reducing forbidden states
        fit_init = gap_objective(result.frequency_init, system.frequency_center, k_fit)

        # Fitness energy for the final state
        fit_final = gap_objective(result.frequency_final, system.frequency_center, k_fit)

        # Weighted objective function: Heavily weight the final state's energy
        objective_function = fit_final - fit_init
        
        #return result.forbidden_states_init, result.forbidden_states_final
       
        return objective_function +  energies.penalty_energy(result.R_init, system) / system.penalty_scale + utils.stiffness_penalty(system, k_bond)
        # penalty 
    return acoustic_compression_grad


def acoustic_auxetic_adaptive_wrapper(system, shift, displacement, k_fit, bandgap_bias, poisson_target, poisson_bias):
    def acoustic_auxetic_adaptive(R, k_bond):
        """
        objective function that adapts the objective function to the state of optimization
        poisson bias: distance of the original poisson before optimization to the target, used to scale poisson_distance
        bandgap bias: used to define the radius of the bandgap distance. essentially we want the optimized (fit_init,fit_final) = (bandgap_bias, 0)
        """
        def gap_objective(frequency, frequency_center, k_fit):
            
            return np.sum(np.exp(-0.5*k_fit * (frequency - frequency_center)**2))

        



        result = forbidden_states_compression(R, k_bond, system, shift, displacement)
        # Fitness energy for the initial state with a penalty for reducing forbidden states
        fit_init = gap_objective(result.frequency_init, system.frequency_center, k_fit)

        # Fitness energy for the final state
        fit_final = gap_objective(result.frequency_final, system.frequency_center, k_fit)

        # Weighted objective function: Heavily weight the final state's energy

        poisson_distance = (result.poisson - poisson_target) / poisson_bias
        bandgap_distance =  (fit_final/bandgap_bias)**2 + (1- (fit_init/bandgap_bias))**2  # eucleadian distance in reduced fitness space of the current fitness as the ideal fitness of fit_final = 0 and fit_initial = bandgap_bias. This usually starts at 1, although can go above
        objective_function = bandgap_distance + poisson_distance**2 #squared to maintain positivity :)
        
        #return result.forbidden_states_init, result.forbidden_states_final
        return objective_function  +  energies.penalty_energy(result.R_init, system) / system.penalty_scale + utils.stiffness_penalty(system, k_bond) # penalty 
    return acoustic_auxetic_adaptive


def acoustic_bandgap_shift_wrapper(system, shift, displacement, frequency_closed, width_closed, frequency_opened, width_opened):  
    """
    Creates objective function that opens one bandgap and closes another upon compression. Can be used to shift
    
    system : system class
    shift, displacement : JAX, M.D. standards
    frequency_closed : frequency of bandgap center being closed upon compression
    width_closed :  width of the bandgap being closed
    frequency_opened : frequency of bandgap center being opened upon compression
    width_opened : width of the bandgap being opened
    
    """
    def acoustic_bandgap_shift(R, k_bond):

        def gap_objective(frequency, frequency_center, k_fit):
            
            return np.sum(np.exp(-0.5*k_fit * (frequency - frequency_center)**2))

        #evaluate biasing width with 2 times inverse variance for the gap widths
        k_fit_closed = 2.0/(width_closed**2) 
        k_fit_opened = 2.0/(width_opened**2) 


        result = forbidden_states_compression(R, k_bond, system, shift, displacement)

        
        # initial state  objective =  number of states in closed (needs to be low) - number of states in the opened bandgap (needs to be high)
        objective_init = gap_objective(result.frequency_init, frequency_closed, k_fit_closed) - gap_objective(result.frequency_init, frequency_opened, k_fit_opened) 

        # final state objective = number of states in opened bandap (needs to be low) - number of states in closed bandgap (needs to be high)
        objective_final = gap_objective(result.frequency_final, frequency_opened, k_fit_opened) - gap_objective(result.frequency_final, frequency_closed, k_fit_closed)

        
        
        objective_function = objective_init + objective_final #Note how we add in this case since different objectives are being achieved
        
 
        return objective_function +  energies.penalty_energy(result.R_init, system) / system.penalty_scale + utils.stiffness_penalty(system, k_bond)# penalty 
    return acoustic_bandgap_shift


#Generate Functional Network Functions for Parameter Sweeps

def generate_acoustic(run, number_of_nodes_per_side, k_angle, perturbation, w_c, dw, opt_steps, output_evolution = False):
    #run: kinda a random number seed
    
    #parameters
    steps = 50
    write_every = 1
    delta_perturbation = 0.1
    nr_trials=500
    ageing_rate=0.1
    success_frac=0.05
    k_fit = 2.0/(dw**2) 
    system = utils.System(number_of_nodes_per_side, k_angle, run, 2.0, 0.35)
    system.initialize()
    system.acoustic_parameters(w_c, dw, nr_trials, ageing_rate, success_frac)
    system.auxetic_parameters(perturbation, delta_perturbation, steps, write_every)
    displacement = system.displacement
    shift = system.shift
    R = system.X
    k_bond = system.spring_constants


    #minimizing the initial configuration

    _, R ,_  = simulate_minimize_penalty(R,
                                        k_bond,
                                        system,
                                        shift,
                                        displacement)

    system.X= R
    displacement = system.displacement
    system.create_spring_constants()
    system.calculate_initial_angles_method(displacement)
    k_bond = system.spring_constants
    R_temp = R
    k_temp = k_bond


    if output_evolution:
        #set up evolution bits
        R_evolution = np.zeros((opt_steps, system.N, 2))
        R_evolution = R_evolution.at[0].set(R_temp)
        k_evolution = np.zeros((opt_steps, k_temp.shape[0], 1))
        k_evolution = k_evolution.at[0].set(k_temp)

    
    exit_flag=0
    
    """
    0: max steps reached
    2: max k_temp exceeded
    3: converged
    
    """
    
    bandgap_contrast = 0
    result = forbidden_states_compression(R_temp, k_temp, system, shift, displacement)
    forbidden_states_init = result.forbidden_states_init
    print('initial forbidden states: ', forbidden_states_init) 


    #initialize the grad functions
    acoustic_function = acoustic_compression_wrapper(system, shift, displacement, k_fit)
    
    grad_acoustic_R = jit(grad(acoustic_function, argnums=0))
    grad_acoustic_k = jit(grad(acoustic_function, argnums=1))


    for i in range(opt_steps):
        
        #acoustic gradients
        gradients_k = grad_acoustic_k(R_temp, k_temp)
        gradients_R = grad_acoustic_R(R_temp, k_temp)


        #evaluate maximum gradients for diagnostics, the condition for explosive gradients is not necessary as the objective function minimizes initial energy
        gradient_max = np.max( np.abs( np.vstack((gradients_k, 
                                                  gradients_R.ravel()[:, np.newaxis] ))))
    
        
        k_temp = utils.update_kbonds(gradients_k, k_temp, learning_rate = 0.02)
        R_temp = utils.update_R(system.surface_mask, gradients_R, R_temp, 0.01)
    
        bandgap_contrast = acoustic_compression_wrapper(system, shift, displacement, k_fit)(R_temp, k_temp)
        
        print(i, gradient_max, bandgap_contrast - energies.penalty_energy(R_temp, system)/system.penalty_scale, energies.penalty_energy(R_temp, system) , utils.stiffness_penalty(system, k_temp))
        
        #set evolution bits for the network
        if output_evolution:
            R_evolution = R_evolution.at[i+1].set(R_temp)
            k_evolution = k_evolution.at[i+1].set(k_temp)

    result = forbidden_states_compression(R_temp, k_temp, system, shift, displacement)
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
    if output_evolution:
        evolution_log = {'position' : R_evolution, 'bond_strengths' : k_evolution}
        return bandgap_contrast, exit_flag, R_temp, k_temp, system, shift, displacement, evolution_log
    else:
        return bandgap_contrast, exit_flag, R_temp, k_temp, system, shift, displacement

def generate_auxetic(run, number_of_nodes_per_side, k_angle, perturbation, opt_steps, output_evolution = False):
    steps = 50
    write_every = 1
    delta_perturbation = 0.1
    nr_trials=500
    dw=0.2
    w_c=2.0
    ageing_rate=0.1
    success_frac=0.05
    system = utils.System(number_of_nodes_per_side, k_angle, run, 2.0, 0.35)
    system.initialize()
    system.acoustic_parameters(w_c, dw, nr_trials, ageing_rate, success_frac)
    system.auxetic_parameters(perturbation, delta_perturbation, steps, write_every)
    displacement = system.displacement
    shift = system.shift
    R = system.X
    k_bond = system.spring_constants

    _, R ,_  = simulate_minimize_penalty(R,
                                        k_bond,
                                        system,
                                        shift,
                                        displacement)

    system.X= R
    displacement = system.displacement
    system.create_spring_constants()
    system.calculate_initial_angles_method(displacement)
    k_bond = system.spring_constants
    


    
    auxetic_function = simulate_auxetic_wrapper(R, k_bond, system,shift,displacement)
    grad_auxetic = jit(grad(auxetic_function, argnums=0))
    grad_auxetic_k = jit(grad(auxetic_function, argnums=1))

    
    R_temp = R
    k_temp = k_bond

    if output_evolution:
        R_evolution = np.zeros((opt_steps, system.N, 2))
        R_evolution = R_evolution.at[0].set(R_temp)
        k_evolution = np.zeros((opt_steps, k_temp.shape[0], 1))
        k_evolution = k_evolution.at[0].set(k_temp)


    poisson = -10
    exit_flag=0
    """
    0: max steps reached
    1: gradients exceeded
    2: max k_temp exceeded
    
    """
    prev_gradient_max = 0



    for i in range(opt_steps):

        #evaluate gradients for bond stiffness and positions
        gradients_R = grad_auxetic(R_temp, k_temp)
        gradients_k = grad_auxetic_k(R_temp, k_temp)

        gradient_max = np.max( np.abs( np.vstack((gradients_k, 
                                                  gradients_R.ravel()[:, np.newaxis] ))))
    
        diff_gradient_max = gradient_max - prev_gradient_max
            
        prev_gradient_max = gradient_max
    


        #update k and R
        k_temp = utils.update_kbonds(gradients_k, k_temp, learning_rate = 0.02)
        R_temp = utils.update_R(system.surface_mask, gradients_R, R_temp,0.01)

        #evaluate new fitness for reporting
        poisson, log, R_init, R_final = simulate_auxetic(R_temp,
                                                        k_temp,
                                                        system,
                                                        shift,
                                                        displacement)
        
        print(i, gradient_max,  poisson, energies.penalty_energy(R_init, system), utils.stiffness_penalty(system, k_temp))


        if output_evolution:
            #set evolution bits for the network
            R_evolution = R_evolution.at[i+1].set(R_temp)
            k_evolution = k_evolution.at[i+1].set(k_temp)
        
    onp.savez(str(run), R_temp = R_temp, k_temp = k_temp, perturbation = perturbation, connectivity = system.E,
             k_angle = k_angle, surface_nodes = system.surface_nodes, poisson = poisson, exit_flag = exit_flag)
    if output_evolution:
        evolution_log = {'position' : R_evolution, 'bond_strengths' : k_evolution}
        return poisson, exit_flag, R_temp, k_temp, system, shift, displacement, evolution_log
    else: 
        return poisson, exit_flag, R_temp, k_temp, system, shift, displacement

#@profile


def generate_auxetic_acoustic_adaptive(run, number_of_nodes_per_side, k_angle, perturbation, w_c, dw, poisson_target, opt_steps, output_evolution = False):

    """
    a combination version that uses a wrapper that implicitly combines scaled objectives

    
    run: run id, also used to as random seed
    poisson_target: the aimed value for the poisson ratio
    perturbation: absolute value of perturbation of the network for compression
    w_c: frequency_center
    dw: width of the bandgap
    """
    #parameters
    steps = 50
    write_every = 1
    delta_perturbation = 0.1
    nr_trials=500
    ageing_rate=0.1
    success_frac=0.05
    k_fit = 2.0/(dw**2) 
    system = utils.System(number_of_nodes_per_side, k_angle, run, 2.0, 0.35)
    system.initialize()
    system.acoustic_parameters(w_c, dw, nr_trials, ageing_rate, success_frac)
    system.auxetic_parameters(perturbation, delta_perturbation, steps, write_every)
    displacement = system.displacement
    shift = system.shift
    R = system.X
    k_bond = system.spring_constants
    
    #minimizing the initial configuration

    _, R ,_  = simulate_minimize_penalty(R,
                                        k_bond,
                                        system,
                                        shift,
                                        displacement)

    system.X= R
    displacement = system.displacement
    system.create_spring_constants()
    system.calculate_initial_angles_method(displacement)
    k_bond = system.spring_constants
    R_temp = R
    k_temp = k_bond

    if output_evolution:
        #set up evolution bits
        R_evolution = np.zeros((opt_steps, system.N, 2))
        R_evolution = R_evolution.at[0].set(R_temp)
        k_evolution = np.zeros((opt_steps, k_temp.shape[0], 1))
        k_evolution = k_evolution.at[0].set(k_temp)


    exit_flag = 0
    """
    0: max steps reached

    2: max k_temp exceeded
    3: converged
    
    """
    
    bandgap_contrast = 0
    
    result = forbidden_states_compression(R_temp, 
                                          k_temp, 
                                          system, 
                                          shift, 
                                          displacement)
    
    poisson = result.poisson
    poisson_bias = np.abs(poisson-poisson_target)  # distance bias - slower distance decline for larger difference.

    forbidden_states_init = result.forbidden_states_init
    forbidden_states_final = result.forbidden_states_final
    bandgap_bias = utils.gap_objective(result.frequency_init, system.frequency_center, k_fit) #this is used to define the radius of the fitness function.
    
    print('initial forbidden states: ', forbidden_states_init, bandgap_bias ) 
    
    # combination adaptive function
    adaptive_function = acoustic_auxetic_adaptive_wrapper(system, shift, displacement, k_fit, bandgap_bias, poisson_target, poisson_bias)
    
    grad_adaptive_R = jit(grad(adaptive_function, argnums=0))
    grad_adaptive_k = jit(grad(adaptive_function, argnums=1))

    print("Step", "max_grad", "bandgap_distance", "poisson_distance",  "forbidden_states_init" , "forbidden_states_init" , "poisson", "energy_penalty", "stiffness penalty")
    

    for i in range(opt_steps):
    
        gradients_R = grad_adaptive_R(R_temp, k_temp)
        gradients_k = grad_adaptive_k(R_temp, k_temp)

        gradient_max = np.max( np.abs( np.vstack((gradients_k, 
                                                  gradients_R.ravel()[:, np.newaxis] ))))
    
    
        
        
        k_temp = utils.update_kbonds(gradients_k, k_temp, learning_rate = 0.02)
        R_temp = utils.update_R(system.surface_mask, gradients_R, R_temp, 0.01)
    
        result = forbidden_states_compression(R_temp, k_temp, system, shift, displacement)

        #extract the progress
        poisson = result.poisson
        forbidden_states_init = result.forbidden_states_init #if this goes zero somehow that would break the optimization
        forbidden_states_final = result.forbidden_states_final
    
        #update distances
        fit_init = utils.gap_objective(result.frequency_init, system.frequency_center, k_fit)

        # Fitness energy for the final state
        fit_final = utils.gap_objective(result.frequency_final, system.frequency_center, k_fit)

        # Weighted objective function: Heavily weight the final state's energy

        poisson_distance = (result.poisson - poisson_target) / poisson_bias
        bandgap_distance =  (fit_final/bandgap_bias)**2 + (1- (fit_init) /bandgap_bias)**2 
    
        
        if np.abs(poisson_distance) < 0.02 and bandgap_distance < 0.05 : 
            print('converged')
            exit_flag = 3
            break
    
        
        print(i, gradient_max, bandgap_distance, poisson_distance, forbidden_states_init, forbidden_states_final, poisson, energies.penalty_energy(R_temp, system), utils.stiffness_penalty(system, k_temp))
        
        if output_evolution: 
            #set evolution bits for the network
            R_evolution = R_evolution.at[i+1].set(R_temp)
            k_evolution = k_evolution.at[i+1].set(k_temp)
   
    np.savez(str(run), 
             R_temp = R_temp, 
             k_temp = k_temp, 
             poisson = poisson, 
             poisson_target = poisson_target,
             perturbation = perturbation,
             connectivity = system.E,
             surface_nodes = system.surface_nodes,
             bandgap_distance = bandgap_distance, 
             forbidden_states_init = result.forbidden_states_init,
             forbidden_states_final = result.forbidden_states_final,
             exit_flag = exit_flag)
    if output_evolution:
        
        evolution_log = {'position' : R_evolution, 'bond_strengths' : k_evolution}
        return poisson_distance, bandgap_distance, exit_flag, R_temp, k_temp, system, shift, displacement, result, evolution_log
    else:
        return poisson_distance, bandgap_distance, exit_flag, R_temp, k_temp, system, shift, displacement, result


def generate_auxetic_acoustic_shift(run, number_of_nodes_per_side, k_angle, perturbation, frequency_closed, width_closed, frequency_opened, width_opened, poisson_target, opt_steps, output_evolution = False):

    """
    run: run id, also used to as random seed
    poisson_target: the aimed value for the poisson ratio
    perturbation: absolute value of perturbation of the network for compression
    frequency_closed : frequency of bandgap center being closed upon compression
    width_closed :  width of the bandgap being closed
    frequency_opened : frequency of bandgap center being opened upon compression
    width_opened : width of the bandgap being opened
    opt_steps: max steps attempted for optimization
    
    """
    #parameters
    steps = 50
    write_every = 1
    delta_perturbation = 0.1
    nr_trials=500
    ageing_rate=0.1
    success_frac=0.05
    k_fit = 2.0/(width_opened**2) 
    
    system = utils.System(number_of_nodes_per_side, k_angle, run, 2.0, 0.35)
    system.initialize()
    system.acoustic_parameters(frequency_opened, width_opened, nr_trials, ageing_rate, success_frac)
    system.auxetic_parameters(perturbation, delta_perturbation, steps, write_every)
    displacement = system.displacement
    shift = system.shift
    R = system.X
    k_bond = system.spring_constants
    
    #minimizing the initial configuration

    _, R ,_  = simulate_minimize_penalty(R,
                                        k_bond,
                                        system,
                                        shift,
                                        displacement)

    system.X= R
    displacement = system.displacement
    system.create_spring_constants()
    system.calculate_initial_angles_method(displacement)
    k_bond = system.spring_constants
    R_temp = R
    k_temp = k_bond

    if output_evolution:
        #set up evolution bits
        R_evolution = np.zeros((opt_steps, system.N, 2))
        R_evolution = R_evolution.at[0].set(R_temp)
        k_evolution = np.zeros((opt_steps, k_temp.shape[0], 1))
        k_evolution = k_evolution.at[0].set(k_temp)


    k_fit_closed = 2.0/(width_closed**2) 
    k_fit_opened = 2.0/(width_opened**2) 
    
    exit_flag=0
    
    """
    0: max steps reached
    2: max k_temp exceeded
    3: converged
    
    """
    
    bandgap_contrast = 0
    
    result = forbidden_states_compression(R_temp, 
                                          k_temp, 
                                          system, 
                                          shift, 
                                          displacement)
    
    poisson = result.poisson
    poisson_bias = np.abs(poisson-poisson_target)


    closed_contrast_ratio = utils.gap_objective(result.frequency_init, frequency_closed, k_fit_closed) / utils.gap_objective(result.frequency_final, frequency_closed, k_fit_closed) 
    opened_contrast_ratio = utils.gap_objective(result.frequency_final, frequency_opened, k_fit_opened) / utils.gap_objective(result.frequency_init, frequency_opened, k_fit_opened)

    #define distances
    poisson_distance = (poisson - poisson_target) / poisson_bias
    bandgap_distance = 0.5 * (closed_contrast_ratio + opened_contrast_ratio)

    
    print(" Contrasts:   Closed,   Opened" )
    print('initial : ', utils.gap_objective(result.frequency_init, frequency_closed, k_fit_closed), utils.gap_objective(result.frequency_init, frequency_opened, k_fit_opened)) 
    print('final   : ', utils.gap_objective(result.frequency_final, frequency_closed, k_fit_closed),  utils.gap_objective(result.frequency_final, frequency_opened, k_fit_opened))
    print("Step", "max_grad", "bandgap_distance", "poisson_distance",  "closed_contrast_ratio" , "opened_contrast_ratio" , "poisson", "energy_penalty", "stiffness penalty")
    # acoustic functions
    acoustic_function = acoustic_bandgap_shift_wrapper(system, shift, displacement, frequency_closed, width_closed, frequency_opened, width_opened)
    
    grad_acoustic_R = jit(grad(acoustic_function, argnums=0))
    grad_acoustic_k = jit(grad(acoustic_function, argnums=1))
        
    #auxetic_functions
    
    auxetic_function = simulate_auxetic_wrapper(R, k_bond, system,shift,displacement)
    grad_auxetic_R = jit(grad(auxetic_function, argnums=0))
    grad_auxetic_k = jit(grad(auxetic_function, argnums=1))
    

    for i in range(opt_steps):
    
        #acoustic gradients
        gradients_acoustic_k = grad_acoustic_k(R_temp, k_temp)
        gradients_acoustic_R = grad_acoustic_R(R_temp, k_temp)
    
        #auxetic gradients
        gradients_auxetic_k = grad_auxetic_k(R_temp, k_temp)
        gradients_auxetic_R = grad_auxetic_R(R_temp, k_temp)        
        
        #evaluate maximum gradients for diagnostics only
        gradient_max = np.max( np.abs( np.vstack((gradients_auxetic_k, 
                                                  gradients_auxetic_R.ravel()[:, np.newaxis], 
                                                  gradients_acoustic_k, 
                                                  gradients_acoustic_R.ravel()[:, np.newaxis] ))))

    
        #normalize gradients 
        gradients_auxetic_k = utils.normalize_gradients(gradients_auxetic_k)
        gradients_auxetic_R = utils.normalize_gradients(gradients_auxetic_R)
        
        gradients_acoustic_k = utils.normalize_gradients(gradients_acoustic_k)
        gradients_acoustic_R = utils.normalize_gradients(gradients_acoustic_R)


        #calculate weighted gradients
        gradients_k = poisson_distance*gradients_auxetic_k + bandgap_distance*gradients_acoustic_k
        gradients_R = poisson_distance*gradients_auxetic_R + bandgap_distance*gradients_acoustic_R
        
        
        k_temp = utils.update_kbonds(gradients_k, k_temp, learning_rate = 0.02)
        R_temp = utils.update_R(system.surface_mask, gradients_R, R_temp, 0.01)
    
        result = forbidden_states_compression(R_temp, k_temp, system, shift, displacement)
    
        #extract the progress
        poisson = result.poisson
    
        closed_contrast_ratio = utils.gap_objective(result.frequency_init, frequency_closed, k_fit_closed) / utils.gap_objective(result.frequency_final, frequency_closed, k_fit_closed) 
        opened_contrast_ratio = utils.gap_objective(result.frequency_final, frequency_opened, k_fit_opened) / utils.gap_objective(result.frequency_init, frequency_opened, k_fit_opened)
    
        #define distances
        poisson_distance = (poisson - poisson_target) / poisson_bias
        bandgap_distance = 0.5 * (closed_contrast_ratio + opened_contrast_ratio)
    
        
        if np.abs(poisson_distance) < 0.02 and bandgap_distance < 0.05: 
            print('converged')
            exit_flag = 3
            break
    
        
        print(i, gradient_max, bandgap_distance, poisson_distance,  closed_contrast_ratio, opened_contrast_ratio, poisson, energies.penalty_energy(R_temp, system), utils.stiffness_penalty(system, k_temp))

        if output_evolution:
            #set evolution bits for the network
            R_evolution = R_evolution.at[i+1].set(R_temp)
            k_evolution = k_evolution.at[i+1].set(k_temp)


    closed_contrasts = [utils.gap_objective(result.frequency_init, frequency_closed, k_fit_closed), utils.gap_objective(result.frequency_final, frequency_closed, k_fit_closed)]
    opened_contrasts = [utils.gap_objective(result.frequency_final, frequency_opened, k_fit_opened), utils.gap_objective(result.frequency_init, frequency_opened, k_fit_opened)]

    np.savez(str(run), 
             R_temp = R_temp, 
             k_temp = k_temp, 
             connectivity = system.E,
             surface_nodes = system.surface_nodes,
             poisson = poisson, 
             poisson_target = poisson_target,
             bandgap_distance = bandgap_distance, 
             closed_contrast_ratio = closed_contrast_ratio,
             opened_contrast_ratio = opened_contrast_ratio,
             closed_contrasts = closed_contrasts,
             opened_contrasts = opened_contrasts,
             forbidden_states_final = result.forbidden_states_final,
             exit_flag = exit_flag)

    if output_evolution:
        evolution_log = {'position' : R_evolution, 'bond_strengths' : k_evolution}
        return poisson_distance, bandgap_distance, exit_flag, R_temp, k_temp, system, shift, displacement, result , evolution_log
    else: 
        return poisson_distance, bandgap_distance, exit_flag, R_temp, k_temp, system, shift, displacement, result
