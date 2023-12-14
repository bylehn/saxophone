import jax.numpy as np
import numpy as onp
from jax.config import config; config.update("jax_enable_x64", True)
from jax_md import energy, minimize
from jax import jit, vmap
from jax import lax
import networkx as nx
import utils
import energies
from collections import namedtuple

Result_forbidden_modes = namedtuple('Result', [
    'D_init',
    'V_init',
    'forbidden_states_init',
    'R_init',
    'D_final',
    'V_final',
    'forbidden_states_final',
    'R_final',
    'log'
])



def simulate_auxetic(R,
                     k_bond,
                     system,
                     shift,
                     displacement
                     ):
    """
    Simulates the auxetic process using a System instance.

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

    # Get the surface nodes.
    top_indices = system.surface_nodes['top']
    bottom_indices = system.surface_nodes['bottom']
    left_indices = system.surface_nodes['left']
    right_indices = system.surface_nodes['right']
    mask = np.ones(R.shape)
    mask = mask.at[left_indices].set(0)
    mask = mask.at[right_indices].set(0)
    num_iterations = int(np.ceil(system.perturbation / system.delta_perturbation))
    # Initialize the cumulative perturbation
    cumulative_perturbation = 0.0

    log = {
            'force': np.zeros((num_iterations*(system.steps // system.write_every),) + R.shape),
            'position': np.zeros((num_iterations*(system.steps // system.write_every),) + R.shape),
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
        fire_init, fire_apply = minimize.fire_descent(force_fn, shift)
        fire_state = fire_init(R_perturbed)

        # Update step function generator with the new start index

        start_idx = i * (system.steps // system.write_every)

        step_fn = step_fn_generator(fire_apply, start_idx)

        # Perform the minimization step
        fire_state, log = lax.fori_loop(0, system.steps, step_fn, (fire_state, log))
        R_perturbed = fire_state.position

        return R_perturbed, log, cumulative_perturbation

    def energy_fn(R, system, **kwargs):
        angle_energy = np.sum(energies.vectorized_angle_energy(system, system.angle_triplets, displacement, R))
        # Bond energy (assuming that simple_spring_bond is JAX-compatible)
        bond_energy = energy.simple_spring_bond(displacement, system.E, length=system.L, epsilon=k_bond[:, 0])(R, **kwargs)

        return bond_energy + angle_energy

    def energy_fn_wrapper(R, **kwargs):
        return energy_fn(R, system, **kwargs)

    R_init = R
    # Initial dimensions (before deformation)
    # Exclude the first and last index for horizontal edges (top and bottom)
    # as these are corners with the left and right edges
    initial_horizontal = np.mean(R_init[right_indices[1:-1]], axis=0)[0] - np.mean(R_init[left_indices[1:-1]], axis=0)[0]

    # Exclude the first and last index for vertical edges (left and right)
    # as these are corners with the top and bottom edges
    initial_vertical = np.mean(R_init[top_indices[1:-1]], axis=0)[1] - np.mean(R_init[bottom_indices[1:-1]], axis=0)[1]

    R_final, log, cumulative_perturbation = lax.fori_loop(0, num_iterations, perturb_and_minimize, (R_init, log, cumulative_perturbation))
    # Final dimensions (after deformation)
    final_horizontal = np.mean(R_final[right_indices[1:-1]], axis=0)[0] - np.mean(R_final[left_indices[1:-1]], axis=0)[0]
    final_vertical = np.mean(R_final[top_indices[1:-1]], axis=0)[1] - np.mean(R_final[bottom_indices[1:-1]], axis=0)[1]

    # Calculate the poisson ratio.
    poisson = utils.poisson_ratio(initial_horizontal, initial_vertical, final_horizontal, final_vertical)
    #fit = fitness(poisson)

    return poisson, log, R_init, R_final

def simulate_auxetic_optimize(R,
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
    poisson, _, _, _ = simulate_auxetic(R, k_bond, system, shift, displacement)

    return poisson



def get_bond_importance(C, V, D, D_range):
    modes = np.where((D > D_range[0]) & (D < D_range[1]))[0]
    delta_E = C.T@V
    EC = delta_E[:, modes]
    bond_importance = np.mean(np.abs(EC), axis=1)
    bond_importance = bond_importance/np.max(bond_importance)
    bond_importance_centered = bond_importance - np.mean(bond_importance)
    bond_importance_normalized = bond_importance_centered/np.max(np.abs(bond_importance_centered))
    
    return bond_importance_normalized.reshape(-1,1)

def create_compatibility(system, R):
    N_b = system.E.shape[0]
    mdict = dict(zip(range(system.N), system.m))
    nx.set_node_attributes(system.G, mdict, 'Mass')

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
    D, V = onp.linalg.eig(DMAT)
    D = np.real(D)
    frequency = np.sqrt(np.abs(D))
    forbidden_states = np.sum(onp.logical_and(frequency > system.frequency_center - system.frequency_width/2,
                                              frequency < system.frequency_center + system.frequency_width/2))
    V = np.real(V)
    return D, V, forbidden_states

def ageSprings(k_old,X,C,V,D,D_range,ageing_rate):
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
    bond_importance=getBondImportance(X,C,V,D,D_range)
    k_new=onp.multiply(k_old,(1+2*ageing_rate*bond_importance))
    return k_new

def optimizeAgeing(C, k, M, w_c, dw, N_trials,ageing_rate,success_frac):
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

    w_range=[w_c-dw/2,w_c+dw/2]
    D_range = [x**2 for x in w_range]
    D, V, forbidden_states_initial = getForbiddenModes(C, k, M, w_c, dw)
    if forbidden_states_initial==0:
        return k,1,0
    for trial in range(1, N_trials+1):

        k=ageSprings(k,X,C,V,D,D_range,ageing_rate)

        D, V, forbidden_states = getForbiddenModes(C, k, M, w_c, dw)
        print(trial,forbidden_states)

        if forbidden_states<=success_frac*forbidden_states_initial:

            return k, 1,trial

    return k,0,trial

def age_springs_compressed(k_old, system, result, C_init, C_final, D_range):
    """
    Aging algorithm for springs when doing compression.

    Returns:
    k_new: new spring constant matrix
    """
    bond_importance_init = scale_bond_importance(get_bond_importance(C_init, result.V_init, result.D_init, D_range))
    bond_importance_final = scale_bond_importance(get_bond_importance(C_final, result.V_final, result.D_final, D_range))

    bond_importance_difference = bond_importance_final#-bond_importance_init

    k_new = k_old * (1 + 2*system.ageing_rate * bond_importance_difference)
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

def forbidden_states_compression(R,
                                 k_bond,
                                 system,
                                 shift,
                                 displacement,
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
    """
    _, log, R_init, R_final = simulate_auxetic(R,
                                               k_bond,
                                               system,
                                               shift,
                                               displacement
                                               )
    
    C_init = create_compatibility(system, R_init)
    C_final = create_compatibility(system, R_final)
    D_init, V_init, forbidden_states_init = get_forbidden_states(C_init, k_bond, system)
    D_final, V_final, forbidden_states_final = get_forbidden_states(C_final, k_bond, system)

    return Result_forbidden_modes(D_init,
                                  V_init,
                                  forbidden_states_init,
                                  R_init,
                                  D_final,
                                  V_final,
                                  forbidden_states_final,
                                  R_final,
                                  log
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

    result = forbidden_states_compression(R,
                                          k_bond,
                                          system,
                                          shift,
                                          displacement
    )
    
    forbidden_states_init_0 = result.forbidden_states_init
    forbidden_states_final_0 = result.forbidden_states_final
    if forbidden_states_init_0 * forbidden_states_final_0 == 0:
        return k_bond, 1, 0
    
    for trial in range(1, system.nr_trials+1):

        result = forbidden_states_compression(R,
                                              k_bond,
                                              system,
                                              shift,
                                              displacement)
        
        C_init = create_compatibility(system, result.R_init)
        C_final = create_compatibility(system, result.R_final)
        k_bond = age_springs_compressed(k_bond, system, result, C_init, C_final, D_range)

        result = forbidden_states_compression(R,
                                              k_bond,
                                              system,
                                              shift,
                                              displacement
        )

        print(trial, result.forbidden_states_init, result.forbidden_states_final)

        if result.forbidden_states_final <= system.success_fraction*forbidden_states_final_0:
            return k_bond, 1, trial

    return k_bond, 0, trial
