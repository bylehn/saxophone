import jax.numpy as np
from jax import vmap
from saxophone.utils import compute_angle_between_triplet
from jax_md import quantity, util
from jax import jit
from jax_md import energy

def constrained_force_fn(R, energy_fn, mask):
    """
    Calculates forces with frozen edges.

    R: position matrix
    energy_fn: energy function
    mask: mask for frozen edges

    output: total force with frozen edges
    """


    def new_force_fn(R):
        force_fn = quantity.force(energy_fn)
        total_force = force_fn(R)
        total_force *= mask
        return total_force

    return new_force_fn

@jit
def compute_force_norm(fire_state):
    return np.linalg.norm(fire_state.force)

def angle_energy(system, triplets, displacement_fn, positions):
    """
    Calculates the harmonic angle energy for a triplet of nodes.

    displacement_fn: displacement function
    theta_0: equilibrium angles
    triplet: triplet of nodes
    positions: position matrix

    output: harmonic angle energy
    """
    def angle(triplet):
        i, j, k = triplet
        pi = np.take(positions, i, axis=0)
        pj = np.take(positions, j, axis=0)
        pk = np.take(positions, k, axis=0)
        return compute_angle_between_triplet(displacement_fn, pi, pj, pk)
    
    angles = vmap(angle)(triplets)


    bond_crossing_penalty = crossing_penalty_function (system, angles)
    return 0.5 * system.k_angles * ((angles - system.initial_angles)**2) + bond_crossing_penalty


def bond_crossing_penalty(system, triplets, displacement_fn, positions):
    """
    Calculates the harmonic angle energy for a triplet of nodes.

    displacement_fn: displacement function
    theta_0: equilibrium angles
    triplet: triplet of nodes
    positions: position matrix

    output: crossing penalty
    """
    def angle(triplet):
        i, j, k = triplet
        pi = np.take(positions, i, axis=0)
        pj = np.take(positions, j, axis=0)
        pk = np.take(positions, k, axis=0)
        return compute_angle_between_triplet(displacement_fn, pi, pj, pk)
    
    angles = vmap(angle)(triplets)

    bond_crossing_penalty = crossing_penalty_function (system, angles)
    
    return bond_crossing_penalty

def crossing_penalty_function (system, angles):
    """
    a version of this bias that uses soft sphere formalism to apply a constraint
    """
    # old version return system.crossing_penalty_strength / (1 + np.exp( system.crossing_penalty_steepness*( angles - system.crossing_penalty_threshold ) ) )
    da = angles / system.crossing_penalty_threshold
    soft_fn = lambda da: system.crossing_penalty_strength / 2 * (util.f32(1.0) - da) ** 2 
    sigmoid_fn = lambda da: 1.0 / (1.0+ np.exp( 50.0*( da - 1.0 ) ) )

    soft_potential = np.where(da < 1.0, soft_fn(da), util.f32(0.0)) + 1e-3
    #np.where(da < 1.0, fn(da), util.f32(0.0)) 
    sigmoid_function = sigmoid_fn(da)
    
    return soft_potential * sigmoid_function
    
# Assume angle_triplets is an array of shape (num_angles, 3)
# Each row in angle_triplets represents a set of indices (i, j, k)

# Vectorize the function
#vectorized_angle_energy = vmap(angle_energy, in_axes=(None, 0, None , None))

# Usage during simulation
#current_positions = ... # Update this during your simulation
#theta_0 = calculate_initial_angles(initial_positions, displacement_fn, E)
#total_angle_energy = np.sum(vectorized_angle_energy(displacement_fn, k, theta_0, angle_triplets_data, current_positions))

def penalty_energy(R, system, **kwargs):
    """
    provides per node penalty energy
    """
    displacement = system.displacement
    crossing_penalty = np.sum(bond_crossing_penalty(system, system.angle_triplets, displacement, R))
    # Bond energy (assuming that simple_spring_bond is JAX-compatible)
    node_energy = energy.soft_sphere_pair(displacement, sigma = system.soft_sphere_sigma, epsilon= system.soft_sphere_epsilon)(R, **kwargs)

    return (crossing_penalty + node_energy)/system.N
