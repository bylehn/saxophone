import jax.numpy as np
from jax import vmap
from saxophone.utils import compute_angle_between_triplet
from jax_md import quantity
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
    k: spring constants
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


    crossing_penalty = system.crossing_penalty_strength / (1 + np.exp( system.crossing_penalty_steepness*( angles - system.crossing_penalty_threshold ) ) )
    return 0.5 * system.k_angle * (angles - system.initial_angles)**2 + crossing_penalty

# Assume angle_triplets is an array of shape (num_angles, 3)
# Each row in angle_triplets represents a set of indices (i, j, k)

# Vectorize the function
#vectorized_angle_energy = vmap(angle_energy, in_axes=(None, 0, None , None))

# Usage during simulation
#current_positions = ... # Update this during your simulation
#theta_0 = calculate_initial_angles(initial_positions, displacement_fn, E)
#total_angle_energy = np.sum(vectorized_angle_energy(displacement_fn, k, theta_0, angle_triplets_data, current_positions))

def test_energy_fn(R, k_bond, system, **kwargs):
        displacement = system.displacement
        angular_energy = np.sum(angle_energy(system, system.angle_triplets, displacement, R))
        # Bond energy (assuming that simple_spring_bond is JAX-compatible)
        bond_energy = energy.simple_spring_bond(displacement, system.E, length=system.distances, epsilon=k_bond[:, 0])(R, **kwargs)
        node_energy = energy.soft_sphere_pair(displacement, sigma = system.soft_sphere_sigma, epsilon= system.soft_sphere_epsilon)(R, **kwargs)

        return bond_energy + angular_energy + node_energy
