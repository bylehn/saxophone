# %%
import jax.numpy as np
import numpy as onp
import networkx as nx
from scipy.spatial import Delaunay
from jax import vmap, lax
from jax import jit
from jax import random  
from jax_md import quantity, space


# %%
class System:
    def __init__(self, nr_points, k_angle, random_seed, r_circle, dx):
        self.nr_points = nr_points
        self.random_seed = random_seed
        self.r_circle = r_circle
        self.dx = dx
        self.k_angles = k_angle

        # Other parameters...
        self.soft_sphere_sigma = 0.3
        self.soft_sphere_epsilon = 2.0
        self.crossing_penalty_threshold = 0.3
        self.crossing_penalty_strength = 2.0
        self.penalty_scale = 1e-5
        self.k_std_threshold = 1.0
        self.k_std_strength = 2.0
        
        # Initialize parameters
        self.initialize_parameters()

    def initialize_parameters(self):
        self.ageing_rate = 0.01
        self.success_fraction = 0.05
        self.perturbation = 1.
        self.delta_perturbation = 0.1
        self.steps = 50
        self.write_every = 1
        
        # Initialize JAX-specific attributes
        self.displacement, self.shift = space.free()

    def initialize_graph(self, X, E, L, surface_mask, degrees):
        self.X = X
        self.E = E
        self.L = L
        self.surface_mask = surface_mask
        self.N = X.shape[0]
        self.degrees = degrees
        self.get_surface_nodes()
        self.extract_surface_bond_mask()
        self.get_mass()
        self.create_spring_constants()
        self.calculate_angle_triplets_method()
        self.calculate_initial_angles_method(self.displacement)

    def get_surface_nodes(self):
        nodes = onp.arange(self.N)
        x_values = nodes % self.nr_points
        y_values = nodes // self.nr_points

        self.surface_nodes = {
            'top': nodes[y_values == (self.nr_points - 1)],
            'bottom': nodes[y_values == 0],
            'left': nodes[x_values == 0],
            'right': nodes[x_values == (self.nr_points - 1)]
        }

        # Create a boolean mask for all surface nodes
        self.surface_mask = onp.zeros(self.N, dtype=bool)
        for nodes in self.surface_nodes.values():
            self.surface_mask[nodes] = True

    def extract_surface_bond_mask(self):
        def is_surface_edge(edge):
            return self.surface_mask[edge[0]] and self.surface_mask[edge[1]]
        
        self.surface_bond_mask = onp.array([is_surface_edge(edge) for edge in self.E])
    def calculate_angle_triplets_method(self):
        self.angle_triplets = calculate_angle_triplets(self.E)

    def get_mass(self):
        m = np.ones(self.N)
        m2 = np.zeros(2 * self.N)
        m2 = m2.at[0:2*self.N:2].set(m)
        m2 = m2.at[1:2*self.N:2].set(m)
        self.mass = np.diag(m2)

    def create_spring_constants(self, k_1=1.0):
        displacements = self.X[self.E[:, 0], :] - self.X[self.E[:, 1], :]
        distances = np.linalg.norm(displacements, axis=1)
        self.spring_constants = (k_1 / distances).reshape(-1, 1)
        self.distances = distances


    def calculate_initial_angles_method(self, displacement_fn):
        from saxophone import utils  # Assuming calculate_initial_angles is defined in utils
        self.initial_angles = utils.calculate_initial_angles(self.X, self.angle_triplets, displacement_fn)

    def acoustic_parameters(self, frequency_center, frequency_width, nr_trials, ageing_rate=0.01, success_fraction=0.05):
        self.frequency_center = frequency_center
        self.frequency_width = frequency_width
        self.nr_trials = nr_trials
        self.ageing_rate = ageing_rate
        self.success_fraction = success_fraction

    def auxetic_parameters(self, perturbation, delta_perturbation, steps, write_every):
        self.perturbation = perturbation
        self.delta_perturbation = delta_perturbation
        self.steps = steps
        self.write_every = write_every

@jit
def fitness(poisson):
    """
    Constructs a fitness function based on the Poisson ratio.
    """
    return (poisson + 1)**2

@jit
def poisson_ratio(initial_horizontal, initial_vertical, final_horizontal, final_vertical):
    """
    Calculate the Poisson ratio based on average edge positions.

    initial_horizontal: initial horizontal edge positions
    initial_vertical: initial vertical edge positions
    final_horizontal: final horizontal edge positions
    final_vertical: final vertical edge positions

    output: Poisson ratio
    """

    delta_horizontal = final_horizontal - initial_horizontal
    delta_vertical = final_vertical - initial_vertical

    return -delta_vertical / delta_horizontal

def poisson_from_config(system, R_init, R_final):
    top_indices = system.surface_nodes['top']
    bottom_indices = system.surface_nodes['bottom']
    left_indices = system.surface_nodes['left']
    right_indices = system.surface_nodes['right']
    # Initial dimensions (before deformation)
    # Exclude the first and last index for horizontal edges (top and bottom)
    # as these are corners with the left and right edges
    initial_horizontal = onp.mean(R_init[right_indices[1:-1]], axis=0)[0] - onp.mean(R_init[left_indices[1:-1]], axis=0)[0]

    # Exclude the first and last index for vertical edges (left and right)
    # as these are corners with the top and bottom edges
    initial_vertical = onp.mean(R_init[top_indices[1:-1]], axis=0)[1] - onp.mean(R_init[bottom_indices[1:-1]], axis=0)[1]

    # Final dimensions (after deformation)
    final_horizontal = onp.mean(R_final[right_indices[1:-1]], axis=0)[0] - onp.mean(R_final[left_indices[1:-1]], axis=0)[0]
    final_vertical = onp.mean(R_final[top_indices[1:-1]], axis=0)[1] - onp.mean(R_final[bottom_indices[1:-1]], axis=0)[1]


    delta_horizontal = final_horizontal - initial_horizontal
    delta_vertical = final_vertical - initial_vertical
    # Calculate the poisson ratio.
    return [-delta_vertical / delta_horizontal , delta_horizontal, delta_vertical]


@jit
def update_kbonds(gradients, k_bond, learning_rate = 0.1, min_k = 0.05):
    """
    Updates spring constants based on gradients.

    gradients: gradients of the energy function
    k_bond: spring constants
    learning_rate: learning rate

    output: updated spring constants
    """
    gradients_perpendicular = gradients - np.mean(gradients)
    gradients_normalized = gradients_perpendicular / np.max(gradients_perpendicular)
    k_bond_new = min_k +  (k_bond - min_k) * (1 - learning_rate * gradients_normalized)

    return k_bond_new

@jit
def update_R(surface_mask, gradients, R_current, max_disp):
    """
    Updates positions based on gradients.
    """
    gradients_normalized = gradients / np.max(np.linalg.norm(gradients,axis=1))
    gradients_normalized *= np.transpose(np.tile(~surface_mask, (2,1)))
    R_updated = R_current - max_disp*gradients_normalized
    
    return R_updated

def remove_zero_rows(log_dict):
    """
    Remove rows (entries) in the log dictionary that are all zeros.
    """
    for key in log_dict:
        log_dict[key] = log_dict[key][~np.all(log_dict[key] == 0.0, axis=(1, 2))]
    return log_dict



def calculate_angle_triplets(E):
    I, J = onp.triu_indices(E.shape[0], k=1)
    mask1 = E[I, 0] == E[J, 0]
    ai1 = onp.stack([E[I[mask1], 1], E[I[mask1], 0], E[J[mask1], 1]], axis=-1)
    mask2 = E[I, 0] == E[J, 1]
    ai2 = onp.stack([E[I[mask2], 1], E[I[mask2], 0], E[J[mask2], 0]], axis=-1)
    mask3 = E[I, 1] == E[J, 0]
    ai3 = onp.stack([E[I[mask3], 0], E[I[mask3], 1], E[J[mask3], 1]], axis=-1)
    mask4 = E[I, 1] == E[J, 1]
    ai4 = onp.stack([E[I[mask4], 0], E[I[mask4], 1], E[J[mask4], 0]], axis=-1)

    return onp.concatenate([ai1, ai2, ai3, ai4], axis=0)


def compute_angle_between_triplet(displacement_fn, pi, pj, pk):
    """
    Computes the angle formed by three points.

    displacement_fn: Function to compute displacement.
    pi, pj, pk: Positions of the three points forming the angle.

    Returns: Angle in radians.
    """
    d_ij = displacement_fn(pi, pj)
    d_kj = displacement_fn(pk, pj)
    u_ij = d_ij / space.distance(d_ij)
    u_kj = d_kj / space.distance(d_kj)
    cos_theta = np.dot(u_ij, u_kj)
    return np.arccos(np.clip(cos_theta, -0.999, 0.999))


def calculate_initial_angles(positions,  angle_triplets_data, displacement_fn):
    """
    Calculates the initial angles for each triplet of nodes.

    positions: position matrix
    displacement_fn: displacement function
    E: edge matrix

    output: initial angles
    """

    def angle(triplet):
        i, j, k = triplet
        pi = np.take(positions, i, axis=0)
        pj = np.take(positions, j, axis=0)
        pk = np.take(positions, k, axis=0)
        return compute_angle_between_triplet(displacement_fn, pi, pj, pk)
    
    angles = vmap(angle)(angle_triplets_data)
    return angles

def is_hermitian(matrix):
    # Calculate the conjugate transpose of the matrix
    conjugate_transpose = np.conj(matrix).T
    
    # Check if the matrix is equal to its conjugate transpose
    return np.allclose(matrix, conjugate_transpose)

def gap_objective(frequency, frequency_center, k_fit):
    # Add a small epsilon to avoid division by zero
    epsilon = 1e-10
    
    # Clip the input to avoid extremely large values
    frequency = np.clip(frequency, -1e6, 1e6)
    
    result = np.sum(np.exp(-0.5 * k_fit * (frequency - frequency_center)**2))
    
    # If the result is zero, return a small positive value instead
    return np.maximum(result, epsilon)

def stiffness_penalty(system, k_bond):
    k_std_normalized = np.std(k_bond*system.distances.reshape(-1,1)) / system.k_std_threshold
    return system.k_std_strength / (1.0 + np.exp(-50.0 * (k_std_normalized - 1.0)))

def normalize_gradients(gradients):
    return gradients / np.max(np.linalg.norm(gradients,axis=1))
    

def create_delaunay_graph(system):
    key = random.PRNGKey(system.random_seed)

    xm, ym = onp.meshgrid(onp.arange(1, system.nr_points + 1), onp.arange(1, system.nr_points + 1))
    X = onp.vstack((xm.flatten(), ym.flatten())).T
    N = X.shape[0]

    surface_mask = onp.logical_or.reduce([
        X[:, 1] == system.nr_points,
        X[:, 1] == 1,
        X[:, 0] == system.nr_points,
        X[:, 0] == 1
    ])

    key, subkey = random.split(key)
    noise = system.dx * 2 * (0.5 - random.uniform(subkey, (N, 2)))
    X_np = onp.array(X, dtype=onp.float64)
    X_np[~surface_mask] += onp.array(noise[~surface_mask])

    DT = Delaunay(X_np)

    ET = onp.empty((0, 2), dtype=int)
    for T in DT.simplices:
        ET = onp.vstack((ET, [T[0], T[1]], [T[1], T[2]], [T[0], T[2]]))

    ET = onp.sort(ET)
    R = onp.linalg.norm(X_np[ET[:, 0], :] - X_np[ET[:, 1], :], axis=1)
    EN = ET[R < system.r_circle, :]
    A = onp.zeros((N, N))
    A[EN[:, 0], EN[:, 1]] = 1
    L = onp.linalg.norm(X_np[ET[:, 0], :] - X_np[ET[:, 1], :], axis=1)
    EL = L[R < system.r_circle]

    G = nx.Graph(A)
    E = onp.array(G.edges)
    L = onp.linalg.norm(X_np[E[:, 0], :] - X_np[E[:, 1], :], axis=1)

    return G, X_np, E, L, surface_mask
