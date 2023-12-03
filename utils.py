# %%
import jax.numpy as np
import numpy as onp
import networkx as nx
from scipy.spatial import Delaunay
from jax import vmap
from jax import jit
from jax.config import config; config.update("jax_enable_x64", True)
from jax_md import quantity, space


# %%
def createDelaunayGraph(NS, rseed, r_c, del_x):
    """
    This function creates a Delaunay graph of a set of points.

    NS: The number of points to generate.
    rseed: The random seed to use.
    r_c: The radius of the circumcircle of each edge in the graph.
    del_x: max noise magnitude from square lattice

    Returns:
    N: The number of points in the graph.
    G: The graph object.
    X: The coordinates of the points.
    E: The edges of the graph.
    """

    # Set the random seed.
    onp.random.seed(rseed)

    # Generate the points.
    xm, ym = onp.meshgrid(onp.arange(1, NS + 1), onp.arange(1, NS + 1))
    X = onp.vstack((xm.flatten(), ym.flatten())).T
    N = X.shape[0]

    # Add some noise to the points.
    X = X + del_x * 2 * (0.5 - onp.random.rand(N, 2))

    # Create the Delaunay triangulation.
    DT = Delaunay(X)

    # Get the edges of the triangulation.
    ET = onp.empty((0, 2), dtype=int)
    for T in DT.simplices:
        ET = onp.vstack((ET, [T[0], T[1]], [T[1], T[2]], [T[0], T[2]]))

    # Sort the edges.
    ET = onp.sort(ET)

    # Get the radii of the circumcircles of the edges.
    R = onp.linalg.norm(X[ET[:, 0], :] - X[ET[:, 1], :], axis=1)

    # Keep only the edges with radii less than r_c.
    EN = ET[R < r_c, :]

    # Create the adjacency matrix.
    A = onp.zeros((N, N))
    A[EN[:, 0], EN[:, 1]] = 1

    # Get the lengths of the edges.
    L = onp.linalg.norm(X[ET[:, 0], :] - X[ET[:, 1], :], axis=1)

    # Keep only the edges with lengths less than r_c.
    EL = L[R < r_c]

    # Create the graph object.
    G = nx.Graph(A)

    # Get the edges of the graph.
    E = onp.array(G.edges)

    # Get the lengths of the edges.
    L = onp.linalg.norm(X[E[:, 0], :] - X[E[:, 1], :], axis=1)

    return N, G, X, E, L

def getSurfaceNodes(G, NS):
    """
    Get the nodes on each surface of the graph.

    G: graph object
    NS: grid size

    output: dictionary with surface names as keys and node arrays as values
    """
    # Retrieve the list of nodes in the graph G
    nodes = np.array(list(G.nodes))
    # Calculate the x and y coordinates of the nodes based on the grid size NS
    x_values = nodes % NS
    y_values = nodes // NS
    # Find the nodes located on the top surface (y = NS - 1)
    top_nodes = nodes[y_values == NS - 1]
    # Find the nodes located on the bottom surface (y = 0)
    bottom_nodes = nodes[y_values == 0]
    # Find the nodes located on the left surface (x = 0)
    left_nodes = nodes[x_values == 0]
    # Find the nodes located on the right surface (x = NS - 1)
    right_nodes = nodes[x_values == NS - 1]
    # Return a dictionary with surface names as keys and node arrays as values
    return {
        'top': top_nodes,
        'bottom': bottom_nodes,
        'left': left_nodes,
        'right': right_nodes
    }


def make_box(R, padding):
    """
    Defines a box length

    R: position matrix
    padding: amount of space to add to the box

    output: box length
    """
    box_length = (np.max((np.max(R[:,0], R[:,1])) - np.min(((np.min(R[:,0], R[:,1])))))) + padding
    return box_length

def create_spring_constants(R,E,k_1):
    """
    Creates spring constants for each edge in the graph

    k_1: spring constant for a spring of unit length
    R: position matrix
    E: edge matrix

    output: spring constants, distances
    """
    displacements = R[E[:, 0],:] - R[E[:, 1], :]
    distance = np.linalg.norm(displacements, axis=1)
    return (k_1/distance).reshape(-1,1), distance

@jit
def compute_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points.
    """
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


#@jit
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

@jit
def update_kbonds(gradients, k_bond, learning_rate = 0.01):
    """
    Updates spring constants based on gradients.

    gradients: gradients of the energy function
    k_bond: spring constants
    learning_rate: learning rate

    output: updated spring constants
    """
    gradients_perpendicular = gradients - np.mean(gradients)
    gradients_normalized = gradients_perpendicular / np.max(gradients_perpendicular)
    k_bond_new = k_bond * (1 - learning_rate * gradients_normalized)

    return k_bond_new

@jit
def compute_force_norm(fire_state):
    return np.linalg.norm(fire_state.force)


def remove_zero_rows(log_dict):
    """
    Remove rows (entries) in the log dictionary that are all zeros.
    """
    for key in log_dict:
        log_dict[key] = log_dict[key][~np.all(log_dict[key] == 0.0, axis=(1, 2))]
    return log_dict



def calculate_angle_triplets(E):
    """
    Calculates the triplets of nodes that form angles.

    E: edge matrix

    output: triplets of nodes that form angles
    """
    I, J = onp.triu_indices(E.shape[0], k=1)
    mask1 = E[I, 0] == E[J, 0]
    ai1 = np.stack([E[I[mask1], 1], E[I[mask1], 0], E[J[mask1], 1]], axis=-1)
    mask2 = E[I, 0] == E[J, 1]
    ai2 = np.stack([E[I[mask2], 1], E[I[mask2], 0], E[J[mask2], 0]], axis=-1)
    mask3 = E[I, 1] == E[J, 0]
    ai3 = np.stack([E[I[mask3], 0], E[I[mask3], 1], E[J[mask3], 1]], axis=-1)
    mask4 = E[I, 1] == E[J, 1]
    ai4 = np.stack([E[I[mask4], 0], E[I[mask4], 1], E[J[mask4], 0]], axis=-1)

    return np.concatenate([ai1, ai2, ai3, ai4], axis=0)


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
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))


def calculate_initial_angles(positions, displacement_fn, E, angle_triplets_data):
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


def angle_energy(displacement_fn, k, theta_0, triplet, positions):
    """
    Calculates the harmonic angle energy for a triplet of nodes.

    displacement_fn: displacement function
    k: spring constants
    theta_0: equilibrium angles
    triplet: triplet of nodes
    positions: position matrix

    output: harmonic angle energy
    """
    i, j, k = triplet
    pi = np.take(positions, i, axis=0)
    pj = np.take(positions, j, axis=0)
    pk = np.take(positions, k, axis=0)
    theta = compute_angle_between_triplet(displacement_fn, pi, pj, pk)
    return 0.5 * k * (theta - theta_0)**2

# Assume angle_triplets is an array of shape (num_angles, 3)
# Each row in angle_triplets represents a set of indices (i, j, k)

# Vectorize the function
vectorized_angle_energy = vmap(angle_energy, in_axes=(None, None, 0, 0, None))

# Usage during simulation
#current_positions = ... # Update this during your simulation
#theta_0 = calculate_initial_angles(initial_positions, displacement_fn, E)
#total_angle_energy = np.sum(vectorized_angle_energy(displacement_fn, k, theta_0, angle_triplets_data, current_positions))
