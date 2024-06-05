# %%
import jax.numpy as np
import numpy as onp
import networkx as nx
from scipy.spatial import Delaunay
from jax import vmap
from jax import jit
from jax import random  
from jax.config import config; config.update("jax_enable_x64", True)
from jax_md import quantity, space


# %%
class System:
    def __init__(self, nr_points, k_angle, random_seed, r_circle, dx):
        self.nr_points = nr_points
        self.random_seed = random_seed
        self.r_circle = r_circle
        self.dx = dx
        self.k_angle = k_angle


        #penalty parameters: node and bond cross: 
        
        #node energy attrbutes 
        self.soft_sphere_sigma = 0.3 #diameter
        self.soft_sphere_epsilon = 2.0 

        #crossing penalty attributes
        self.crossing_penalty_strength = 2.0 # epsilon of soft angle strength
        self.crossing_penalty_threshold = 0.3 #radians

        self.penalty_scale = 1e-5 #per node penalty energy that scales to 1 unit in objective functions
       
        
        # Initialize attributes
        self.N = None
        self.G = None
        self.X = None
        self.E = None
        self.L = None
        self.surface_nodes = None  
        self.surface_mask = None
        self.surface_bond_mask = None
        self.mass = None
        self.spring_constants = None
        self.distances = None
        self.angle_triplets = None
        self.initial_angles = None
        self.displacement = None
        self.shift = None
        # Acoustic properties
        self.m = None
        self.frequency_center = None
        self.frequency_width = None
        self.ageing_rate = 0.01
        self.success_fraction = 0.05
        self.nr_trials = None
        self.degrees = None
        # Auxetic properties
        self.perturbation = 1.
        self.delta_perturbation = 0.1
        self.steps = 50
        self.write_every = 1


# %%

    def initialize(self):
        """
        Initializes the system by setting up the graph, calculating necessary properties,
        and preparing the system for simulation.
        """
        self.create_delaunay_graph()
        R = self.X
        displacement, shift = space.free()
        self.displacement = displacement
        self.shift = shift
        self.get_mass()
        self.m = np.ones(self.N)
        self.create_spring_constants()
        self.calculate_angle_triplets_method()
        self.calculate_initial_angles_method(displacement)
        self.k_angles = (2*self.k_angle/ ( self.degrees* (self.degrees-1) ) ) [self.angle_triplets[:,1]]  #equal distribution of angle stiffness to all triplets centred on a node 
        self.get_surface_nodes()
        
    def auxetic_parameters(self, perturbation=1., delta_perturbation=0.1, steps=50, write_every=1):
        """
        Sets the parameters for the auxetic simulation.
        """
        self.perturbation = perturbation
        self.delta_perturbation = delta_perturbation
        self.steps = steps
        self.write_every = write_every

    def acoustic_parameters(self, frequency_center, frequency_width, nr_trials, ageing_rate=0.01, success_fraction=0.05):
        """
        Sets the parameters for the acoustic simulation.
        """
        self.frequency_center = frequency_center
        self.frequency_width = frequency_width
        self.nr_trials = nr_trials
        self.ageing_rate = ageing_rate
        self.success_fraction = success_fraction

    def create_delaunay_graph(self):
        # Initialize JAX PRNGKey
        key = random.PRNGKey(self.random_seed)

        # Generate the points
        xm, ym = np.meshgrid(np.arange(1, self.nr_points + 1), np.arange(1, self.nr_points + 1))
        X = np.vstack((xm.flatten(), ym.flatten())).T
        N = X.shape[0]

        # Determine surface nodes
        surface_mask = onp.logical_or.reduce([
            X[:, 1] == self.nr_points,
            X[:, 1] == 1,
            X[:, 0] == self.nr_points,
            X[:, 0] == 1
        ])
    
        # Split the key for the next random operation
        key, subkey = random.split(key)
    
        # Add noise to the non-surface points
        noise = self.dx * 2 * (0.5 - random.uniform(subkey, (N, 2)))
        
        # Convert to numpy array for Delaunay triangulation
        X_np = onp.array(X, dtype = np.float64)

        X_np[~surface_mask] += noise[~surface_mask]

        # Create the Delaunay triangulation
        DT = Delaunay(X_np)

        # Process the edges
        ET = onp.empty((0, 2), dtype=int)
        for T in DT.simplices:
            ET = onp.vstack((ET, [T[0], T[1]], [T[1], T[2]], [T[0], T[2]]))

        ET = onp.sort(ET)

        # Calculate edge radii and lengths
        R = onp.linalg.norm(X_np[ET[:, 0], :] - X_np[ET[:, 1], :], axis=1)
        EN = ET[R < self.r_circle, :]
        A = onp.zeros((N, N))
        A[EN[:, 0], EN[:, 1]] = 1
        L = onp.linalg.norm(X_np[ET[:, 0], :] - X_np[ET[:, 1], :], axis=1)
        EL = L[R < self.r_circle]

        # Create the graph object
        G = nx.Graph(A)
        E = onp.array(G.edges)
        L = onp.linalg.norm(X_np[E[:, 0], :] - X_np[E[:, 1], :], axis=1)

        # Store results as attributes
        self.N = N
        self.G = G
        self.X = X_np
        self.E = E
        self.L = L
        self.surface_mask = surface_mask
        self.get_surface_nodes()
        self.extract_surface_bond_mask()
        self.get_mass()
        self.degrees  = np.array([val for (node, val) in G.degree()])
        
    def get_surface_nodes(self):
        """
        Get the nodes on each surface of the graph and store them in self.surface_nodes.

        Output: Updates self.surface_nodes with a dictionary containing surface nodes.
        """
        if self.G is None:
            raise ValueError("Graph not created. Call createDelaunayGraph first.")

        nodes = onp.array(list(self.G.nodes))
        x_values = nodes % self.nr_points
        y_values = nodes // self.nr_points

        top_nodes = nodes[y_values == self.nr_points - 1]
        bottom_nodes = nodes[y_values == 0]
        left_nodes = nodes[x_values == 0]
        right_nodes = nodes[x_values == self.nr_points - 1]

        # Store the result in self.surface_nodes
        self.surface_nodes = {
            'top': top_nodes,
            'bottom': bottom_nodes,
            'left': left_nodes,
            'right': right_nodes
        }
    def extract_surface_bond_mask(self):
        """
        Extract a mask indicating all the bonds (edges) between the surface nodes.
        """
        # Ensure surface nodes are calculated
        if not hasattr(self, 'surface_nodes'):
            self.get_surface_nodes()
    
        # Combine all surface nodes into a single set
        surface_nodes_set = set()
        for key in self.surface_nodes:
            surface_nodes_set.update(self.surface_nodes[key])
    
        # Create a boolean mask for edges between surface nodes
        self.surface_bond_mask = np.array([edge[0] in surface_nodes_set and edge[1] in surface_nodes_set for edge in self.E])
    
        

    def get_mass(self):
        """
        Calculate the mass matrix and store it in self.M.
        """
        if self.G is None or self.N is None:
            raise ValueError("Graph not created. Call createDelaunayGraph first.")

        m = onp.ones(self.N)
        mdict = dict(zip(range(self.N), m))
        nx.set_node_attributes(self.G, mdict, 'Mass')

        m2 = onp.zeros(2 * self.N)
        m2[0:2 * self.N:2] = m
        m2[1:2 * self.N:2] = m
        self.mass = onp.diag(m2)

    def create_spring_constants(self, k_1=1.0):
        """
        Creates spring constants for each edge in the graph based on current state.

        Output: Updates self.spring_constants and self.distances.
        """
        if self.X is None or self.E is None:
            raise ValueError("Graph properties not set. Call createDelaunayGraph first.")

        displacements = self.X[self.E[:, 0], :] - self.X[self.E[:, 1], :]
        distances = np.linalg.norm(displacements, axis=1)
        self.spring_constants = (k_1 / distances).reshape(-1, 1)
        self.distances = distances

    @jit
    def compute_distance(point1, point2):
        """
        Calculate the Euclidean distance between two points.
        """
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def calculate_angle_triplets_method(self):
        """
        Wrapper method to calculate the triplets of nodes that form angles.
        """
        self.angle_triplets = calculate_angle_triplets(self.E)  

    def calculate_initial_angles_method(self, displacement_fn):
        """
        Wrapper method to calculate the initial angles for each triplet of nodes.
        """

        self.initial_angles = calculate_initial_angles(self.X,  self.angle_triplets, displacement_fn)

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
    """
    Calculates the triplets of nodes that form angles.

    Note: this considers ALL angles at a node, not just the planar subset. This is consistent with NOMM paper.

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
    
    return np.sum(np.exp(-0.5*k_fit * (frequency - frequency_center)**2))

def normalize_gradients(gradients):
    return gradients / np.max(np.linalg.norm(gradients,axis=1))


