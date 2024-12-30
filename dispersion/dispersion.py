import numpy as np
import scipy.linalg as scla
import scipy.sparse as spa
import copy

class GetBranchedBro:
    def __init__(self, L):
        self.L = L
        
    def periodic_distance(self, d):
        """
        Adjusts distances according to periodic boundary conditions
        d: distance vector in 2D
        """
        for dims in range(2):  # For each dimension (x and y)
            if np.abs(d[dims]) < self.L[dims]/2:
                pass  # Distance is already minimal
            else:
                # Adjust distance using periodic boundaries
                if d[dims] > 0:
                    d[dims] -= self.L[dims]/2
                else:
                    d[dims] += self.L[dims]/2
        return d
        
    def FT_compatability(self, C, N, X, E, q):
        """
        Computes Fourier transform of compatibility matrix
        C: Compatibility matrix
        N: Number of nodes
        X: Node positions
        E: Edges (connections between nodes)
        q: Wave vector
        """
        C_ft = np.array(copy.deepcopy(C), dtype=np.complex128)
        for ei, (u, v) in enumerate(E):  # For each edge
            R_u = X[u]  # Position of first node
            b_vec = self.periodic_distance(X[v, :] - X[u, :])  # Vector between nodes
            R_v = R_u + b_vec  # Position of second node
            R_uv = (R_u + 0.5*b_vec)  # Midpoint of the edge
            
            # Apply phase factors
            for d in range(2):
                C_ft[2*v+d, ei] *= np.exp(-1j*q@(R_v - R_uv))
                C_ft[2*u+d, ei] *= np.exp(-1j*q@(R_u - R_uv))
        return C_ft

    def getDV(self, Cq, k, M):
        """
        Calculates eigenvalues and eigenvectors of dynamical matrix
        Cq: Fourier transformed compatibility matrix
        k: Spring constants
        M: Mass matrix
        """
        kd = np.diag(np.squeeze(k))  # Diagonal matrix of spring constants
        Kq = Cq @ kd @ np.conj(Cq.T)  # Stiffness matrix in Fourier space
        DMAT = scla.inv(M) @ Kq  # Dynamical matrix
        
        # Calculate eigenvalues and eigenvectors
        if scla.ishermitian(DMAT):
            D, V = scla.eigh(DMAT)  # Use specialized routine for Hermitian matrices
        else:
            D, V = scla.eig(DMAT)
            D = np.sort(D)
        V = np.real(V)
        return D, V

    def create_q_path(self, qx_i, qx_f, qy_i, qy_f, num_points):
        num_points = num_points if num_points % 2 != 0 else num_points + 1
        q_x = np.linspace(qx_i, qx_f, num_points)
        q_y = np.linspace(qy_i, qy_f, num_points)
        return np.column_stack((q_x, q_y))

    def get_dispersion(self, C, k, M, N, X, E, q_values):
        """
        Calculates dispersion relation for given q-points
        Returns frequencies for each band at each q-point
        """
        w = [[] for _ in range(2*N)]  # Initialize frequency arrays
        for q in q_values:
            Cq = self.FT_compatability(C, N, X, E, q)
            D, V = self.getDV(Cq, k, M)
            for band in range(2*N):
                w[band].append(np.sqrt(np.abs(np.real(D[band]))))
        return np.array(w).T 
