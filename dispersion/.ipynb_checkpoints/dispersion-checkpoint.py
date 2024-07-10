import numpy as np
import scipy.linalg as scla
import scipy.sparse as spa
import copy

class GetBranchedBro:
    def __init__(self, L):
        self.L=L
        
    def periodic_distance(self, d):
        for dims in range(2):
            if np.abs(d[dims]) < self.L/2:
                pass
            else:
                if d[dims]>0:
                    d[dims] -= self.L/2
                else:
                    d[dims] += self.L/2
        return d
        
    def FT_compatability(self, C, N, X, E, q):
        C_ft = np.array(copy.deepcopy(C), dtype=np.complex128)
        for ei, (u, v) in enumerate(E):
            R_u = X[u]
            b_vec = self.periodic_distance(X[v, :] - X[u, :])
            R_v = R_u + b_vec
            R_uv = (R_u + b_vec)
            for d in range(2):
                C_ft[2*v+d, ei] *= np.exp(-1j*q@(R_v - R_uv)*((-1) ** d))
                C_ft[2*u+d, ei] *= np.exp(-1j*q@(R_u - R_uv)*((-1) ** d))
        return C_ft

    def getDV(self, Cq, k, M):
        kd = np.diag(np.squeeze(k))
        Kq = Cq @ kd @ np.conj(Cq.T)
        DMAT = scla.inv(M) @ Kq
        if scla.ishermitian(DMAT) == True:
            D, V = scla.eigh(DMAT)
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
        w = [[] for _ in range(2*N)]
        for q in q_values:
            Cq = self.FT_compatability(C, N, X, E, q)
            D, V = self.getDV(Cq, k, M)
            for band in np.arange(0,2*N):
                w[band].append(np.sqrt(np.abs(np.real(D[band]))))
        w = np.array(w).T
        return w 