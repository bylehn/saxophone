# -*- coding: utf-8 -*-
"""
eigenvalue solver for phononics
"""
import time
import numpy as np
from matplotlib  import pyplot as plt
from scipy.spatial import Delaunay
import networkx as nx
from matplotlib import rc
from copy import deepcopy
import matplotlib.gridspec as gridspec

rc('text', usetex=True)
rc('font', size=16)
rc('legend', fontsize=13)
rc('text.latex', preamble=r'\usepackage{cmbright}')

def removeBond(i, C, k, G, E):
    """
    Remove a bond (edge) from the adjacency matrix and check if the graph remains connected.
    Args:
        i (int): Index of the bond to be removed.
        C (ndarray): 2N x Nb matrix containing bond vectors for each edge.
        k (ndarray): Nb x 1 array containing bond weights.
        G (graph): Graph object containing the edges and nodes.

    Returns:
        C_n (ndarray): 2N x Nb-1 matrix with the ith column removed.
        k_n (ndarray): Nb-1 x 1 array with the ith element removed.
        c_flag (bool): Flag indicating if the graph remains connected after removing the bond.
    """
    C_n = np.delete(C, i, axis=1)
    k_n = np.delete(k, i, axis=0)
    G_n=deepcopy(G)
    G_n.remove_edge(E[i,0],E[i,1])

    nc = nx.number_connected_components(G_n)
   

    return C_n, k_n, nc



def getObjective(C, k, M, kappa, w_c, dw):
    kd = np.diag(np.squeeze(k))
    K = C @ kd @ C.T
    DMAT = np.linalg.solve(M, K)
    D, V = np.linalg.eig(DMAT)
    d = np.real(D)
    F = np.sum(np.exp(-kappa * ((d - w_c) ** 2)))
    V=np.real(V)
    return F, d, V


def getModes(C, k, M, kappa, w_c, dw):
    kd = np.diag(np.squeeze(k))
    K = C @ kd @ C.T
    DMAT = np.linalg.solve(M, K)
    D, V = np.linalg.eig(DMAT)
    d = np.real(D)
    F = np.sum(np.exp(-kappa * ((d - w_c) ** 2)))
    V=np.real(V)
    return F, d, V

def plotmodeM(G, X, V, mode):
    N = G.number_of_nodes()
    V=np.real(V)
    X_ = X[:, 0]
    Y_ = X[:, 1]
    dx = V[0:2 * N:2, mode]
    dy = V[1:2 * N:2, mode]
    m=np.fromiter(nx.get_node_attributes(G,'Mass').values(),dtype=float)
    MSizes =10 * m / np.max(m)
    # print(MSizes)
    LWidths = 2
    nx.draw(G, pos=X, node_size=20*m, width=LWidths, node_color='k', edge_color='k')
    for i in range(N):
        plt.arrow(X_[i], Y_[i], dx[i], dy[i], head_width=0.15, head_length=0.15, fc='r', ec='r')
    # plt.axis('equal')
    plotlimits = [min(X_.min(), Y_.min()) - 2, max(X_.max(), Y_.max()) + 2]
    plt.xlim(plotlimits)
    plt.ylim(plotlimits)
    plt.axis('off')


def plotmodeD(G,X,V,mode):
    X_ = X[:, 0]
    Y_ = X[:, 1]
    X2=X+np.stack((V[0:2*N:2,mode],V[1:2*N:2,mode])).T
    pos2 = dict(zip(range(G.number_of_nodes()), X2))
    nx.draw(G, pos2, node_size=10, node_color="blue")
    
    plotlimits = [min(X_.min(), Y_.min()) - 2, max(X_.max(), Y_.max()) + 2]
    plt.xlim(plotlimits)
    plt.ylim(plotlimits)
    
def plotmodeE(G,X,C,V,mode):
    X_ = X[:, 0]
    Y_ = X[:, 1]
    delta_E=C.T@V
    EC=delta_E[:,mode]
    pos = dict(zip(range(G.number_of_nodes()), X))
    edges=nx.draw_networkx_edges(G, pos, edge_color=EC,edge_cmap=plt.cm.bwr)
    plt.colorbar(edges)
    nx.draw_networkx_nodes(G, pos, node_size=10, node_color='k',alpha=1)
    
    plotlimits = [min(X_.min(), Y_.min()) - 2, max(X_.max(), Y_.max()) + 2]
    plt.axis('equal')
    plt.xlim(plotlimits)
    plt.ylim(plotlimits)
    
def multiplots(G,X,V,d,d_range):
    modes = np.where((d > d_range[0]) & (d < d_range[1]))[0]
    N_modes = len(modes)
    Nrows = 4
    Ncols = np.ceil(N_modes / Nrows).astype(int)
    
    # Plotting modes with colored nodes and arrows
    fig1 = plt.figure(figsize=(10, 10))
    gs1 = gridspec.GridSpec(Ncols, Nrows, figure=fig1, wspace=0.2, hspace=0.2)
    for i, mode in enumerate(modes):
        ax = plt.subplot(gs1[i])
        plotmodeM(G, X,V, mode)
        ax.set_aspect('equal')

    fig2 = plt.figure(figsize=(10, 10))
    gs2 = gridspec.GridSpec(Ncols, Nrows, figure=fig2, wspace=0.2, hspace=0.2)
    for i, mode in enumerate(modes):
        ax = plt.subplot(gs2[i])
        plotmodeD(G, X, V, mode)
        ax.set_aspect('equal')
        
def plotmodeC(G, X, V, mode):
    N = X.shape[0]
    cmap = plt.cm.hsv(np.linspace(0, 1, 360))
    alpha = np.arctan2(V[1:2*N:2, mode], V[0:2*N:2, mode])
    idx = np.ceil(np.rad2deg(np.mod(alpha, 2*np.pi)))
    idx[idx == 0] = 360
    m=np.fromiter(nx.get_node_attributes(G,'Mass').values(),dtype=float)
    MSizes =20 * m / np.max(m)
    
    MX = np.column_stack([V[0:2*N:2, mode], V[1:2*N:2, mode]])
    MV = np.linalg.norm(MX, axis=1)
    MV = MV/max(MV)
    
    pos = {i: X[i, :] for i in range(N)}
    node_color = [cmap[int(idx[i]-1), :] * MV[i] for i in range(N)]
    
    
    nx.draw_networkx_edges(G, pos, edge_color='k')
    nx.draw_networkx_nodes(G, pos, node_size=MSizes, node_color=node_color, alpha=1)
    plt.xlim(min(X.ravel()) - 2, max(X.ravel()) + 2)
    plt.ylim(min(X.ravel()) - 2, max(X.ravel()) + 2)
    plt.axis('equal')
    plt.axis('off')
    plt.show()

    
    
    
def createDelaunayGraph(NS, rseed, r_c, del_x):

    # This function creates a Delaunay graph of a set of points.

    # Parameters:
    #   NS: The number of points to generate.
    #   rseed: The random seed to use.
    #   r_c: The radius of the circumcircle of each edge in the graph.
    #   del_x: max noise magnitude from square lattice

    # Returns:
    #   N: The number of points in the graph.
    #   G: The graph object.
    #   X: The coordinates of the points.
    #   E: The edges of the graph.

    # Set the random seed.
    np.random.seed(rseed)

    # Generate the points.
    xm, ym = np.meshgrid(np.arange(1, NS + 1), np.arange(1, NS + 1))
    X = np.vstack((xm.flatten(), ym.flatten())).T
    N = X.shape[0]

    # Add some noise to the points.
    X = X + del_x * 2 * (0.5 - np.random.rand(N, 2))

    # Create the Delaunay triangulation.
    DT = Delaunay(X)

    # Get the edges of the triangulation.
    ET = np.empty((0, 2), dtype=int)
    for T in DT.simplices:
        ET = np.vstack((ET, [T[0], T[1]], [T[1], T[2]], [T[0], T[2]]))

    # Sort the edges.
    ET = np.sort(ET)

    # Get the radii of the circumcircles of the edges.
    R = np.linalg.norm(X[ET[:, 0], :] - X[ET[:, 1], :], axis=1)

    # Keep only the edges with radii less than r_c.
    EN = ET[R < r_c, :]

    # Create the adjacency matrix.
    A = np.zeros((N, N))
    A[EN[:, 0], EN[:, 1]] = 1

    # Create the graph object.
    G = nx.Graph(A)

    # Get the edges of the graph.
    E = np.array(G.edges)

    return N, G, X, E


def createSquareGraph(NS,rseed,r_c,del_x):
    np.random.seed(rseed)
    xm, ym = np.meshgrid(np.arange(1, NS+1), np.arange(1, NS+1))
    X = np.vstack((xm.flatten(), ym.flatten())).T
    N = X.shape[0]
    
    
    T=np.tile(X,(N,1,1))
    T2=np.moveaxis(T,0,1)
    TD=T-T2
    E1=np.array(np.where(np.linalg.norm(TD,axis=2)<r_c))
    E=np.transpose(E1[:,E1[0]>E1[1]])
    
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(E)
    E=np.array(G.edges)
    X = X + del_x * 2 * (0.5 - np.random.rand(N, 2))
    return N,G,X,E

def createCompatibility(N,X,E):
    # This function creates the compatibility matrix for a set of points.

    # Parameters:
    #   N: The number of points.
    #   X: The coordinates of the points.
    #   E: The edges of the graph.

    # Returns:
    #   C: The compatibility matrix.
    
    N_b = E.shape[0]
    mdict=dict(zip(range(N), m))
    nx.set_node_attributes(G,mdict,'Mass')
    C = np.zeros((2 * N, N_b))
    
    for i in range(N_b):
        b_vec = X[E[i, 0], :] - X[E[i, 1], :]
        b_vec = b_vec / np.linalg.norm(b_vec)
        for ki in range(2):
            j = E[i, ki]
            xind = 2 * j 
            C[xind:xind + 2, i] = ((-1) ** ki) * b_vec
    

    return C

def getDV(C, k, M):
    
    # This function computes the dynamic modes and their corresponding eigenvalues for a system of coupled oscillators.

    # Parameters:
    #   C: The compatibility matrix.
    #   k: The stiffness matrix.
    #   M: The mass matrix.

    # Returns:
    #   D: The eigenvalues.
    #   V: The eigenvectors.
    
    kd = np.diag(np.squeeze(k))
    K = C @ kd @ C.T
    DMAT = np.linalg.inv(M) @ K
    D, V = np.linalg.eig(DMAT)
    D = np.real(D)
    V = np.real(V)
    return D, V

StartTime=time.time()
N,G,X,E=createDelaunayGraph(20, 25, 2.0, 0.4)

pos = dict(zip(range(N), X))
fig = plt.figure(figsize=(16, 8))
fig.suptitle("Phonon Dispersion", fontsize=24)
ax=plt.subplot(1, 2, 1)
nx.draw(G, pos, node_size=0.1, node_color="blue")
ax.set_aspect('equal')
plt.title("Network Topology")

N_b = E.shape[0]
k = np.ones(N_b)
m = np.ones(N)
mdict=dict(zip(range(N), m))
nx.set_node_attributes(G,mdict,'Mass')
C = np.zeros((2 * N, N_b))

C=createCompatibility(N,X,E)


m2 = np.zeros(2 * N)
m2[0:2 * N:2] = m
m2[1:2 * N:2] = m
M = np.diag(m2)

D,V=getDV(C, k, M)

plt.subplot(1, 2, 2)
plt.hist(np.sqrt(np.abs(D)), bins=np.arange(-0.025, 3.025, 0.05))
plt.xlabel(r"$\omega$")
plt.ylabel(r"$C(\omega)$")
plt.title("Number of States")

print(time.time()-StartTime)