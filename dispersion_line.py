# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 16:22:52 2023
dispersion curve
@author: Abhishek Sharma
"""

import numpy as np
from matplotlib  import pyplot as plt
from scipy.spatial import Delaunay
import networkx as nx
from matplotlib import rc
import scipy.linalg as scla
import time
import matplotlib.gridspec as gridspec

rc('text', usetex=True)
rc('font', size=16)
rc('legend', fontsize=13)
rc('text.latex', preamble=r'\usepackage{cmbright}')


def getForbiddenModes(C, k, M, w_c, dw):
    kd = np.diag(np.squeeze(k))
    K = C @ kd @ C.T
    DMAT = np.linalg.inv(M) @ K
    D, V = scla.eig(DMAT,overwrite_a=True)

    w=np.sqrt(np.abs(np.real(D)))
    forbidden_states=np.sum(np.logical_and(w>w_c-dw/2,w<w_c+dw/2))
    #V=np.real(V)
    return D, V,forbidden_states

def createCompatibility(N,X,E):
    
    N_b = E.shape[0]

    C = np.zeros((2 * N, N_b), dtype=np.float64)
    
    for i in range(N_b):
        b_vec = X[E[i, 0], :] - X[E[i, 1], :]
        b_vec = b_vec / np.linalg.norm(b_vec)
        for ki in range(2):
            j = E[i, ki]
            xind = 2 * j 
            C[xind:xind + 2, i] = ((-1) ** ki) * b_vec
    

    return C

def createSpringConstants(X,E,k_1):
    #k_1 is the spring constant for a spring of unit length
    displacements=X[E[:,0],:]-X[E[:,1],:]
    distance=np.linalg.norm(displacements,axis=1)
    return k_1/distance,distance

def createRectangularGraph(NX,NY,rseed,r_c,del_x):
    np.random.seed(rseed)
    xm, ym = np.meshgrid(np.arange(1, NX+1), np.arange(1, NY+1))
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
    X = X+ del_x * 2 * (0.5 - np.random.rand(N, 2))
    return N,G,X,E


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
    nx.draw(G, pos=X, node_size=2*m, width=LWidths, node_color='k', edge_color='k')
    for i in range(N):
        plt.arrow(X_[i], Y_[i], dx[i], dy[i], head_width=0.15, head_length=0.15, fc='r', ec='r')
    # plt.axis('equal')
    plotlimits = [min(X_.min(), Y_.min()) - 2, max(X_.max(), Y_.max()) + 2]
    plt.xlim(min(X.ravel()) - 2, max(X.ravel()) + 2)
    plt.ylim(min(X.ravel()) - 2, max(X.ravel()) + 2)
    plt.axis('equal')
    plt.axis('off')


def plotmodeD(G,X,V,mode):
    N = G.number_of_nodes()

    X2=X+np.stack((V[0:2*N:2,mode],V[1:2*N:2,mode])).T
    pos2 = dict(zip(range(G.number_of_nodes()), X2))
    nx.draw(G, pos2, node_size=10, node_color="blue")
    
    plt.xlim(min(X.ravel()) - 2, max(X.ravel()) + 2)
    plt.ylim(min(X.ravel()) - 2, max(X.ravel()) + 2)
    plt.axis('equal')
    plt.axis('off')
    
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
        plotmodeD(G, X,V, mode)
        ax.set_aspect('equal')

    fig2 = plt.figure(figsize=(10, 10))
    gs2 = gridspec.GridSpec(Ncols, Nrows, figure=fig2, wspace=0.2, hspace=0.2)
    for i, mode in enumerate(modes):
        ax = plt.subplot(gs2[i])
        plotmodeC(G, X, V, mode)
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


N,G,X,E=createRectangularGraph(15,1, 6565, 1.1, 0)
k,L=createSpringConstants(X, E, 1)
m = np.ones(N)
mdict=dict(zip(range(N), m))
nx.set_node_attributes(G,mdict,'Mass')

C=createCompatibility(N,X,E)

M = np.eye(2 * N-4)


C=np.delete(C,[0,1,-2,-1], axis=0)

kd = np.diag(np.squeeze(k))
K = C @ kd @ C.T
DMAT = np.linalg.inv(M) @ K

D,V = scla.eig(DMAT,overwrite_a=True)

fig1 = plt.figure(figsize=(10, 10))
plt.hist(np.sqrt(np.abs(D)), bins=np.arange(-0.025, 4.025, 0.05), density=False)
plt.xlabel(r'$\omega$')
plt.ylabel(r'$\rho(\omega)$')

V_plot = np.zeros((2*N,2*N-4))
V_plot[2:-2,:]=V

multiplots(G,X,2*V_plot,D,np.array([1.8,2.1])**2.0)
