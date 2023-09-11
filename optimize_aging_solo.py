"""
eigenvalue solver for phononics, directed ageing
"""

import numpy as np
from matplotlib  import pyplot as plt
from scipy.spatial import Delaunay
import networkx as nx
from matplotlib import rc
import scipy.linalg as scla
import time


rc('text', usetex=True)
rc('font', size=16)
rc('legend', fontsize=13)
rc('text.latex', preamble=r'\usepackage{cmbright}')


def createDelaunayGraph(NS,rseed,r_c,del_x):
    np.random.seed(rseed)
    xm, ym = np.meshgrid(np.arange(1, NS+1), np.arange(1, NS+1))
    X = np.vstack((xm.flatten(), ym.flatten())).T
    N = X.shape[0]
    X = X + del_x * 2 * (0.5 - np.random.rand(N, 2))
    
    DT = Delaunay(X)
    ET = np.empty((0, 2), dtype=int)
    for T in DT.simplices:
        ET = np.vstack((ET, [T[0], T[1]], [T[1], T[2]], [T[0], T[2]]))
    
    ET=np.sort(ET)
    R=np.linalg.norm(X[ET[:, 0], :] - X[ET[:, 1], :],axis=1)
    EN=ET[R<r_c,:]
    A = np.zeros((N, N))
    A[EN[:,0],EN[:,1]]=1
    
    
    G = nx.Graph(A)
    E=np.array(G.edges)
    
    return N,G,X,E
​
​
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
    X = X+ del_x * 2 * (0.5 - np.random.rand(N, 2))
    return N,G,X,E
​
​
​
​
def getBondImportance(X,C,V,D,D_range):
    modes = np.where((D > D_range[0]) & (D < D_range[1]))[0]
    delta_E=C.T@V
    EC=delta_E[:,modes]
    bondImportance=np.mean(np.abs(EC),axis=1)
    return bondImportance/np.max(bondImportance)
​
​
​
def createCompatibility(N,X,E):
    
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
    
​
    return C
​
def createSpringConstants(X,E,k_1):
    #k_1 is the spring constant for a spring of unit length
    displacements=X[E[:,0],:]-X[E[:,1],:]
    distance=np.linalg.norm(displacements,axis=1)
    return k_1/distance,distance
​
def getForbiddenModes(C, k, M, w_c, dw):
    kd = np.diag(np.squeeze(k))
    K = C @ kd @ C.T
    DMAT = np.linalg.inv(M) @ K
    D, V = scla.eig(DMAT,overwrite_a=True)
    D = np.real(D)
    w=np.sqrt(np.abs(D))
    forbidden_states=np.sum(np.logical_and(w>w_c-dw/2,w<w_c+dw/2))
    V=np.real(V)
    return D, V,forbidden_states
​
def ageSprings(k_old,X,C,V,D,D_range,ageing_rate):
    bond_importance=getBondImportance(X,C,V,D,D_range)
    bond_importance_centered=bond_importance-np.mean(bond_importance)
    bond_importance_normalized=bond_importance_centered/np.max(np.abs(bond_importance_centered))
​
    k_new=k_old*(1+2*ageing_rate*bond_importance_normalized)
    return k_new
def optimizeAgeing(C, k, M, w_c, dw, N_trials,ageing_rate,success_frac):
    w_range=[w_c-dw/2,w_c+dw/2]
    D_range = [x**2 for x in w_range]
    D, V, forbidden_states_initial = getForbiddenModes(C, k, M, w_c, dw)
    if forbidden_states_initial==0:
        return k,1,0
    for trial in range(1, N_trials+1):
        k=ageSprings(k,X,C,V,D,D_range,ageing_rate)
    
        D, V, forbidden_states = getForbiddenModes(C, k, M, w_c, dw)
#        print(trial,forbidden_states)
    
        if forbidden_states<=success_frac*forbidden_states_initial:
            
            return k, 1,trial
    
    return k,0,trial
    
StartTime=time.time()
​
​
runseed=10324
np.random.seed(runseed)
​
​
N_runs=1000
seeds=np.random.randint(0,103653,size=((N_runs)))
​
N_trials=1000
dw=0.05
w_c=0.5
​
ageing_rate=0.01
success_frac=0.05
freq_range=[w_c-dw/2,w_c+dw/2]
D_range = [x**2 for x in freq_range]
​
F_stack=[]
​
​
​
N,G,X,E=createDelaunayGraph(11, seeds[511], 2.0, 0.30)
​
​
k,L=createSpringConstants(X, E, 1)
m = np.ones(N)
mdict=dict(zip(range(N), m))
nx.set_node_attributes(G,mdict,'Mass')
​
C=createCompatibility(N,X,E)
​
m2 = np.zeros(2 * N)
m2[0:2 * N:2] = m
m2[1:2 * N:2] = m
M = np.diag(m2)
​
k2,result_flag,final_trial=optimizeAgeing(C, k, M, w_c, dw, N_trials,ageing_rate,success_frac)
​
​
fig = plt.figure(figsize=(18, 6))
​
plt.subplot(1, 3, 1)
pos = dict(zip(range(N), X))
edges=nx.draw_networkx_edges(G, pos, width=2*k2/k, alpha=0.6,edge_color='k')
plt.title("Graph Final")
​
plt.subplot(1,3, 2)
D, V, forbidden_states = getForbiddenModes(C, k2, M, w_c, dw)
plt.hist(np.sqrt(np.abs(D)), bins=np.arange(-0.025, 4.025, 0.05), density=True)
plt.xlabel(r'$\omega$')
plt.ylabel(r'$\rho(\omega)$')
print(str(forbidden_states)+' Forbidden States')
​
​
plt.subplot(1,3, 3)
​
plt.hist(k2/k, bins=np.arange(-0.025, np.max(k2/k)+0.25, 0.05), density=True)
plt.xlabel(r'$k$')
plt.ylabel(r'$\rho(k)$')
​
​
plt.show()
print(time.time()-StartTime)