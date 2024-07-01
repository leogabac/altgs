# ============================================================= 
# Modifications to montecarlo_tools
# The idea is to not rely on pandas, and make it faster
# God bless whoever reads this code
# Author: leogabac
# ============================================================= 

import os
import sys

sys.path.insert(0, '../icenumerics/')
import icenumerics as ice

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import itertools

from math import isclose

from auxiliary import *

from numba import jit,prange,float64,int64,complex128


ureg = ice.ureg
idx = pd.IndexSlice

def create_lattice(a,N,spos=(0,0)):
    """
        Creates a (N,N,3) matrix where the element A[i,j,:] = [x,y,z] are the coordinates of a vertex.
        Note that x varies with the columns so A[i,j+1,:] = [x+a,y,z]
        ----------
        Parameters:
        * a: lattice constant
        * N: vertices per side
    """
    xs,ys = spos
    x = np.arange(xs,N*a,a)
    y = np.arange(ys,N*a,a)
    X,Y = np.meshgrid(x,y)
    lattice = np.zeros((N,N,3))
    lattice[:,:,0]=X
    lattice[:,:,1]=Y
    return lattice


def trj2numpy(trj):
    """
        Takes a trj df and converts it to numpy arrays separated by centers, directions, and relative coordinate.
        ----------
        Parameters
        * trj: trajectory DataFrame
    """
    centers = trj[['x','y','z']].to_numpy()
    dirs = trj[['dx','dy','dz']].to_numpy()
    rels = trj[['cx','cy','cz']].to_numpy()
    return centers,dirs,rels


def numpy2trj(centers,dirs,rels):
    """
        Takes numpy arrays and converts them into a trj dataframe
        ----------
        Parameters
        * centers: center of the trap
        * dirs: normalized direction
        * rels: relative position to the center of the trap
    """

    trj = np.concatenate([centers,dirs,rels],axis=1)
    trj = pd.DataFrame(trj,columns=['x','y','z','dx','dy','dz','cx','cy','cz'])
    trj['id'] = list(range(len(trj)))
    trj['frame'] = [0]*len(trj)
    trj = trj.set_index(['frame', 'id'])
    return trj


def flip_spin(dirs,rels,idx):
    """
        Flips the spin at idx.
        ----------
        Parameters
        * centers: center of the trap
        * dirs: normalized direction
        * rels: relative position to the center of the trap
    """

    # sanity check
    dirs1 = dirs.copy()
    rels1 = rels.copy()
    
    dirs1[idx] = -dirs1[idx]
    rels1[idx] = -rels1[idx]

    return dirs1, rels1


@jit(nopython=True)
def get_idx_from_position(centers,pos,tol=0.1):
    """
        Get the index in the centers array from a position vector.
        ----------
        * centers: centers of the traps
        * pos: np array with a 3D coordinate
    """
    
    for i,center in enumerate(centers):
        distance = np.linalg.norm(center - pos)
        if np.isclose(0,distance,atol=tol):
            return i


def is_horizontal(direction):
    """
        Checks if a given direction is horizontal.
        ----------
        Parameters:
        * direction
    """
    x = np.array([1,0,0])
    dotP = np.dot(direction,x)

    if isclose(abs(dotP),1,rel_tol=1e-3):
        return True
    else:
        return False


@jit(nopython=True)
def fix_position(position,a,size):
    """
        Fixes the position to fit in the box
        0 < x < size*a, and 
        0 < y < size*a
        ----------
        Parameters:
        * position: Position vector in 3D
        * a: lattice constant
        * size: size of the system
    """
    L = size*a

    # Apply BC to x
    position[0] = position[0] % L
    if position[0]<0:
        position[0] += L

    # Apply BC to y
    position[1] = position[1] % L
    if position[1]<0:
        position[1] += L

    return position


def flip_loop(a,size,centers,dirs,rels, idx = None):
    """
        Flips all colloids in a randomly selected loop
        ----------
        Parameters:
        * a: lattice constant
        * size: vertices per side
        * centers: centers of the traps
        * dirs: directions of the colloids
        * rels: relative posisition of particles wrt the centers
    """
    # choose one random colloid
    if idx is None:
        sel = np.random.randint(len(centers))
    else:
        sel = idx

    # decide if horizontal or vertical & get displacements in loop
    if is_horizontal(dirs[sel]):
        # down, up, right, left
        shift = [
        np.array([0,0,0]),
        np.array([0,a,0]),
        np.array([a/2,a/2,0]),
        np.array([-a/2,a/2,0]) ]
    else:
        # right, left, up, down
        shift = [
        np.array([0,0,0]),
        np.array([-a,0,0]),
        np.array([-a/2,a/2,0]),
        np.array([-a/2,-a/2,0]) ]

    # get those positions & fix pbc
    pos_pbc = np.zeros((4,3))
    for j in range(len(shift)):
        cpos = fix_position(centers[sel] + shift[j],a,size)
        pos_pbc[j,:] = cpos

    # get the indices
    indices = [get_idx_from_position(centers,x) for x in pos_pbc]

    # flip them (kinda hard coded oops)
    dirs, rels = flip_spin(dirs,rels,indices[0])
    dirs, rels = flip_spin(dirs,rels,indices[1])
    dirs, rels = flip_spin(dirs,rels,indices[2])
    dirs, rels = flip_spin(dirs,rels,indices[3])

    return dirs, rels, pos_pbc


def indices_lattice(vrt_lattice,centers,a,N):
    """
        Make a matrix of size (L,L,4) where the (i,j,:) element
        points to the 4 colloids associated to vertex (i,j)
        ----------
        Parameters:
        * vrt_lattice: (L,L,3) array with the vertex positions in real space
        * centers: centers of the traps
    """

    rows, cols = vrt_lattice.shape[:2]
    indices_matrix = np.zeros((rows,cols,4)) # intialize

    for i in range(rows):
        for j in range(cols):
            # current position at (i,j)           
            cur_vrt = vrt_lattice[i,j,:] 

            # get the positions with pbc
            up = fix_position(cur_vrt + np.array([0,a/2,0]),a,N)
            down = fix_position( cur_vrt + np.array([0,-a/2,0]), a, N)
            left = fix_position( cur_vrt + np.array([-a/2,0,0]), a, N)
            right = fix_position(cur_vrt + np.array([a/2,0,0]), a, N)

            # get the indices
            up_idx = get_idx_from_position(centers,up,tol=0.05)
            down_idx = get_idx_from_position(centers,down,tol=0.05)
            left_idx = get_idx_from_position(centers,left,tol=0.05)
            right_idx = get_idx_from_position(centers,right,tol=0.05)

            indices_matrix[i,j,:] = np.array([up_idx,down_idx,left_idx,right_idx])
    
    return indices_matrix


@jit(nopython=True)
def get_topological_charge_at_vertex(indices,dirs):
    """
        Computes the topological charge at a given vertex.
        ----------
        Parameters:
        * indices: the 4 indices that point to the 4 colloids in the vertes
        * dirs: directions of all vertices
    """
    
    towards = np.array([
        [0,-1,0], #up
        [0,1,0], #down
        [1,0,0], #left
        [-1,0,0] #right
    ])

    charge = 0
    for i in range(len(indices)):
        idx = int(indices[i])

        # this is a dot product between direction \cdot towards
        # this will give 1 if the spin points towards the vertex
        # this will give -1 if the spin points away
        charge += dirs[idx][0]*towards[i][0] + dirs[idx][1]*towards[i][1] + dirs[idx][2]*towards[i][2] 
        

    return charge


@jit(nopython=True)
def get_charge_lattice(indices_lattice,dirs):
    """
        Computes the topological charge in the current lattice
        ----------
        Parameters:
        * indices_lattice
        * dirs
    """

    rows, cols = indices_lattice.shape[:2]
    charges = np.zeros((rows,cols))

    for i in range(rows):
        for j in range(cols):
            charges[i,j] = get_topological_charge_at_vertex(indices_lattice[i,j,:],dirs)
    
    return charges


@jit(nopython=True)
def is_accepted(dE,T, kB =1):
    """
        Acceptance function for simulated annealing.
        ----------
        Parameters:
        * dE: Energy difference
        * T: Temperature
        * kB (obtional): Bolzman constant, defaults to 1.
    """
    division = (dE/kB/T)
    if dE < 0:
        return True
    else:
        r = np.random.rand()
        if r < np.exp(-division):
            return True
        else:
            return False


@jit(nopython=True)
def charge_op(charged_vertices):
    """
        Computes the kappa order parameter
        ----------
        Parameters:
        * charged_vertices: Array (N,N) where (i,j) has the charge of vertex (i,j)
    """

    kappa = 0
    rows,cols = charged_vertices.shape 
    for i in range(rows):
        for j in range(cols):
            kappa += charged_vertices[i,j]*(-1)**(i+j)

    return kappa


def get_objective_function(indices_matrix,dirs,N):
    """
        Computes the objective function
        ----------
        Parameters
        * indices_matrix: association matrix
        * dirs: traps directions
        * N: vertices per side
    """

    q = get_charge_lattice(indices_matrix,dirs)
    kappa = charge_op(q)
    return np.abs(2*N**2 - np.abs(kappa))


def obj_icerule(indices_matrix,dirs,N):
    """
        Computes the objective function
        ----------
        Parameters
        * indices_matrix: association matrix
        * dirs: traps directions
        * N: vertices per side
    """

    q = get_charge_lattice(indices_matrix,dirs)
    kappa = charge_op(q)
    return np.abs(kappa)

def display_vertices(trj,N,a,ax):
    """
        Plots the topological charges of a given trj.
        ----------
        Parameters:
        * trj: trajectory dataframe
        * N: vertices per side
        * a: lattice constant
        * ax: matplotlib.Axes object
    """
    
    # generate the topology
    centers, dirs, rels = trj2numpy(trj)
    vrt_lattice = create_lattice(a.magnitude,N,spos=(0,0))
    indices_matrix = indices_lattice(vrt_lattice,centers, a.magnitude, N)

    # lattice with the topological charges
    q = get_charge_lattice(indices_matrix,dirs)

    rows, cols = q.shape

    for i in range(rows):
        for j in range(cols):

            if q[i,j] < 0:
                c = 'blue'
            elif q[i,j] >0:
                c = 'red'
            else:
                c='k'

            ax.add_artist( plt.Circle(
                vrt_lattice[i,j,:2], # position
                0.9*np.abs(q[i,j]), # radius
                color=c
                ))


def normalize(x):
    """
        What the name says bro.
        ----------
        Parameters:
        * x: some np.array
    """

    norma = np.linalg.norm(x)
    if norma == 0:
        return x
    else:
        return x/np.linalg.norm(x)


def display_arrows(trj,N,a,ax):
    """ 
        Displays the arrows at vertces
        ----------
        Parameters
        * trj: trajectory DataFrame
        * N: vertices per side
        * a: lattice constant
        * ax: plt.Axes object
    """

    offset = 2.5
    # generate the topology
    centers, dirs, rels = trj2numpy(trj)
    vrt_lattice = create_lattice(a.magnitude,N,spos=(0,0))
    indices_matrix = indices_lattice(vrt_lattice,centers, a.magnitude, N)

    rows, cols = indices_matrix.shape[:2]

    for i in range(rows):
        for j in range(cols):

            # get the position
            x,y,z = tuple(vrt_lattice[i,j,:])
            # get the directions of the colloids related to the vertices
            cidxs = [int(k) for k in  indices_matrix[i,j,:]]
            # get the total direction of the arrow at vertex
            # this is only the vector sum of all of the directions at the vetex
            arrow_direction = normalize( np.sum(dirs[cidxs], axis=0) )
            dx,dy,dz= tuple(arrow_direction)

            ax.add_artist( plt.Arrow(x-offset*dx,y-offset*dy,2*offset*dx,2*offset*dy, width=5, color='black'))


def dipole_lattice(centers,dirs,rels,vrt_lattice,indices_matrix):
    """
        Computes the lattice of the dipoles at each vertes
        ----------
        Parameters:
        * centers: centers of the traps
        * dirs: directions of the colloids
        * rels: relative position wrt the center of the trap
        * vrt_lattice: lattice with the position of the vertices
        * indices_matrix: assocition matrix
    """

    rows, cols = indices_matrix.shape[:2]
    arrow_lattice = np.zeros((rows,cols,3))
    
    for i in range(rows):
        for j in range(cols):

            # get the directions of the colloids related to the vertices
            cidxs = [int(k) for k in  indices_matrix[i,j,:]]
            # get the total direction of the arrow at vertex
            # this is only the vector sum of all of the directions at the vetex
            arrow_direction = normalize( np.sum(dirs[cidxs], axis=0) )
            arrow_lattice[i,j,:] = arrow_direction


    return arrow_lattice


def display_lines(trj,N,a,ax):
    """
        Displays the lines connecting the dipole arros at each vertex.
        ----------
        Parameters:
        * trj: trajectory DataFrame
        * N: vertices per side
        * a: lattice constant 
        * ax: plt.Axes object
    """

    from matplotlib.lines import Line2D

    # some plotting parameters
    offset = 5

    # generate the topology
    centers, dirs, rels = trj2numpy(trj)
    vrt_lattice = create_lattice(a.magnitude,N,spos=(0,0))
    indices_matrix = indices_lattice(vrt_lattice,centers, a.magnitude, N)

    rows, cols = indices_matrix.shape[:2]

    for i in range(rows):
        for j in range(cols):

            # get the position
            x,y,z = tuple(vrt_lattice[i,j,:])
            # get the directions of the colloids related to the vertices
            cidxs = [int(k) for k in  indices_matrix[i,j,:]]
            # get the total direction of the arrow at vertex
            # this is only the vector sum of all of the directions at the vetex
            arrow_direction = normalize( np.sum(dirs[cidxs], axis=0) )
            dx,dy,dz= tuple(arrow_direction)


            ax.add_line( Line2D([x-offset*dx, x+offset*dx ],[y-offset*dy, y+offset*dy ], color='#d10014', linewidth=3)    )


@jit(nopython=True)
def numba_pbc_displacement(xi,xj,L):
    """
        Determines the displacement vector with PBC.
        ----------
        Parameters:
        * xi: position 
        * xj: position
        * L: N*a
    """
    xij = xi - xj
    ox = np.array([xij[0], xij[0]+L, xij[0]-L])
    oy = np.array([xij[1], xij[1]+L, xij[1]-L])
    oz = np.array([xij[2], xij[2]+L, xij[2]-L])
    
    ix = np.argmin(np.abs(ox))
    iy = np.argmin(np.abs(oy))
    iz = np.argmin(np.abs(oz))
    
    xij_pbc = np.array([ox[ix], oy[iy], oz[iz]])
    
    return xij_pbc


@jit(float64(float64[:],float64[:],float64),nopython=True)
def numba_pbc_distance(xi,xj,L):
    """
        Computes the distance between xi an xj with PBC 
        ----------
        Parameters:
        * xi: position 
        * xj: position 
        * L: N*a 
    """ 
       
    xij_pbc = numba_pbc_displacement(xi,xj,L)
    
    return np.sqrt((xij_pbc**2).sum())


@jit(nopython=True)
def dotp(x,y):
    """
        Hardcoded dot product
        ----------
        Parameters:
        x: array 
        y: array
    """
    return x[0]*y[0] + x[1]*y[1]



@jit(
    complex128(float64[:, :], float64[:, :], float64[:, :], int64, float64, float64[:]),
    nopython=True,
    fastmath=True,
)
def single_msf_colloids(centers,dirs,rels,N,a,q):

    cutoff =  10*a
    suma = 0 # intialize

    # here i want to loop through all pairs
    for i in range(len(centers)):
        for j in range(i,len(centers)):
            
            riajb = numba_pbc_displacement( centers[i], centers[j], N*a )

            if np.sqrt((riajb**2).sum()) < cutoff:

                qhat = q/np.linalg.norm(q)
                Sia_perp = dirs[i][:2] - qhat*dotp(qhat,dirs[i][:2])
                Sjb_perp = dirs[j][:2] - qhat*dotp(qhat,dirs[j][:2])
                
                suma += ( dotp(Sia_perp,Sjb_perp) * np.exp(1j * dotp(q,riajb[:2]) ) )
            else:
                continue

            
    # idk if i should adjust the denominator for the fact that i added a cutoff
    return suma

@jit(nopython=True,parallel=True,fastmath=True)
def magnetic_structure_factor(centers,dirs,rels,N,a,reciprocal_lattice,rc_pairs,progress_proxy):

    rows, cols = reciprocal_lattice.shape[:2]
    msf = np.zeros((rows,cols),dtype=np.complex128)

    for j in prange(len(rc_pairs)):

        idx = rc_pairs[j]
        q = reciprocal_lattice[idx[0],idx[1],:]
        msf[idx[0],idx[1]] = single_msf_colloids(centers,dirs,rels,N,a,q)

        progress_proxy.update(1)

    # i did not divide by the number of spins...
    return msf/2/(N**2)

@jit(nopython=True)
def charge_correlations(fcharge, fvertices, pairs, N, a):

    L = N*a
    # initialize correlation array
    # corr, dx/a, dy/a

    corr = np.zeros( (len(pairs),3) )

    # loop through pairs
    for k in range(len(pairs)):

        # compute the product of paris qi* qj
        corr[k,0] = fcharge[ pairs[k,0] ] * fcharge[ pairs[k,1] ] 

        # take displacement with pbc
        rij_pbc = numba_pbc_displacement(fvertices[ pairs[k,0]], fvertices[pairs[k,1]], L) / a
        corr[k,1] = rij_pbc[0]
        corr[k,2] = rij_pbc[1]
        

    
    return corr

## NADA DE ESTO SIRVE PA PURA PINCHE VERGA

@jit(nopython=True)
def pick_elements(ox, ix):
    n_cols = ox.shape[1]
    result = np.empty(n_cols, dtype=ox.dtype)
    
    for col in range(n_cols):
        result[col] = ox[ix[col], col]
    
    return result

@jit(nopython=True)
def vectorized_pbc_displacements(centers,pairs,L):
    dx = centers[pairs[:,0],0] - centers[pairs[:,1],0]
    dy = centers[pairs[:,0],1] - centers[pairs[:,1],1]
    
    ox = np.zeros((3,len(dx)))
    ox[0,:] = dx
    ox[1,:] = dx + L
    ox[2,:] = dx - L
    
    oy = np.zeros((3,len(dy)))
    oy[0,:] = dy
    oy[1,:] = dy + L
    oy[2,:] = dy - L
    
    ix = np.argmin(np.abs(ox),axis=0)
    iy = np.argmin(np.abs(oy),axis=0)
    return pick_elements(ox,ix), pick_elements(oy,iy)

@jit(nopython=True)
def vectorized_single_msf(pairs,centers,dirs,N,a,q):
    dx,dy = vectorized_pbc_displacements(centers,pairs,N*a)
    distances = np.sqrt(dx**2 + dy**2)
    
    b = distances <= 10*a

    qhat = q/np.linalg.norm(q)
    # here the idea is the following
    # compute Sia - qhat * dot(Sia,qhat)
    # here each row has a different Sia
    # the operation dirs[pairs[:,0],:2] only selects spins according to pairs
    # and produces a matrix where each row is a different spin
    # then the dot product with qhat is simply the matrix/vector product S*hat,
    # where now each row is the result of all different dot products
    Sia = dirs[pairs[b,0],:2] - (qhat[:,np.newaxis] * np.dot(dirs[pairs[b,0],:2],qhat)).T
    Sjb = dirs[pairs[b,1],:2] - (qhat[:,np.newaxis] * np.dot(dirs[pairs[b,1],:2],qhat)).T
    return np.sum(np.sum(Sia*Sjb, axis=1)*np.exp(1j*(q[0]*dx[b]+q[1]*dy[b])))


@jit(nopython=True,fastmath=True)
def vector_msf(pairs,centers,dirs,N,a,reciprocal_lattice,rc_pairs,progress_proxy):

    rows, cols = reciprocal_lattice.shape[:2]
    msf = np.zeros((rows,cols),dtype=np.complex128)

    for j in range(len(rc_pairs)):

        idx = rc_pairs[j]
        q = reciprocal_lattice[idx[0],idx[1],:]
        msf[idx[0],idx[1]] = vectorized_single_msf(pairs,centers,dirs,N,a,q)

        
        progress_proxy.update(1)


    return msf/2/(N**2)
