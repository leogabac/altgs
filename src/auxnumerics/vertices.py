# ==============================
# These are some functions to deal with vertices computations
# ==============================

import os
import sys
import numpy as np
import pandas as pd
import math
import itertools

from tqdm import tqdm
from math import isclose
from numba import jit,prange,float64,int64,complex128

# own
sys.path.insert(0, '../icenumerics/')
import icenumerics as ice
import auxiliary as aux

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

def trj2trj(trj):
    return numpy2trj(*trj2numpy(trj))

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
            up = aux.fix_position(cur_vrt + np.array([0,a/2,0]),a,N)
            down = aux.fix_position( cur_vrt + np.array([0,-a/2,0]), a, N)
            left = aux.fix_position( cur_vrt + np.array([-a/2,0,0]), a, N)
            right = aux.fix_position(cur_vrt + np.array([a/2,0,0]), a, N)

            # get the indices
            up_idx = aux.get_idx_from_position(centers,up,tol=0.05)
            down_idx = aux.get_idx_from_position(centers,down,tol=0.05)
            left_idx = aux.get_idx_from_position(centers,left,tol=0.05)
            right_idx = aux.get_idx_from_position(centers,right,tol=0.05)

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
            #arrow_direction = normalize( np.sum(dirs[cidxs], axis=0) )
            arrow_direction =  np.where( np.sum(dirs[cidxs], axis=0) ==0,0,1 )
            arrow_lattice[i,j,:] = arrow_direction


    return arrow_lattice

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
