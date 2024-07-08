# =============================================================
# Some auxiliary functions to deal with order parameters 
# God bless whoever reads this code 
# Author: leogabac 
# =============================================================
import os
import sys
sys.path.insert(0, '../icenumerics/')
import icenumerics as ice 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools

from math import isclose

from auxiliary import *

ureg = ice.ureg
idx = pd.IndexSlice



def create_chiral_space_lattice(a=30,L=10,spos=(15,15)):
    """
        Create a matrix of shape (L,L,3) that stores the coordinates of the centers of the lattice.
        ----------
        Parameters:
        * a: lattice constant
        * L: size of the system
        * spos: Offcenter tuple.
    """
    xstart,ystart = spos
    xcoords = np.linspace(xstart,L*a-xstart,L)
    ycoords = np.linspace(ystart,L*a-ystart,L)
    zcoords = [0]

    global_pos = [ np.array(element) for element in itertools.product(xcoords,ycoords,zcoords) ]

    return np.reshape(global_pos,(L,L,3))

def create_chiral_lattice(col,global_pos,a=30,L=10):
    """
        Create a lattice where each site (i,j) has four colloids associated to it.

        Returns a matrix of shape (L,L,4). Where A[i,j,:] are the four indices of
        the colloidal ice object related to the site (i,j).
        ----------
        Parameters:
        * col: colloidal ice object
        * global_pos: chiral space lattice of size (L,L,3)
        * a: lattice constant
        * L: size of the system
    """

    global_idx = np.zeros((L,L,4))
    for i in range(L):
        for j in range(L):
            curpos = global_pos[i,j,:]
            right =  fix_position(curpos + np.array([a/2,0,0]),a,L).round()
            top = fix_position(curpos + np.array([0,a/2,0]),a,L).round()
            left =  fix_position(curpos + np.array([-a/2,0,0]),a,L).round()
            bottom = fix_position(curpos + np.array([0,-a/2,0]),a,L).round()
             
            right_id = get_index_from_position(col,right)
            top_id = get_index_from_position(col,top)
            left_id = get_index_from_position(col,left)
            bottom_id = get_index_from_position(col,bottom)
             
            global_idx[i,j,:] = np.array([right_id,top_id,left_id,bottom_id])
    
    return global_idx

def normalize_spin(col,idx):
    """
        Normalize the direction of the spin. Sometimes they are not.
        ----------
        * col: colloidal ice object
        * idx: index of the colloid of interest
    """
    vector = col[int(idx)].direction
    return vector / np.linalg.norm(vector)


def calculate_single_chirality(col,idxs):
    """
        Computes the chirality of a single cell.
        ----------
        * col: colloidal ice object
        * idxs: indices of the four colloids in a single cell
    """


    up = np.array([0,1,0])
    down = -up
    right = np.array([1,0,0])
    left = -right

    positive = [up,left,down,right]
    negative = [left,up,right,down]

    # try positive chirality
    sum_spins = sum(np.dot(normalize_spin(col,idx),spin) for idx,spin in zip(idxs,positive))

    if isclose(sum_spins,4,rel_tol=1e-2):
        return 1
    elif isclose(sum_spins,-4,rel_tol=1e-2):
        return -1
    else:
        return 0
    
def calculate_chirality(col,chiral_lattice,a,L):
    """
        Calculate the chirality of a state.
        ----------
        * col: colloidal ice object
        * chiral_lattice: matrix of size (L,L,4)
        * a: lattice constant
        * L: size of the system
    """

    s = 0 # initialize

    for i in range(L):
        for j in range(L):
            s += calculate_single_chirality(col,chiral_lattice[i,j,:]) * (-1)**(i+j)

    return s

def get_chirality_on_realization(params,folder,angle,realization,last_frame=2399):
    """
        CAREFUL WITH THE PATH!

        Given a specific realization angle,i and the last frame of the simulation.
        Compute the chirality.
        ----------
        * params: dictionary with the parameters of the simulation
        * angle: particular angle (colatitude)
        * realization: which realization it is being selected
        * last_frame: last frame of the simulation
    """
    
    # Wasting bc i am lazy af
    particle = params["particle"]
    trap = params["trap"]
    particle_radius = params["particle_radius"]
    L = params["lattice_constant"].magnitude
    N = params["size"]

    angle_path = f"../data/{folder}/angles/{angle}/ctrj/ctrj{realization}.csv"
    current_ctrj = pd.read_csv(angle_path,index_col=[0,1])

    if last_frame is None:
        last_frame = current_ctrj.index.get_level_values("frame").unique().max()
    
    state_ctrj = current_ctrj.loc[idx[last_frame,:]].drop(["t", "type"],axis=1)
    current_col = get_colloids_from_ctrj(state_ctrj,particle,trap,particle_radius,L,N)

    pos_lattice = create_chiral_space_lattice(a=L,L=N,spos=(L/2,L/2))
    idx_lattice = create_chiral_lattice(current_col,pos_lattice,L,N)
    cur_chirality = calculate_chirality(current_col,idx_lattice,L,N)

    return cur_chirality

def get_chirality_on_realization2(params,data_path,realization,sel_frame=2399):
    """
        I made this one to not break old codes...
        This thing does the same as the one above

        Given a specific realization angle,i and the last frame of the simulation.
        Compute the chirality.
        ----------
        * params: dictionary with the parameters of the simulation
        * angle: particular angle (colatitude)
        * realization: which realization it is being selected
        * sel_frame: last frame of the simulation
    """
    
    # Wasting bc i am lazy af
    particle = params["particle"]
    trap = params["trap"]
    particle_radius = params["particle_radius"]
    L = params["lattice_constant"].magnitude
    N = params["size"]

    angle_path = f"{data_path}/ctrj/ctrj{realization}.csv"
    current_ctrj = pd.read_csv(angle_path,index_col=[0,1])

    if sel_frame is None:
        sel_frame = current_ctrj.index.get_level_values("frame").unique().max()
    
    state_ctrj = current_ctrj.loc[idx[sel_frame,:]].drop(["t", "type"],axis=1)
    current_col = get_colloids_from_ctrj(state_ctrj,particle,trap,particle_radius,L,N)

    pos_lattice = create_chiral_space_lattice(a=L,L=N,spos=(L/2,L/2))
    idx_lattice = create_chiral_lattice(current_col,pos_lattice,L,N)
    cur_chirality = calculate_chirality(current_col,idx_lattice,L,N)

    return cur_chirality

def shift_vertices(vertices,x_shift,y_shift):
    """
        Shifts the vertices positions
        ----------
        Parameters:
        * vertices (pd Dataframe): Vertices df
        * x_shift
        * y_shift
    """

    vertices_shifted = vertices.copy()
    vertices_shifted["x"] = vertices_shifted["x"] + x_shift
    vertices_shifted["y"] = vertices_shifted["y"] + y_shift

    return vertices_shifted

def apply_pbd_to_vertices(vertices,lattice_constant,size):
    """
        Apply Periodic Boundary Conditions to shifted vertices.
        ----------
        Parameters:
        * vertices: Dataframe of shifted vertices.
        * lattice_constant
        * size: size of the system
    """

    L = lattice_constant*size

    x = vertices["x"].to_numpy()
    y = vertices["y"].to_numpy()

    # Apply BC to x
    x = x % L
    sel = x<0
    x[sel] = x[sel] + L

    # Apply BC to y
    y = y % L
    sel = x<0
    y[sel] = y[sel] + L

    vertices["x"] = x
    vertices["y"] = y

    return vertices

def where_in_space_lattice(pos,space_lattice,N,tol=0.1):
    """
        Returns indices i,j in A[i,j,:] where pos is stored
        ----------
        Parameters:
        * pos: Position of a vertex
        * space_lattice: Tensor of coordinates
        * N: size of the system
    """

    for j in range(N):
        for k in range(N):

            current_space_pos = space_lattice[j,k,:2]
            distance = np.linalg.norm(pos - current_space_pos)
                
            if isclose(distance,0,abs_tol=tol):
                return j,k



def create_charge_order_lattice(params, path, space_lattice,tol=0.1):
    """
        Creates a matrix with topological charges
        ----------
        Parameters:
        * params: simulation parameters
        * path: where the vertices file is located
        * space_lattice: correspondent lattice with the centers of the cells.
    """

    
    a = params["lattice_constant"].magnitude
    N = params["size"]

    charges = np.zeros((N,N))
    vertices = pd.read_csv(path, index_col=0)
    vertices_corrected = apply_pbd_to_vertices(shift_vertices(vertices,a/2,-a/2),a,N)

    for i,pos in enumerate(vertices_corrected[["x","y"]].to_numpy()):

        j,k = where_in_space_lattice(pos, space_lattice, N,tol=tol)

        charges[j,k] = vertices_corrected["charge"][i]

    return charges

def get_charge_order(charge_lattice):
    """
        Complementary order parameter.
        ----------
        Parameters
        * charge lattice
    """

    s = 0 # initialize

    N = charge_lattice.shape[0]

    for i in range(N):
        for j in range(N):

            s+= (-1)**(i+j) * charge_lattice[i,j]
    
    return s

def get_charge_order_on_realization(params, folder, angle, realization, tol=0.1):
    """
        Gets the complementary order parameter on a spcific realization.
        Supposes that the vertices file is single frame.
        * params: Simulation parameters
        * folder: Specific folder test
        * angle
        * realization
    """

    a = params["lattice_constant"].magnitude
    N = params["size"]

    path = f"../data/{folder}/vertices/{angle}/vertices{realization}.csv"
    space_lattice = create_chiral_space_lattice(a=a,L=N,spos=(a/2,a/2))
    charge_lattice = create_charge_order_lattice(params, path,space_lattice,tol=tol)
    return get_charge_order(charge_lattice)

def create_charge_order_lattice_from_multiindex(params, path, space_lattice,frame,tol=0.1):
    """
        Creates a matrix with topological charges from a multiindex file at a given frame
        ----------
        Parameters:
        * params: simulation parameters
        * path: where the vertices file is located
        * space_lattice: correspondent lattice with the centers of the cells.
    """

    
    a = params["lattice_constant"].magnitude
    N = params["size"]

    charges = np.zeros((N,N))
    full_vertices = pd.read_csv(path, index_col=[0,1])
    vertices = full_vertices.loc[frame,:]
    
    # here goes some shitty code that extracts the single frame that i need
    
    vertices_corrected = apply_pbd_to_vertices(shift_vertices(vertices,a/2,-a/2),a,N)

    for i,pos in enumerate(vertices_corrected[["x","y"]].to_numpy()):

        j,k = where_in_space_lattice(pos, space_lattice, N,tol=tol)

        charges[j,k] = vertices_corrected["charge"][i]

    return charges

def get_charge_order_on_frame_on_realization(params,path,frame,realization,tol=0.1):
    """
        Gets the complementary order parameter on a spcific realization at desired frame.
        Assumes that the vertices file is a multiindex.
        * params: Simulation parameters
        * folder: Specific folder test
        * angle
        * realization
    """
    
    a = params["lattice_constant"].magnitude
    N = params["size"]
    vrt_path = os.path.join(path,"vertices",f"vertices{realization}.csv")
    space_lattice = create_chiral_space_lattice(a=a,L=N,spos=(a/2,a/2))
    charge_lattice = create_charge_order_lattice_from_multiindex(params, vrt_path,space_lattice,frame,tol=tol)
    return get_charge_order(charge_lattice)
    
    


