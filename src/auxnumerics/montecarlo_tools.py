# ============================================================= 
# Some auxiliary functions to deal with simulated annealing
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
from math import isclose

ureg = ice.ureg
idx = pd.IndexSlice


# Some auxiliary functions

def flip_colloid_at_index(col, index):
    """
        Flips the direction of a given colloid at a certain index.
        ----------
        Parameters:
        * col: colloidal ice object
        * index
    """

    #col2 = col.copy(deep = True) 
    c = col[index]
    c.colloid = -c.colloid
    c.direction = -c.direction
    col[index] = c
    return col

def flip_colloids(col, amount = 1, indices = None):
    """
        Flips many colloids randomly.
        If indices is None, picks randomly.
        ----------
        Parameters:
        * col: colloidal ice object
        * amount
        * indices (list or None)
    """

    if indices is None:
        indices = np.random.randint(0,len(col)-1,amount)

    for index in indices:
        col = flip_colloid_at_index(col,index)
    return col

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
        
def get_index_from_position(col,pos, tol=0.1):
    """
        Gets the index of a colloid in a particular position.
        ----------
        Parameters:
        * col: colloidal ice object
        * pos: Position vector in 3D
        * tol: Tolerance, defaults to 0.1
    """

    for idx,c in enumerate(col):
        currentPos = c.center.magnitude.round()
        sepNorm = np.linalg.norm(currentPos - pos)

        if isclose(0,sepNorm,abs_tol=tol):
            return idx


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

def flip_loop(col, a=30, size=10):
    """
        Flips spins in a counter clockwise loop.
        ----------
        Parameters:
        * col: colloidal ice object
        * a: lattice constant
        * size
    """

    sel = np.random.randint(0,len(col))
    if is_horizontal(col[sel].direction):
            displacements = [
            np.array([0,0,0]),
            np.array([0,a,0]),
            np.array([a/2,a/2,0]),
            np.array([-a/2,a/2,0]) ]
    else:
            displacements = [
            np.array([0,0,0]),
            np.array([-a,0,0]),
            np.array([-a/2,a/2,0]),
            np.array([-a/2,-a/2,0]) ]

    positions = [ col[sel].center.magnitude + d for d in displacements]
    positions = [ fix_position(x,a,size).round() for x in positions]
    idxs = [get_index_from_position(col,x) for x in positions]

    col2 = flip_colloids(col,indices=idxs)
    return col2