# ============================================================= 
# Some auxiliary functions to deal with colloidal ice systems
# God bless whoever reads this code
# Author: leogabac
# ============================================================= 

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
import vertices as vrt


ureg = ice.ureg
idx = pd.IndexSlice


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
    if position[0] < 0:
        position[0] += L

    # Apply BC to y
    position[1] = position[1] % L
    if position[1] < 0:
        position[1] += L

    return position


def classify_vertices(vrt):
    """
        Classifies the vertices in I, II, III, IV, V, VI types.
        Returns a DataFrame
        ----------
        Parameters:
        * vrt (pd Dataframe): Vertices df
    """

    vrt["type"] = np.NaN

    vrt.loc[vrt.eval("coordination==4 & charge == -4"),"type"] = 1
    vrt.loc[vrt.eval("coordination==4 & charge == -2"),"type"] = 2
    vrt.loc[vrt.eval("coordination==4 & charge == 0 & (dx**2+dy**2)==0"),"type"] = 3
    vrt.loc[vrt.eval("coordination==4 & charge == 0 & (dx**2+dy**2)>0"),"type"] = 4 # Dipolo
    vrt.loc[vrt.eval("coordination==4 & charge == 2"),"type"] = 5
    vrt.loc[vrt.eval("coordination==4 & charge == 4"),"type"] = 6
    return vrt


def vrt_dict(path):
    """
        Walks path and imports all DFs into a Dictionary, classifies the vertices and drops boundaries.
        Returns a dictionary with all the DataFrames.
        ----------
        Parameters:
        * path: Path where the vertices are located.
    """

    _, _, files = next(os.walk(path))
    verticesExp = {} # Initialize
    numberExperiments = len(files)
    for i in range(1,numberExperiments+1):
        filePath = os.path.join(path,f"vertices{i}.csv")

        # Check if the file exists, if not, skips.
        if not os.path.isfile(filePath):
            continue

        vrt = pd.read_csv(filePath, index_col=[0,1])
        vrt = classify_vertices(vrt)
        vrt = vrt.dropna()
        verticesExp[f"{i}"] = vrt
    return verticesExp


def vrt_counts(verticesDict):
    """
        Loops the verticesDict with all experiments and gets the counts for vertex type
        Returns a dictionary with the counts DF for all experiments
        ----------
        Parameters:
        * verticesDict: Dictionary from getVerticesDict()
    """

    countsDict = {}
    for key,experiment in verticesDict.items():
        currentCount = ice.count_vertices(experiment)
        countsDict[key] = currentCount
    return countsDict


def vrt_averages(counts,framerate):
    """
        Averages over all realizations.
        ----------
        Parameters:
        * counts (Dict): Counts dictionary with all experiments.
        * framerate: Framerate from the simulation
    """
    # Get a list containing all the different frames
    allFrames = counts["1"].index.get_level_values('frame').unique().to_list()
    time = np.array(allFrames)/framerate
    numberFrames = len(allFrames)
    numberRealizations = len(counts)

    fractions = pd.DataFrame(columns=["time","1.0","2.0","3.0","4.0","5.0","6.0"], data=np.zeros((numberFrames,7)))

    for key,experiment in counts.items():
        for vertexType,vrt in experiment.groupby("type"):
            vertexFraction = np.array(vrt.fraction)
            fractions[str(vertexType)] += vertexFraction

    fractions = fractions / numberRealizations
    fractions["time"] = time
    return fractions


def do_vertices(params,data_path):

    vrt_path = os.path.join(data_path,"vertices/")
    vertices = vrt_dict(vrt_path)
    counts = vrt_counts(vertices)
    vrt_ts = vrt_averages(counts,params["framespersec"].magnitude)
    types = vrt_ts.columns.to_list()[1:]
    t = vrt_ts["time"].to_numpy()
    vrt_cuentas = vrt_ts[types].to_numpy()
    return t, vrt_cuentas


def trj2col(params,ctrj):
    """
        Reconstruct the colloidal ice object from simulation parameters.
        Notice that this version uses a params dict.
        ----------
        Parameters:
        * ctrj (pd Dataframe): lammps ctrj without "t" and "type" columns
        * params: Dictionary with all simulation parameters
    """
    particle = params["particle"]
    trap = params["trap"]
    particle_radius = params["particle_radius"]
    a = params["lattice_constant"]
    N = params["size"]

    centers = [row[:3].to_list() * ureg.um for _,row in ctrj.iterrows()]
    directions = [row[3:6].to_list() * ureg.um for _,row in ctrj.iterrows()]
    arrangement = {
        "centers":centers,
        "directions":directions
    }

    col = ice.colloidal_ice(arrangement, particle, trap,
                            height_spread=0,
                            susceptibility_spread=0.1,
                            periodic=True)
    col.region = np.array([[0,0,-3*(particle_radius/a/N).magnitude],[1,1,3*(particle_radius/a/N).magnitude]])*N*a
    return col


def vrtcount_sframe(vrt, column="type"):
    """
        Counts the vertices of a single frame df.
        ----------
        Parameters:
        * vrt (pd Dataframe): Vertices dataframe.
        * column (optional)
    """
    vrt_count = vrt.groupby(column).count().iloc[:,0]
    types = vrt_count.index.get_level_values(column).unique()
    counts = pd.DataFrame({"counts": vrt_count.values}, index=types)
    counts["fraction"] = counts["counts"] / counts["counts"].sum()
    return counts


def vrt_lastframe(path,last_frame=2399):
    """
        Computes the vertices of only the last frame.
        ----------
        Parameters:
        * path: Filepath where the ctrj file is located
        * last_frame
    """

    ctrj = pd.read_csv(path,index_col=[0,1])

    if last_frame is None:
        last_frame = ctrj.index.get_level_values("frame").unique().max()
    
    ctrj = ctrj.loc[idx[last_frame,:]].drop(["t", "type"],axis=1)

    try:
        v = ice.vertices()
        v = v.trj_to_vertices(ctrj)
    except:
        vrt_lastframe(path,last_frame=last_frame-1)
  
    return v.vertices


def vrt_at_frame(ctrj,frame):
    """
        Computes the vertices of a specific frame.
        ----------
        Parameters:
        * path: Filepath where the ctrj file is located
        * last: last frame of the simulation
    """

    ctrj = ctrj.loc[idx[frame,:]].drop(["t", "type"],axis=1)

    v = ice.vertices()
    v = v.trj_to_vertices(ctrj)

    return v.vertices


def min_from_domain(f,domain):
    """
        Returns the value in domain that minimizes f.
        ----------
        Parameters:
        * f: function
        * domain: iterable
    """

    feval = [f(x) for x in domain]
    idx = np.argmin(feval)
    return domain[idx]


def positions_from_trj(ctrj):
    """
        Given a ctrj file. Retrieves the positions of the particles.
        This is used to compute energy.
        ----------
        Parameters:
        * ctrj
    """
    x = (ctrj['x'] + ctrj['cx']).to_numpy()
    y = (ctrj['y'] + ctrj['cy']).to_numpy()
    z = (ctrj['z'] + ctrj['cz']).to_numpy()

    stuff = pd.DataFrame(data=np.column_stack((x,y,z)), columns=['x','y','z'], index=list(range(1,len(ctrj)+1)))
    stuff.rename_axis('id', inplace=True)
    return stuff


def dropvis(ctrj):
    """
        Drop some columns of the ctrj files for drawing.
        ----------
        Parameters:
        * ctrj
    """
    return ctrj.drop(columns={'type','t'})


def load_ctrj_and_vertices(params,data_path,size,realization = 1):
    """
        Loads trj and vertices object.
        ----------
        Parameters:
        * params
        * data_path
        * size
        * realization
    """
    params['size'] = size
    ctrj = pd.read_csv(os.path.join(data_path,str(size),'trj','trj1.csv'),index_col=[0,1])
    vrt = pd.read_csv(os.path.join(data_path,str(size),'vertices','vertices1.csv'),index_col=[0,1])
    last_frame = vrt.index.get_level_values('frame').unique()[-1]
    
    v = ice.vertices()
    v.vertices = vrt
    
    return params,ctrj,v,last_frame

def get_rparalell(ctrj,particle,frame):
    
    """
        Computes the r_parallel component
       ----------
        Parameters:
        * ctrj
        * particle: particle id in dataframe
        * frame
    """   
        
    psel = ctrj.loc[idx[frame,particle]]
    
    r = psel[['cx','cy','cz']].to_numpy()
    
    # Note that here I used the absolute value of the direction
    # To only get whether it is vertical or horizontal component
    direction = psel[['dx','dy','dz']].to_numpy()
    direction = np.abs(direction)/np.linalg.norm(direction)
    #direction = direction/np.linalg.norm(direction)
    
    rp = np.dot(r,direction)
    
    
    return rp

def autocorrelation(ts):
    """
        Computes the autocorrelation function of a given timeseries. For all times.
        ts[0]ts[i] for all i
        ----------
        Parameters:
        * ts: timeseries
    """
   
    element = ts[0]
    return [tsi * element for tsi in ts]

def correlate_bframes(params,ts,sframes, stime= 0, etime = 60):
    """
        Computes the autocorrelations between some times (start,enD)
        Returns an array in which the rows are the different particles.
        ----------
        Parameters:
        * params
        * ts: rparallel timeseries where each row is a different particle
        * sframes: frames used in ts
        * start: start time (s)
        * end: end time (s)
    
    """
    
    startframe = params['framespersec'].magnitude * stime
    goalframe = params['framespersec'].magnitude * etime
    low = [ startframe <= sf for sf in sframes]
    high = [ sf <= goalframe for sf in sframes]
    whichframes = [h and l for (h,l) in zip(high,low)]
    subselframes = np.array(sframes)[whichframes]
    return subselframes, [ autocorrelation(np.array(ps)[whichframes]) for ps in ts]
    
    
def bint(x):
    return np.array([int(i) for i in x])


