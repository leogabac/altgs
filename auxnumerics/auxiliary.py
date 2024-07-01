# ============================================================= 
# Some auxiliary functions to deal with colloidal ice systems
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

ureg = ice.ureg
idx = pd.IndexSlice


def plotColloid(trj, frame):

    """ 
        Plots a particle system at a given frame for a given lammps trajectory.
        Returns a (fig,ax).
        ----------
        Parameters:
        * trj (pd Dataframe): lammps trajectory.
        * frame
    """

    fig, ax = plt.subplots(figsize=(5, 5)); # Initialize

    trj_particle = trj[trj.type==1]
    trj_trap = trj[trj.type==2]

    xparticle = np.array(trj_particle.loc[idx[frame,:],"x"])
    yparticle = np.array(trj_particle.loc[idx[frame,:],"y"])
    ax.plot(xparticle,yparticle,'o', color='y')

    xtrap = np.array(trj_trap.loc[idx[frame,:],"x"])
    ytrap = np.array(trj_trap.loc[idx[frame,:],"y"])
    ax.plot(xtrap,ytrap,'o', color='g')

    ax.axis("square");
    return fig,ax

def save_colloid_fig(path,trj,frame):
    
    fig, ax = plt.subplots(figsize=(5, 5)); # Initialize

    trj_particle = trj[trj.type==1]
    trj_trap = trj[trj.type==2]

    xparticle = np.array(trj_particle.loc[idx[frame,:],"x"])
    yparticle = np.array(trj_particle.loc[idx[frame,:],"y"])
    ax.plot(xparticle,yparticle,'o', color='y')

    xtrap = np.array(trj_trap.loc[idx[frame,:],"x"])
    ytrap = np.array(trj_trap.loc[idx[frame,:],"y"])
    ax.plot(xtrap,ytrap,'o', color='g')

    ax.axis("square");
    
    fig.savefig(path+str(frame)+".png",dpi=300)
    plt.close(fig)
    return None
    

def classifyVertices(vrt):
    """
        Classifies the vertices in I, II, III, IV, V, VI types.
        Returns a DataFrame
        ----------
        Parameters:
        * vrt (pd Dataframe): Vertices df
    """

    vrt["type"] = np.NaN

    vrt.loc[vrt.eval("coordination==4 & charge == -4"),"type"] = "I"
    vrt.loc[vrt.eval("coordination==4 & charge == -2"),"type"] = "II"
    vrt.loc[vrt.eval("coordination==4 & charge == 0 & (dx**2+dy**2)==0"),"type"] = "III"
    vrt.loc[vrt.eval("coordination==4 & charge == 0 & (dx**2+dy**2)>0"),"type"] = "IV" # Dipolo
    vrt.loc[vrt.eval("coordination==4 & charge == 2"),"type"] = "V"
    vrt.loc[vrt.eval("coordination==4 & charge == 4"),"type"] = "VI"
    return vrt

def getVerticesDict(path):

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
        if not os.path.isfile(filePath):continue
        
        vrt = pd.read_csv(filePath, index_col=[0,1])
        vrt = classifyVertices(vrt)
        vrt = vrt.dropna()
        verticesExp[f"{i}"] = vrt
    return verticesExp

def getVerticesCount(verticesDict):
    
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

def getVerticesAverage(counts,framerate):
    
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
    

    fractions = pd.DataFrame(columns=["time","I","II","III","IV","V","VI"], data = np.zeros((numberFrames,7)))

    for key,experiment in counts.items():
        for vertexType,vrt in experiment.groupby("type"):
            vertexFraction = np.array(vrt.fraction)
            fractions[vertexType] += vertexFraction

    fractions = fractions / numberRealizations
    # fractions["theta"] = time * np.pi/2/60 * (180/np.pi)
    fractions["time"] = time
    return fractions

def getPaintedFrame(trj,ctrj,frame,framerate):

    """
        Visualize the charges of a particular frame.
        ----------
        Parameters:
        * trj (pd Dataframe): lammps trj
        * ctrj (pd Dataframe): lammps ctrj
        * frame
        * framerate
    """
    v = ice.vertices()
    v = v.trj_to_vertices(ctrj.loc[frame])
    currentTime = frame / framerate
    f,ax = plotColloid(trj,frame)
    ax.set_title('t = {} s'.format(currentTime))
    v.display(ax)

def saveAllPaintedFrames(trj,ctrj,frames,framerate,path):
    
    """
        Save all painted frames of a simulation.
        ----------
        Parameters:
        * trj (pd Dataframe): lammps trj
        * ctrj (pd Dataframe): lammps ctrj
        * frames: list of frames to export
        * framerate
    """
    for frame in frames:
        figPath = path + f"{frame}.png";
        try:
            getPaintedFrame(trj,ctrj,frame,framerate);
            print(frame)
            plt.savefig(figPath, dpi=300);
            plt.close()
        except:
            print("skip")
            continue
    return None
    
def get_colloids_from_ctrj(ctrj,particle,trap,particle_radius,a,N):

    """
        Reconstruct the colloidal ice object from simulation parameters.
        ----------
        Parameters:
        * ctrj (pd Dataframe): lammps ctrj without "t" and "type" columns
        * particle: particle simulation object
        * trap: trap simulation object
        * particle_radius
        * a: lattice constant
        * N: system size
    """
    centers = [ row[:3].to_list() * ureg.um for _,row in ctrj.iterrows()]
    directions = [ row[3:6].to_list() * ureg.um for _,row in ctrj.iterrows()]
    arrangement = {
        "centers" : centers,
        "directions" : directions
    }

    col = ice.colloidal_ice(arrangement, particle, trap,
            height_spread = 0, 
            susceptibility_spread = 0.1,
            periodic = True)
    col.region = np.array([[0,0,-3*(particle_radius/a/N).magnitude],[1,1,3*(particle_radius/a/N).magnitude]])*N*a
    
    return col

def get_colloids_from_ctrj2(ctrj,params):

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

    centers = [ row[:3].to_list() * ureg.um for _,row in ctrj.iterrows()]
    directions = [ row[3:6].to_list() * ureg.um for _,row in ctrj.iterrows()]
    arrangement = {
        "centers" : centers,
        "directions" : directions
    }

    col = ice.colloidal_ice(arrangement, particle, trap,
            height_spread = 0, 
            susceptibility_spread = 0.1,
            periodic = True)
    col.region = np.array([[0,0,-3*(particle_radius/a/N).magnitude],[1,1,3*(particle_radius/a/N).magnitude]])*N*a
    
    return col

def count_vertices_single(vrt, column = "type"):
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


def get_vertices_last_frame(path,last_frame=2399):

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
        get_vertices_last_frame(path,last_frame=last_frame-1)
  

    return v.vertices

def get_vertices_at_frame(ctrj,frame):

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


def get_min_from_domain(f,domain):
    
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

def get_pbc_distance(params,xi,xj):
    """
        Returns the real distance between xi and xj in 3D space with PBC.
        ASSUMES POSITION UNITS ARE IN MICROMETERS
        ----------
        Parameters:
        * params: simulation parameters
        * xi: particle position
        * xj: particle position
    """
    
    L = params['size']*params['lattice_constant'].magnitude
    
    # For reference, this line returns the true differences
    # [ xi-xj , yi-yj , zi-zj ]
    xij_pbc = np.array(list(map( lambda xu: get_min_from_domain(np.abs,[xu,xu+L,xu-L]), xi-xj )))
    distance = np.sqrt(sum(xij_pbc**2))
    
    #return distance * params['lattice_constant'].units, xij_pbc/distance
    return distance, xij_pbc/distance

def dipole_pair_energy(params,xi,xj,Bhat = np.array([1,0,0])):
    
    """
        Computes the magnetic dipole interaction energy between a pair of particles.
        Returns a value in pN*nm
       ----------
        Parameters:
        * params: simulation parameters
        * xi: particle position
        * xj: particle position
        * Bhat: Magnetic field direction
    """    
    
    # Get the distance and magnetic moment
    distance, rhat = get_pbc_distance(params,xi,xj)
    
    #dimensional = (- params['mu0']*params['m']**2/4/np.pi/distance**3).to( ureg.pN * ureg.nm)
    dimensional = (params['freedom']/distance**3)
    adimensional = 3*Bhat.dot(rhat)**2 - 1
    
    
    return (dimensional * adimensional)

def calculate_energy(params,sel_particles):
    
    """
        Computes the magnetic dipole total energy in the system
        Returns a value in pN*nm
       ----------
        Parameters:
        * params: simulation parameters
        * sel_particles: some dataframe with id and x y z coordinates at given frame
        * frame
    """   
    #particles = trj[trj.type==1]
    #sel_particles = particles.loc[idx[frame,['x','y','z']]]
    n = len(sel_particles)
    H = sum(
        dipole_pair_energy(params,
                    sel_particles.loc[idx[i]].to_numpy(), 
                    sel_particles.loc[idx[j]].to_numpy())
        for i in range(1,n+1) for j in range(i+1,n+1)
        )
    
    return H

def get_coordinates_at_frame(trj,frame):
    """
        Given a trj file. Retrieves the the positions of the particles at a given frame.
        ----------
        Parameters:
        * trj
        * frame
    """
    
    particles = trj[trj.type==1]
    sel_particles = particles.loc[idx[frame,['x','y','z']]]
    
    return sel_particles

def get_positions_from_ctrj(ctrj):
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
    
    stuff = pd.DataFrame(data =np.column_stack((x,y,z)), columns=['x','y','z'], index=list(range(1,len(ctrj)+1)))
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
        Previously get_ctrj_and_vertices_from_file
        Loads ctrj and vertices object.
        ----------
        Parameters:
        * params
        * data_path
        * size
        * realization
    """
    params['size'] = size
    ctrj = pd.read_csv(os.path.join(data_path,str(size),'ctrj','ctrj1.csv'),index_col=[0,1])
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
        Computes the autocorrelations between some times (start,end)
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
