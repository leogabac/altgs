# ====================
# Test 03
# Start from an ordered state 
# z->x, let it be, x->z 
# ====================

import os
import sys

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
sys.path.insert(0,'../../../icenumerics/')
sys.path.insert(0,'../auxnumerics/')

import icenumerics as ice 
import concurrent.futures 
import auxiliary as aux

from parameters import params 
from tqdm import tqdm 
import importlib

ureg = ice.ureg 
idx = pd.IndexSlice




def create_simulation(params,trj,size,realization):
    
    global fx
    global fy
    global fz

    N = size 
    a = params['lattice_constant']
    
    params['total_time'] = 180*ureg.s
    params['trap_stiffness'] = 1e-3*ureg.pN/ureg.nm
    params['dt'] = 0.1*ureg.ms
    params['max_field'] = 0*ureg.mT


    sp = ice.spins()
    sp.create_lattice('square', [N,N], lattice_constant=a ,border='periodic')

    particle = ice.particle(radius=params['particle_radius'],
                            susceptibility=params['particle_susceptibility'],
                            diffusion=params['particle_diffusion'],
                            temperature=params['particle_temperature'],
                            density=params['particle_density']
    )

    trap = ice.trap(trap_sep=params['trap_sep'],
                    height=params['trap_height'],
                    stiffness=params['trap_stiffness']
    )

    params['particle'] = particle
    params['trap'] = trap


    col = aux.trj2col(params,trj)

    world = ice.world(
        field=params['max_field'],
        temperature=params['sim_temp'],
        dipole_cutoff=params['sim_dipole_cutoff'],
        boundaries=['p','p','p']
    )

    col.simulation(world,
                   name=f'./lammps_files/trj{realization}',
                   include_timestamp=False,
                   targetdir=r'.',
                   framerate=params['framespersec'],
                   timestep=params['dt'],
                   run_time=params['total_time'],
                   output=['x','y','z','mux','muy','muz'],
                   processors=1
    )

    col.sim.field.fieldx="".join(fx)
    col.sim.field.fieldy="".join(fy)
    col.sim.field.fieldz="".join(fz)

    return col

def run_simulation(params,trj,size,realization):

    col = create_simulation(params,trj,size,realization) 
    col.run_simulation()

def load_simulation(params,trj,data_path,size,realization):

    print(f'Saving {realization}...')
    col = create_simulation(params,trj,size,realization)
    col.sim.base_name = os.path.join(col.sim.dir_name,col.sim.file_name)
    col.sim.script_name = col.sim.base_name+'.lmpin'
    col.sim.input_name = col.sim.base_name+'.lmpdata'
    col.sim.output_name =  col.sim.base_name+'.lammpstrj'
    col.sim.log_name =  col.sim.base_name+'.log'
    trj_path = os.path.join(data_path,'trj')

    try:
        os.mkdir(trj_path)
    except:
        pass
    

    ice.get_ice_trj_low_memory(col,dir_name=trj_path)

# ===== MAIN ROUTINE ===== #

if len(sys.argv) != 2:
    print("Usage: python testXX.py <size>")
    sys.exit(1)

# importing the field
script_name = sys.argv[0][:-3]
module_name = f'{script_name}_field'
module = importlib.import_module(module_name)
fx = module.fx
fy = module.fy
fz = module.fz


SIZE = int(sys.argv[1]) 
REALIZATIONS = [1,2,3,4,5,6,7,8,9,10]
DATA_PATH = f'../../data/{script_name}/'
GSTRJ = pd.read_csv(f'../../data/states/ice/{SIZE}.csv', index_col='id')

try:
    SIZE_PATH = os.path.join(DATA_PATH, str(SIZE))
    os.mkdir(SIZE_PATH)
except:
    pass


with concurrent.futures.ThreadPoolExecutor(max_workers=14) as executor:
    results = list(
        executor.map(
            run_simulation,
            [params] * len(REALIZATIONS),
            [GSTRJ] * len(REALIZATIONS),
            [SIZE] * len(REALIZATIONS),
            REALIZATIONS
        )
    )
for i in REALIZATIONS:
     print(f'===== Realization {i} =====')
     load_simulation(params,GSTRJ,SIZE_PATH,SIZE,i)
