import os
import sys

import numpy as np
import pandas as pd

from tqdm import tqdm

sys.path.insert(0, '../../icenumerics/')
sys.path.insert(0, './auxnumerics/')
sys.path.insert(0, './testing/')
import icenumerics as ice


from thermal_parameters import params
import auxiliary as aux
import montecarlo_colloids as mc

def save_ice(data_path,N):
    # Importing the file
    ctrj = pd.read_csv(os.path.join(data_path,'af4',f'{N}.csv'), index_col=0)

    # Declaring some variables
    particle = ice.particle(
        radius = params['particle_radius'],
        susceptibility = params['particle_susceptibility'],
        diffusion = params["particle_diffusion"],
        temperature = params["particle_temperature"],
        density = params["particle_density"]
    )

    trap = ice.trap(
        trap_sep = params["trap_sep"],
        height = params["trap_height"],
        stiffness = params["trap_stiffness"]
    )

    params['particle'] = particle
    params['trap'] = trap

    col = aux.trj2col(params,ctrj)

    col1 = col.copy(deep=True)

    pps = int(N**2)
    flips = [i for i in range(pps,2*pps)]
    col1 = mc.flip_colloids(col1,indices=flips)

    col1.to_ctrj().to_csv(os.path.join(data_path,'ice',f'{N}.csv'))


data_path = "../data/small_states/"

for size in tqdm(range(10,31,1)):
    save_ice(data_path,size)
