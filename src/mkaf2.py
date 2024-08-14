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

ureg = ice.ureg
idx = pd.IndexSlice


def save_af2(data_path,N):
    a = params["lattice_constant"]
    sp = ice.spins()
    sp.create_lattice("square",[N,N],lattice_constant=a, border="periodic")

    particle = ice.particle(radius = params["particle_radius"],
                susceptibility = params["particle_susceptibility"],
                diffusion = params["particle_diffusion"],
                temperature = params["particle_temperature"],
                density = params["particle_density"])

    trap = ice.trap(trap_sep = params["trap_sep"],
                height = params["trap_height"],
                stiffness = params["trap_stiffness"])

    col = ice.colloidal_ice(sp, particle, trap,
                            height_spread = params["height_spread"], 
                            susceptibility_spread = params["susceptibility_spread"],
                            periodic = params["isperiodic"])

            
    col.region = np.array([[0,0,-3*(params["particle_radius"]/a/N).magnitude],[1,1,3*(params["particle_radius"]/a/N).magnitude]])*N*a
        
    params['particle'] = particle
    params['trap'] = trap

    col1 = col.copy(deep=True)
    
    pps = int(N**2)
    
    flipsv = [pps + k + n for k in range(0,pps,N) for n in range(0,N,2)]
    flipsh = [0 + k + n for k in range(0,pps,N) for n in range(1,N,2)]
    flipsh2 = [0 + k + n for k in range(0,pps,N*2) for n in range(0,N,1)]
    flips = flipsv + flipsh + flipsh2
    col1 = mc.flip_colloids(col1, indices=flips)

    col1.to_ctrj().to_csv(os.path.join(data_path,'af2',f'{N}.csv'))


data_path = "../data/small_states/"

for size in tqdm(range(10,31,1)):
    save_af2(data_path,size)
