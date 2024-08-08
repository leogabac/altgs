# ====================
# Test 16
# Start from an ordered state
# Fix one particle in place
# ====================

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, '../../../icenumerics/')
sys.path.insert(0, '../auxnumerics/')

import icenumerics as ice
import concurrent.futures
import auxiliary as aux
import vertices as vrt

from parameters import params
from tqdm import tqdm
import importlib
import argparse

ureg = ice.ureg
idx = pd.IndexSlice


def create_simulation(params, trj, size, realization):
    global fx
    global fy
    global fz

    N = size
    a = params['lattice_constant']

    sp = ice.spins()
    sp.create_lattice('square', [N,N], lattice_constant=a, border='periodic')

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

    fixed_trap = ice.trap(trap_sep=0*params['trap_sep'],
                          height=params['trap_height'],
                          stiffness=params['trap_stiffness']
    )

    # make the structure of the colloid
    centers = [row[['x','y','z']].to_list() * ureg.um for _,row in trj.iterrows()]
    directions = [row[['dx','dy','dz']].to_list() * ureg.um for _,row in trj.iterrows()]
    arrangement = {"centers": centers,
                   "directions": directions
                }

    # select one particle to fix
    idx = 42
    traps = [fixed_trap if (i == idx) else trap for i in range(len(centers))]

    col = ice.colloidal_ice(arrangement, particle, traps,
                            height_spread=0,
                            susceptibility_spread=0.1,
                            periodic=True)

    particle_radius = params['particle_radius']
    col.region = np.array([[0,0,-3*(particle_radius/a/N).magnitude],[1,1,3*(particle_radius/a/N).magnitude]])*N*a

    # move the particle a tiny bit
    col[idx].colloid += col[idx].direction * params['lattice_constant']/8
    col[idx].center += col[idx].direction * params['trap_sep']/2

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

    col.sim.field.fieldx = "".join(fx)
    col.sim.field.fieldy = "".join(fy)
    col.sim.field.fieldz = "".join(fz)

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
    col.sim.output_name = col.sim.base_name+'.lammpstrj'
    col.sim.log_name = col.sim.base_name+'.log'

    trj_path = os.path.join(data_path,'trj')

    col.load_simulation()

    try:
        os.mkdir(trj_path)
    except:
        pass

    # col.trj.to_csv(os.path.join(trj_path,f'{realization}.csv'))
    trj = ice.get_ice_trj(col.trj, bounds=col.bnd, trap_types=[2,3])
    trj.to_csv(os.path.join(trj_path,f'trj{realization}.csv'))
    ice.get_ice_trj_low_memory(col,dir_name=trj_path)

# ===== MAIN ROUTINE ===== #


parser = argparse.ArgumentParser(description="to x axis, relaxation and go back")

# flags
parser.add_argument('-s', '--sims', action='store_true', help='run simulations')
parser.add_argument('-v', '--vertices', action='store_true', help='run vertices')
parser.add_argument('-a', '--averages', action='store_true', help='run vertices averages')
parser.add_argument('-k', '--kappa', action='store_true', help='run order paramater')
parser.add_argument('-r', '--rparallel', action='store_true', help='get rparallel')

# positional arguments
parser.add_argument('size', type=str, help='The size input')

args = parser.parse_args()

sth_passed = any([args.sims, args.vertices, args.averages, args.kappa, args.rparallel])
if not sth_passed:
    args.sims = True
    args.vertices = True
    args.averages = True
    args.kappa = True
    args.rparallel = True


# importing the field

params['total_time'] = 9 * ureg.s
params["max_field"] = 20 * ureg.mT

script_name = sys.argv[0][:-3]
module_name = f'{script_name}_field'
module = importlib.import_module(module_name)
fx = module.fx
fy = module.fy
fz = module.fz


SIZE = int(args.size)
REALIZATIONS = list(range(1, 11))
DATA_PATH = f'../../data/{script_name}/'
SIZE_PATH = os.path.join(DATA_PATH, str(SIZE))
GSTRJ = pd.read_csv(f'../../data/states/ice/{SIZE}.csv', index_col='id')

if not os.path.isdir(SIZE_PATH):
    os.makedirs(SIZE_PATH)

if args.sims:

    print('RUNNING SIMULATIONS')
    os.system('clear')

    print('Time: ', params['total_time'])
    print('Field:', params["max_field"])

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

if args.vertices:
    string_part = f'{script_name}'

    # os.system('clear')
    os.chdir('../')
    # print('CLEANING')
    # os.system(f'python cleaning.py {string_part}')

    os.system('clear')
    print('VERTICES')
    # os.system(f'python compute_vertices.py {string_part}')
    os.system(f'python bulk_vertices.py')
    os.chdir('./testing')

if args.averages:
    t, vrt_counts = aux.do_vertices(params, SIZE_PATH)

    df = pd.DataFrame(vrt_counts, columns=['I', 'II', 'III', 'IV', 'V', 'VI'])
    df['time'] = t
    df.to_csv(os.path.join(SIZE_PATH, 'average_counts.csv'), index=False)


if args.kappa:
    path = os.path.join(SIZE_PATH, 'trj')

    cumm_kappa = []
    for r in REALIZATIONS:
        # load the trj files
        trj = pd.read_csv(os.path.join(path, f'xtrj{r}.csv'), index_col=['frame', 'id'])
        frames = trj.index.get_level_values('frame').unique()
        for frame in frames:
            # take the last frame state
            sel_trj = trj.loc[idx[frame, :]]
            # compute the topology and OP
            centers, dirs, rels = vrt.trj2numpy(sel_trj)
            dirs = dirs / np.max(dirs)
            vrt_lattice = vrt.create_lattice(params['lattice_constant'].magnitude, int(SIZE))
            idx_lattice = vrt.indices_lattice(vrt_lattice, centers, params['lattice_constant'].magnitude, int(SIZE))
            kappa = vrt.charge_op(vrt.get_charge_lattice(idx_lattice, dirs))

            data = [r, frame / params['framespersec'].magnitude, kappa]
            cumm_kappa.append(data)

    df = pd.DataFrame(cumm_kappa, columns=['realization', 't', 'kappa'])
    df.to_csv(os.path.join(SIZE_PATH, 'kappa.csv'), index=False)

if args.rparallel:
    path = os.path.join(SIZE_PATH, 'trj')
    make_headers = True
    for r in REALIZATIONS:
        # load the trj files
        trj = pd.read_csv(os.path.join(path, f'xtrj{r}.csv'), index_col=['frame', 'id'])

        for pid, cdf in trj.groupby('id'):

            rel = cdf[['cx','cy','cz']].to_numpy()
            spin = cdf[['dx','dy','dz']].to_numpy()
            spin = spin/np.linalg.norm(spin, axis=1)[:,np.newaxis]

            # rparallel = np.sum(rel*np.abs(spin), axis=1)
            rparallel = np.sum(spin*np.abs(spin), axis=1)
            time = cdf.t.to_numpy()

            df = pd.DataFrame(rparallel, columns=['rp'])
            df['t'] = time
            df['id'] = [pid]*len(df)
            df['realization'] = [r]*len(df)

            if make_headers:
                df.to_csv(os.path.join(SIZE_PATH,'rparallel.csv'),index=False)
                make_headers = False
            else:
                df.to_csv(os.path.join(SIZE_PATH,'rparallel.csv'),mode='a',index=False,header=False)
