# ====================
# Test 13
# Start from an ordered state
# Instantaneous quench but in the y direction
# ====================

import os
import sys

import numpy as np
import pandas as pd
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

    params['size'] = size
    N = size
    a = params['lattice_constant']

    print('time sanity check: ', params['total_time'])

    sp = ice.spins()
    sp.create_lattice('square', [N, N], lattice_constant=a, border='periodic')

    particle = ice.particle(radius=params['particle_radius'],
                            susceptibility=params['particle_susceptibility'],
                            diffusion=params['particle_diffusion'],
                            temperature=params['particle_temperature'],
                            density=params['particle_density'])

    trap = ice.trap(trap_sep=params['trap_sep'],
                    height=params['trap_height'],
                    stiffness=params['trap_stiffness'])

    params['particle'] = particle
    params['trap'] = trap

    col = aux.trj2col(params, trj)

    world = ice.world(
        field=params['max_field'],
        temperature=params['sim_temp'],
        dipole_cutoff=params['sim_dipole_cutoff'],
        boundaries=['p', 'p', 'p']
    )

    col.simulation(world,
                   name=f'./lammps_files/trj{realization}',
                   include_timestamp=False,
                   targetdir=r'.',
                   framerate=params['framespersec'],
                   timestep=params['dt'],
                   run_time=params['total_time'],
                   output=['x', 'y', 'z', 'mux', 'muy', 'muz'],
                   processors=1)

    col.sim.field.fieldx = "".join(fx)
    col.sim.field.fieldy = "".join(fy)
    col.sim.field.fieldz = "".join(fz)

    return col


def run_simulation(params, trj, size, realization):

    col = create_simulation(params, trj, size, realization)
    col.run_simulation()


def load_simulation(params, trj, data_path, size, realization):

    print(f'Saving {realization}...')
    col = create_simulation(params, trj, size, realization)
    col.sim.base_name = os.path.join(col.sim.dir_name, col.sim.file_name)
    col.sim.script_name = col.sim.base_name+'.lmpin'
    col.sim.input_name = col.sim.base_name+'.lmpdata'
    col.sim.output_name = col.sim.base_name+'.lammpstrj'
    col.sim.log_name = col.sim.base_name+'.log'
    trj_path = os.path.join(data_path, 'trj')

    try:
        os.mkdir(trj_path)
    except:
        pass

    ice.get_ice_trj_low_memory(col, dir_name=trj_path)

# ===== MAIN ROUTINE ===== #


parser = argparse.ArgumentParser(description="Heavyside step function field profile but on the y axis")

# flags
parser.add_argument('-s', '--sims', action='store_true', help='run simulations')
parser.add_argument('-v', '--vertices', action='store_true', help='run vertices')
parser.add_argument('-a', '--averages', action='store_true', help='run vertices averages')
parser.add_argument('-k', '--kappa', action='store_true', help='run kappa order parameter')

# positional arguments
parser.add_argument('size', type=str, help='The size input')

args = parser.parse_args()

sth_passed = any([args.sims, args.vertices, args.averages, args.kappa])
if not sth_passed:
    args.sims = True
    args.vertices = True
    args.averages = True
    args.kappa = True

# importing the field
script_name = sys.argv[0][:-3]
module_name = f'{script_name}_field'

params['total_time'] = 20 * ureg.s
params["max_field"] = 20 * ureg.mT

module = importlib.import_module(module_name)
fx = module.fx
fy = module.fy
fz = module.fz

print(fx)
print(fy)
print(fz)

SIZE = args.size
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
                [int(SIZE)] * len(REALIZATIONS),
                REALIZATIONS
            )
        )
    for i in REALIZATIONS:
        print(f'===== Realization {i} =====')
        load_simulation(params, GSTRJ, SIZE_PATH, int(SIZE), i)

if args.vertices:
    string_part = f'{script_name}'

    os.system('clear')
    os.chdir('../')
    print('CLEANING')
    os.system(f'python cleaning.py {string_part}')

    os.system('clear')
    print('VERTICES')
    os.system(f'python compute_vertices.py {string_part}')
    os.chdir('./testing')

# this might be broken, but i will test post sim
if args.averages:
    t, vrt_counts = aux.do_vertices(params, SIZE_PATH)

    df = pd.DataFrame(vrt_counts, columns=['I', 'II', 'III', 'IV', 'V', 'VI'])
    df['time'] = t
    df.to_csv(os.path.join(SIZE_PATH, 'average_counts.csv'), index=False)
