# ====================
# Test 10
# Start from an ordered state
# z->x
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


parser = argparse.ArgumentParser(description="Run a rotation z -> x for many different times")

# flags
parser.add_argument('-s', '--sims', action='store_true', help='run simulations')
parser.add_argument('-v', '--vertices', action='store_true', help='run vertices')
parser.add_argument('-a', '--averages', action='store_true', help='run vertices averages')
parser.add_argument('-k', '--kappa', action='store_true', help='run kappa order parameter')

# positional arguments
parser.add_argument('size', type=str, help='The size input')
parser.add_argument('time', type=str, help='The size input')

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

params['total_time'] = int(args.time) * ureg.s

module = importlib.import_module(module_name)
fx = module.fx
fy = module.fy
fz = module.fz

print(fx)
print(fy)
print(fz)

SIZE = args.size
REALIZATIONS = list(range(1, 11))
FIELDS = list(range(21))

DATA_PATH = f'../../data/{script_name}/'
SIZE_PATH = os.path.join(DATA_PATH, str(SIZE))
GSTRJ = pd.read_csv(f'../../data/states/ice/{SIZE}.csv', index_col='id')

if not os.path.isdir(SIZE_PATH):
    os.makedirs(SIZE_PATH)


if args.sims:
    print('RUNNING SIMULATIONS')
    for B_mag in FIELDS:
        os.system('clear')

        params["max_field"] = B_mag * ureg.mT
        print('Time: ', params['total_time'])
        print('Field:', params["max_field"])
        field_path = os.path.join(
            SIZE_PATH, str(module.rtime), str(B_mag) + 'mT')
        if not os.path.isdir(field_path):
            os.makedirs(field_path)

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
            load_simulation(params, GSTRJ, field_path, int(SIZE), i)

# i will take advantage and not properly using my previous code haha
if args.vertices:
    string_part = f'{script_name}/{SIZE}/{str(module.rtime)}'

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
    global_time = str(module.rtime)
    timepath = os.path.join(SIZE_PATH, global_time)
    FIELDS = next(os.walk(timepath))[1]
    for i, field in tqdm(enumerate(FIELDS)):

        # this part computes the vertices average for all fields,
        # and saves them to a file
        path = os.path.join(timepath, field)
        t, vrt_counts = aux.do_vertices(params, path)

        df = pd.DataFrame(vrt_counts, columns=[
                          'I', 'II', 'III', 'IV', 'V', 'VI'])
        df['time'] = t
        df['field'] = [int(field[:-2])] * len(t)
        df['total_time'] = [global_time] * len(t)

        if i == 0 and int(global_time) == 1:
            print('Making headers')
            df.to_csv(os.path.join(SIZE_PATH, 'average_counts.csv'), index=False)
        else:
            df.to_csv(os.path.join(SIZE_PATH, 'average_counts.csv'), mode='a', index=False, header=False)

if args.kappa:
    field_path = os.path.join(SIZE_PATH, str(module.rtime))
    FIELDS = next(os.walk(field_path))[1]

    for i, field in tqdm(enumerate(FIELDS)):
        path = os.path.join(SIZE_PATH, field, 'trj')

        cumm_kappa = []
        for r in REALIZATIONS:
            # load the trj files
            trj = pd.read_csv(os.path.join(
                path, f'xtrj{r}.csv'), index_col=['frame', 'id'])
            last_frame = trj.index.get_level_values('frame').unique()[-1]
            # take the last frame state
            sel_trj = trj.loc[idx[last_frame, :]]
            # compute the topology and OP
            centers, dirs, rels = vrt.trj2numpy(sel_trj)
            dirs = dirs / np.max(dirs)
            vrt_lattice = vrt.create_lattice(
                params['lattice_constant'].magnitude, int(SIZE))
            idx_lattice = vrt.indices_lattice(
                vrt_lattice, centers, params['lattice_constant'].magnitude, int(SIZE))
            kappa = vrt.charge_op(vrt.get_charge_lattice(idx_lattice, dirs))

            data = [int(field[:-2]), r, last_frame/params['framespersec'].magnitude, kappa]
            cumm_kappa.append(data)

        df = pd.DataFrame(cumm_kappa, columns=[
                          'field', 'realization', 't', 'kappa'])
        if i == 0:
            df.to_csv(os.path.join(field_path, 'kappa.csv'), index=False)
        else:
            df.to_csv(os.path.join(field_path, 'kappa.csv'),
                      mode='a', index=False, header=False)
