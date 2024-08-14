import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from tqdm import tqdm
sys.path.insert(0, '../../icenumerics/')
sys.path.insert(0, './auxnumerics/')
sys.path.insert(0, './testing/')
import icenumerics as ice
import importlib


from thermal_parameters import params
import auxiliary as aux
import montecarlo_colloids as mc

ureg = ice.ureg
idx = pd.IndexSlice

def lmp_to_numpy(string):
    string = string.replace('sin','np.sin')
    string = string.replace('cos','np.cos')
    string = string.replace('PI','np.pi')
    string = string.replace('e6','')
    return string

def join_field_component(fi):
    return "".join( list( map(lmp_to_numpy,fi) ) )

# matplotlib latex setup
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})


# retrieving input
if len(sys.argv) != 3:
    print("Usage: python run.py <testXX> <size>")
    sys.exit(1)

script_name = sys.argv[0][:-3]
testno = sys.argv[1]
size = int(sys.argv[2])

# importing files
data_path = f'../data/{testno}/{size}'
targetdir = os.path.join(data_path,'frames')

if not os.path.isdir(targetdir):
    os.mkdir(targetdir)

# dinamically import the correct magnetic fields
module_name = f'{testno}_field'
module = importlib.import_module(module_name)
fx = module.fx
fy = module.fy
fz = module.fz

# importing data
trj = pd.read_csv(os.path.join(data_path,'trj','xtrj2.csv'),index_col=['frame','id'])
vertices = pd.read_csv(os.path.join(data_path,'vertices','vertices2.csv'),index_col=['frame','vertex'])
time = trj.t.unique()
# trj = trj.drop(['type'],axis=1)
v = ice.vertices()
v.vertices = vertices

frames = trj.index.get_level_values('frame').unique().to_list()

# Evaluating the magnetic fields
v_Bmag = params['max_field'].magnitude
nfx = eval(join_field_component(fx))
nfy = eval(join_field_component(fy))
nfz = eval(join_field_component(fz))


i = 0
for frame in tqdm(frames[::5]):

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(121)

    region_limit = size*params['lattice_constant'].magnitude
    ice.draw_frame(trj, frame_no=frame,
               region=[0,region_limit,0,region_limit],
               radius=params["particle_radius"].magnitude,
               cutoff=params["trap_sep"].magnitude/2,
               particle_color='#75b7ea',
               trap_color='gray',
               ax = ax)

    v.display(ax,sl=frame,dpl_scale=0.5,dpl_width=2.5,circle_scale=0.5)
    ax.set_title(f'$t = {frame/20:1.2f}$')

    ax2  = fig.add_subplot(122,projection='3d')
    ax2.quiver(
    0,0,0,  # Starting point
    nfx[frame], nfy[frame], nfz[frame],  # Direction
    length=1,  # Length of the arrow
    color='r',  # Color of the arrow
    arrow_length_ratio=0.1  # Ratio of the arrow head
    )

    ax2.quiver(
    0,0,0,
    nfx[frame], 0, 0,
    color='g',  
    arrow_length_ratio=0.1
    )

    ax2.quiver(
    nfx[frame],0,0,
    0, 1, 0,
    length = nfy[frame],
    color='g',
    arrow_length_ratio=0.1
    )

    ax2.quiver(
    nfx[frame],nfy[frame],0,
    0, 0, 1,
    length = nfz[frame],
    color='g',
    arrow_length_ratio=0.1
    )

    ax2.set_xlim([0, 20])
    ax2.set_ylim([0, 20])
    ax2.set_zlim([0, 20])

    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_zticks([])

    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_zticklabels([])

    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$y$')
    ax2.set_zlabel('$z$')

    fig.savefig(os.path.join(targetdir,f'{i}.jpeg'), dpi=300, bbox_inches='tight')
    plt.close()

    i = i + 1   

os.chdir(targetdir)
os.system(r'ffmpeg -framerate 10 -i %d.jpeg -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p output.mp4')
