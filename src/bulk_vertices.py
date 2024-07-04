# ============================================================= 
# Script to compute the vertices for all sizes
# God bless whoever reads this code
# Author: leogabac
# ============================================================= 


import os
import sys

import numpy as np
import pandas as pd

from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.insert(0, '../../icenumerics/')
sys.path.insert(0, './auxnumerics/')
sys.path.insert(0, './testing/')
import icenumerics as ice

from parameters import params

ureg = ice.ureg
idx = pd.IndexSlice

SIZES = ['10'] 
DATA_PATH = r'../data/test01/'
REALIZATIONS = 10

print("COMPUTING VERTICES")
for strsize in SIZES: 
    print(f"===== size {strsize} =====")
    params['size'] = int(strsize)
    
    trj_path = os.path.join(DATA_PATH,strsize,"trj")
    vrt_path = os.path.join(DATA_PATH,strsize,"vertices")
    try:
        os.mkdir(vrt_path)
    except:
        pass

    for i in range(1,REALIZATIONS+1):

        trj_file = os.path.join(trj_path,f"xtrj{i}.csv")
        vrt_file = os.path.join(vrt_path,f"vertices{i}.csv")
        
        
        if os.path.isfile(vrt_file):
            print(f'{vrt_file}... exists')
            continue
        
        # Importing files
        print(f"working on ... {trj_file}")
        try:
            trj_raw = pd.read_csv(trj_file, index_col=[0,1])
        except:
            print(f"skipping trj")
            continue

        # Doing shit with the vertices
        v = ice.vertices()
        frames = trj_raw.index.get_level_values("frame").unique()

        v.trj_to_vertices(trj_raw.loc[frames[::10]])

        print(f"saving... {vrt_file}")
        v.vertices.to_csv(vrt_file)
