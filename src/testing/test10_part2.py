
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

# now here the idea is to compute the averaged vertices for all fields
# and save them as a csvif len(sys.argv) != 2:

if len(sys.argv) != 2:
    print("Usage: python testXX_part2.py <size>")
    sys.exit(1)

SCRIPT_NAME = sys.argv[0][:-3]
SIZE = int(sys.argv[1]) 
DATA_PATH = f'../../data/test10/{SIZE}'
FIELDS = next(os.walk(DATA_PATH))[1]

for i,field in tqdm(enumerate(FIELDS)):

    # this part computes the vertices average for all fields, and saves them to a file
    path = os.path.join(DATA_PATH,field)
    t, vrt_counts = aux.do_vertices(params,path)

    df = pd.DataFrame(vrt_counts, columns = ['I','II','III','IV','V','VI'])
    df['time'] = t
    df['field'] = [int(field[:-2])] * len(t)

    if i==0:
        df.to_csv(os.path.join(DATA_PATH,'average_counts.csv'),index=False)
    else:
        df.to_csv(os.path.join(DATA_PATH,'average_counts.csv'),mode='a',index=False,header=False)


