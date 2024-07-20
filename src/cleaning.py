
import os
import sys

import numpy as np
import pandas as pd

from tqdm import tqdm
from IPython.display import clear_output

import matplotlib as mpl 
import matplotlib.pyplot as plt

sys.path.insert(0, '../../icenumerics/')
sys.path.insert(0, './auxnumerics/')
import icenumerics as ice

import auxiliary as aux
import chirality_tools as chir
ureg = ice.ureg

idx = pd.IndexSlice

def is_low_memory(df):
    """
        Determines if the file comes from low_memory
    """
    
    if 'mux' in df.columns:
        return True
    else:
        return False

def is_regular(df):
    """
        Determines if the file was regularly generated
    """
    
    if 'type' in df.columns:
        return True
    else:
        return False

def frames_dict(df):
    
    oldframes = df.index.get_level_values('frame').unique().to_list()
    
    mapping = [i for i,frame in enumerate(oldframes)]
    return mapping

def id_dict(df):
    oldid = df.index.get_level_values('id').unique().to_list()
    mapping = [i for i,oid in enumerate(oldid)]
    return mapping
    

def clean_data(sim_path,realization):
    
    # load the file
    filepath = os.path.join(sim_path,'trj',f'trj{realization}.csv')
    
    # If the file does not exist, just return
    if not os.path.isfile(filepath):
        return None
    else:
        print("Cleaning...",filepath)
    
    df = pd.read_csv(filepath,index_col=[0,1])
    
    if is_low_memory(df):
        
        # delete last row
        last_frame = df.index.get_level_values('frame').unique().to_list()[-1]
        dfclean = df.loc[idx[:last_frame-1,:]].drop(columns={'mux','muy','muz'})
        
        # remap the frames
        m1 = frames_dict(dfclean)
        m2 = id_dict(dfclean)
        dfclean.index = pd.MultiIndex.from_tuples([ (frame,i) for frame in m1 for i in m2 ],  names = ['frame','id'])
        dfclean.to_csv(os.path.join(sim_path,'trj',f'xtrj{realization}.csv'))
    
    elif is_regular(df):
        dfclean = df.drop(columns={'type'})
        dfclean.to_csv(os.path.join(sim_path,'trj',f'xtrj{realization}.csv'))
        
    else:
        print("Skip")
    
if len(sys.argv) != 2:
    print("Usage: python cleaning.py <testXX>")
    sys.exit(1)

script_name = sys.argv[0][:-3]
usr_input = sys.argv[1]

DRIVE = f'../data/{usr_input}/'
SIZES = next(os.walk(DRIVE))[1]

for size in SIZES:
    ctrjpath = os.path.join(DRIVE,size)
    for r in range(1,11):
        trypath = os.path.join(ctrjpath,'ctrj',f'xtrj{r}.csv')
        if os.path.isfile(trypath):
            continue
        else:
            clean_data(ctrjpath,r)
        
