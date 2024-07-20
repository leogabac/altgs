import numpy as np
from parameters import params

fx = [
    'v_Bmag*sin(PI/2/60e6*time)*(time<=60e6)'
]

fy = [
    '0*v_Bmag*(time<=60e6)'
]

fz = [
    'v_Bmag*cos(PI/2/60e6*time)*(time<=60e6)'
]



