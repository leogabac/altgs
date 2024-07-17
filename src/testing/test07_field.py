import numpy as np
from parameters import params

fx = [
    'v_Bmag/120e6*time*(time<=120e6)',
]

fy = [
    '0*v_Bmag*(time<=120e6)',
]

fz = [
    '0*v_Bmag*(time<=120e6)'
]


