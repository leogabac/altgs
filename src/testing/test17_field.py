import numpy as np
from parameters import params

fx = [
    f'0*v_Bmag*(time<=10e6)+',
    f'v_Bmag*(time>10e6)*(time<=20e6)'
]

fy = [
    f'0*v_Bmag*time'
]

fz = [
    f'0*v_Bmag*time'
]
