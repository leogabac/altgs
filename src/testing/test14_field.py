import numpy as np
from parameters import params

fx = [
    'v_Bmag*sin(PI/2/3e6*time)*(time<=3e6)+',
    'v_Bmag*(time>3e6)*(time<=13e6)+',
    'v_Bmag*cos(-PI/2/3e6*(time-13e6))*(time>13e6)*(time<=16e6)'
]

fy = [
    '0*v_Bmag*(time<=3e6)+',
    '0*v_Bmag*(time>3e6)*(time<=13e6)+',
    '0*v_Bmag*(time>13e6)*(time<=16e6)'
]

fz = [
    'v_Bmag*cos(PI/2/3e6*time)*(time<=3e6)+',
    '0*v_Bmag*(time>3e6)*(time<=13e6)+',
    'v_Bmag*-1*sin(-PI/2/3e6*(time-13e6))*(time>13e6)*(time<=16e6)'
]
