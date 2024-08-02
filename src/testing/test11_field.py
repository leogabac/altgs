import numpy as np
from parameters import params

rtime = params['total_time'].magnitude

fx = [
    f'v_Bmag*sin(PI/2/{rtime}e6*time)*(time<={rtime}e6)'
]

fy = [
    f'0*v_Bmag*(time<={rtime}e6)'
]

fz = [
    f'v_Bmag*cos(PI/2/{rtime}e6*time)*(time<={rtime}e6)'
]



