import os
import sys
import numpy as np
sys.path.insert(0, '../../icenumerics/')
import icenumerics as ice
ureg = ice.ureg

TOTAL_TIME = 20*ureg.s

params = {
    "particle_radius":1.4*ureg.um,
    "particle_susceptibility":0.4,
    "particle_diffusion":0.14*ureg.um**2/ureg.s,
    "particle_temperature":300*ureg.K,
    "particle_density":1000*ureg.kg/ureg.m**3,

    "trap_sep":3*ureg.um,
    "trap_height":8*ureg.pN*ureg.nm,
    "trap_stiffness":100e-3*ureg.pN/ureg.nm,
    "height_spread":0,
    "susceptibility_spread":0,
    "isperiodic":True,

    "total_time":TOTAL_TIME,
    "framespersec":20*ureg.Hz,
    "dt":0.1*ureg.ms,
    "max_field":10*ureg.mT,
    "sim_temp":300*ureg.K,
    "sim_dipole_cutoff":40*ureg.um,
}

params["lattice_constant"] = params["trap_sep"]+(2*params["particle_radius"]+1*ureg.um)*np.sqrt(2)
params['size'] = 30


params['mu0'] = (4*np.pi)*1e-7 * ureg.H/ureg.m
params['m'] = np.pi * (2*params['particle_radius'])**3 *params['particle_susceptibility']*params['max_field']/6/params['mu0']
params['kb'] = 1.380649e-23 * ureg.J / ureg.K
params['kbT'] = (params['kb'] * params['sim_temp']).to(ureg.nm * ureg.pN)
params['freedom'] = - (params['mu0']*params['m']**2/4/np.pi).to(ureg.pN * ureg.nm * ureg.um**3).magnitude
