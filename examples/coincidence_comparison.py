import numpy as np
import math
from darklim import constants
import time
from scipy import integrate, interpolate
from darklim import elf
import darklim.sensitivity._sens_est as sens_est
import matplotlib.pyplot as plt
import matplotlib as mpl
import darklim.detector._detector as detector
mpl.rcParams.update({'font.size': 17})
t_start = time.time()

#fun = lambda x: sens_est.drde_wimp_obs(x, 1, 1e-41, 'GaAs', 1)
#E_interp = np.geomspace(0.1, 1000, int(1e4))
#dRdE_interp = fun(E_interp)

fig, ax = plt.subplots(2, 1, figsize=(12,14))



SE = sens_est.SensEst(5.3e-3, 1., 1., 'GaAs', 1.)
fun_2fold_lee = lambda x: sens_est.n_fold_lee(x,m=2,n=2,e0=0.020,R=0.12,w=100e-6) / 5.3e-3
fun_3fold_lee = lambda x: sens_est.n_fold_lee(x,m=3,n=3,e0=0.020,R=0.12,w=100e-6) / 5.3e-3

keV_arr = np.geomspace(0.1e-3, 1, 1000)
dRdE_2fold_arr = fun_2fold_lee(keV_arr)
dRdE_3fold_arr = fun_3fold_lee(keV_arr)
integral_2fold = sum(dRdE_2fold_arr[1:] * np.diff(keV_arr))
integral_3fold = sum(dRdE_3fold_arr[1:] * np.diff(keV_arr))

ax[1].plot(keV_arr, dRdE_2fold_arr, 'c-', label='2-fold coinc. LEE')
ax[1].plot(keV_arr, dRdE_3fold_arr, 'r-', label='3-fold coinc. LEE')
ax[1].text(0.5, 0.18, f'Total rate = {integral_2fold:.3e} counts/kg/day', ha='center', color='c', transform=ax[1].transAxes, fontsize=18)
ax[1].text(0.5, 0.10, f'Total rate = {integral_3fold:.3e} counts/kg/day', ha='center', color='r', transform=ax[1].transAxes, fontsize=18)



fun = elf.get_dRdE_lambda_GaAs_electron(mX_eV=1e8, sigmae=1e-41, mediator='massive', kcut=0, method='grid', withscreening=True, gain=1)
E_interp = np.geomspace(0.1e-3, 1, int(1e5))
dRdE_interp = fun(E_interp)


ax[0].plot(E_interp, dRdE_interp, 'k-', label='Deposited energy, DM-e scattering')

for i, coin in enumerate([1, 2]):

    E_obs_keV, dRdE_obs_DRU, energies_obs_keV = \
        detector.convert_dRdE_dep_to_obs_gaas(E_interp, dRdE_interp, pce=0.40, lce_per_channel=0.10, res=0.17, n_coincidence_light=coin, calorimeter_threshold_eV=0.25*3.7)
    print(f'Out of {len(energies_obs_keV)}, {sum(energies_obs_keV > 0)} are detected')
    print(f'Energies range between {min(energies_obs_keV[energies_obs_keV > 0])} and {max(energies_obs_keV[energies_obs_keV > 0])} keV are detected')
    integral = sum(dRdE_obs_DRU[1:] * np.diff(E_obs_keV))
    print(f'Integral is {integral} counts/kg/day')

    color = ['m', 'g'][i]
    ax[0].plot(E_obs_keV, dRdE_obs_DRU, '-', color=color, alpha=0.5, label=f'Observed energy (Coincidence in {coin} light detectors)')
    ax[0].text(0.5, 0.18 - 0.08*i, f'Total rate = {integral:.3e} counts/kg/day', ha='center', color=color, transform=ax[0].transAxes, fontsize=18)
    
for i in range(2):
    ax[i].set_xscale('log')
    ax[i].set_yscale('log')
    ax[i].set_xlim([1e-3, 1e-1])
    ax[i].set_xlabel('Energy [keV]')
    ax[i].set_ylabel('Rate [DRU]')
    ax[i].legend(loc='upper left')
ax[0].set_ylim([1e-8, 1e2])
ax[1].set_ylim([1e-8, 1e5])
fig.tight_layout()
fig.savefig('fig_coincidence_comparison.png')

t_end = time.time()
print(f'Took {(t_end-t_start)/60} minutes')

