import darklim
import numpy as np
from datetime import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import multiprocessing as mp
mpl.rcParams.update({'font.size': 15})

Eth = 1e-3
mass_det_kg = 10e-3
t_days = 1 

E_recoil_keV = np.geomspace(Eth, 100, 10000)
m_limit_GeV = np.geomspace(8e-3, 1e3, 96)
x_limit_cm2 = np.zeros_like(m_limit_GeV)
x_multiphonon_massive_cm2 = np.zeros_like(m_limit_GeV)
x_multiphonon_massless_cm2 = np.zeros_like(m_limit_GeV)
sigma0 = 1e-35

fig, ax = plt.subplots(1, 1, figsize=(10,6))

# Existing limits
curves_dir = 'ExistingLimits/'
m_limit, x_limit = np.loadtxt(curves_dir + 'CRESST_III_2019.txt').transpose()
#ax.plot(m_limit, x_limit, '--', lw=1.5)#, label='CRESST 2019')
m_limit, x_limit = np.loadtxt(curves_dir + 'CRESSTIII-Si-2022_cm.txt').transpose()
#ax.plot(m_limit, x_limit, '--', lw=1.5)#, label='CRESST 2022')
m_limit, x_limit = np.loadtxt(curves_dir + 'CRESST_III_2024.txt').transpose()
#ax.plot(m_limit, x_limit, '--', lw=1.5)#, label='CRESST 2024')

colors = ['#d62728', '#ff7f0e', '#bcbd22', '#2ca02c', '#17becf', '#1f77b4','#e377c2', '#9467bd', '#8c564b']

def process_mass(m, sigma0):
    
    rate_DRU = darklim.limit.drde(E_recoil_keV, m, sigma0, tm='Al2O3')

    n_events = sum(rate_DRU[:-1] * np.diff(E_recoil_keV)) * mass_det_kg * t_days

    if n_events > 0:
        x = sigma0 / n_events * 2.3
    else:
        x = np.inf


    fxn = darklim.elf.get_dRdE_lambda_Al2O3_phonon(mX_eV=m*1e9, mediator='massive', sigman=sigma0, dark_photon=False, suppress_darkelf_output=False, gain=1.)
    rate_DRU = np.array([fxn(e) for e in E_recoil_keV])

    n_events = sum(rate_DRU[:-1] * np.diff(E_recoil_keV)) * mass_det_kg * t_days

    if n_events > 0:
        x_massive = sigma0 / n_events * 2.3
    else:
        x_massive = np.inf


    fxn = darklim.elf.get_dRdE_lambda_Al2O3_phonon(mX_eV=m*1e9, mediator='massless', sigman=sigma0, dark_photon=False, suppress_darkelf_output=False, gain=1.)
    rate_DRU = np.array([fxn(e) for e in E_recoil_keV])

    n_events = sum(rate_DRU[:-1] * np.diff(E_recoil_keV)) * mass_det_kg * t_days

    if n_events > 0:
        x_massless = sigma0 / n_events * 2.3
    else:
        x_massless = np.inf


    return x, x_massless, x_massive


# Main parallel execution block
with mp.Pool(processes=24) as pool:
    results = pool.starmap(process_mass, [(mass, sigma0) for mass in m_limit_GeV])


for i, result in enumerate(results):
    x_limit_cm2[i] = result[0]
    x_multiphonon_massless_cm2[i] = result[1]
    x_multiphonon_massive_cm2[i] = result[2]

ax.plot(m_limit_GeV, x_limit_cm2, 'k-', lw=3, label='Spin-Indep NR')
ax.plot(m_limit_GeV, x_multiphonon_massless_cm2, 'b-', lw=3, label='Multiphonon, Massless Mediator')
ax.plot(m_limit_GeV, x_multiphonon_massive_cm2, 'r-', lw=3, label='Multiphonon, Massive Mediator')


ax.set_xscale('log')
ax.set_yscale('log')
#ax.set_xlim([1e-3, 1e3])
#ax.set_ylim([1e-40, 1e-34])
ax.set_xlabel('DM Mass [GeV]')
ax.set_ylabel('DM-nucleon [cm2]')
ax.text(5e-1, 1e-29, '10 g Al2O3, 1 day, 1 eV threshold')
ax.legend(loc='lower left')

fig.tight_layout()
fig.savefig('background_free.png')

