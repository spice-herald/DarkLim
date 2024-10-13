import numpy as np
from datetime import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import math

curves_dir = 'ExistingLimits/'

# Some figure setup
mpl.rcParams.update({'font.size': 17})
mpl.rcParams.update({'axes.linewidth': 2})
fig, ax = plt.subplots(1, 1, figsize=(14,11))
xmin = 5e-1; xmax = 2e3
ymin = 1e-34; ymax = 1e-18

# Existing limits
m_limit, x_limit = np.loadtxt(curves_dir + 'CRESST_III_2019.txt').transpose()
ax.plot(m_limit*1e3, x_limit, '--', lw=1.5, label='CRESST 2019')
m_limit, x_limit = np.loadtxt(curves_dir + 'CRESSTIII-Si-2022_cm.txt').transpose()
ax.plot(m_limit*1e3, x_limit, '--', lw=1.5, label='CRESST 2022')
m_limit, x_limit = np.loadtxt(curves_dir + 'CRESST_III_2024.txt').transpose()
ax.plot(m_limit*1e3, x_limit, '--', lw=1.5, label='CRESST 2024')
m_limit, x_limit = np.loadtxt(curves_dir + 'LZ_SI_2022.txt').transpose()
ax.plot(m_limit*1e3, x_limit, '--', lw=1.5, label='LZ 2022')

# Simulation

for (baseline_meV, color) in \
    zip([370, 200, 100, 80, 50, 30, 20],
        ['#d62728', '#ff7f0e', '#bcbd22', '#2ca02c', '#17becf', '#1f77b4','#e377c2', '#9467bd', '#8c564b']):
        
    #m_limit, x_limit = np.loadtxt(f'results/silicon_1sec_{baseline_meV}meV_NR/limit.txt').transpose()
    #ax.plot(m_limit*1e3, x_limit, '-', color=color, lw=3, label=f'E_threshold = {baseline_meV*5} meV')

    try:
        m_limit, x_limit = np.loadtxt(f'results/silicon_power_bkgd_1sec_{baseline_meV}meV_NR/limit.txt').transpose()
        ax.plot(m_limit*1e3, x_limit, '-', color=color, lw=2.5, label=f'E_threshold = {baseline_meV*5} meV')
    except FileNotFoundError:
        pass

    #m_limit, x_limit = np.loadtxt(f'results/silicon_1sec_{baseline_meV}meV_phonon_massive/limit.txt').transpose()
    #ax.plot(m_limit*1e3, x_limit, '--', color=color, lw=3)

    try:
        m_limit, x_limit = np.loadtxt(f'results/silicon_power_bkgd_1sec_{baseline_meV}meV_phonon_massive/limit.txt').transpose()
        ax.plot(m_limit*1e3, x_limit, '--', color=color, lw=2.5)
    except FileNotFoundError:
        pass

# Print the current date and time
now = datetime.now()
now = now.strftime("%Y-%m-%d %H:%M:%S")
#ax.text(0.97, 0.97, f'Plot made at {now}', transform=ax.transAxes, ha='right', va='top')

ax.set_xscale('log')
ax.set_xlim([xmin, xmax])
ax.set_yscale('log')
ax.set_ylim([ymin, ymax])
ax.set_xlabel('DM Mass [MeV]')
ax.set_ylabel('DM-nucleon [cm2]')
ax.legend(loc='best', fontsize=17)

#ax.text(2e1, 1e-22, 'Al2O3, 0.4 grams, 1 second', fontsize=25)

fig.tight_layout()
fig.savefig('silicon_NR.png')

