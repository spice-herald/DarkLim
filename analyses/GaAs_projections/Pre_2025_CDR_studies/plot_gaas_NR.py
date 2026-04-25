import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math

curves_dir = 'ExistingLimits/'

# Some figure setup
mpl.rcParams.update({'font.size': 20})
mpl.rcParams.update({'axes.linewidth': 2})
fig, ax = plt.subplots(1, 1, figsize=(11, 11))
xmin = 1e1; xmax = 1e6;
ymin = 1e-48; ymax = 1e-35;

# Existing limits
m_limit, x_limit = np.loadtxt(curves_dir + 'SENSEI_Migdal_NR.txt').transpose()
ax.plot(m_limit, x_limit, '--', lw=1.5, label='SENSEI Migdal')
m_limit, x_limit = np.loadtxt(curves_dir + 'CRESST_III_2019.txt').transpose()
ax.plot(m_limit*1e3, x_limit, '--', lw=1.5, label='CRESST III')
m_limit, x_limit = np.loadtxt(curves_dir + 'Darkside50_Migdal_2023.txt').transpose()
ax.plot(m_limit*1e3, x_limit, '--', lw=1.5, label='Darkside Migdal')
m_limit, x_limit = np.loadtxt(curves_dir + 'LZ_SI_2022.txt').transpose()
ax.plot(m_limit*1e3, x_limit, '--', lw=1.5, label='LZ')

# Simulation
for fold, ls  in zip([2, 3], ['-', ':', '-.']):
    for lce, color in zip([5, 10, 25], ['r', 'b', 'm', 'g', 'y']):
        m_limit, x_limit = np.loadtxt(f'results_gaas_oi_scan_phonon_massless_100days_{fold}fold_lce{lce:02d}_darkphoton/HeRALD_FC_100d_{fold}device_{fold}fold_100mus.txt').transpose()
        ax.plot(m_limit*1e3, x_limit, lw=2, ls=ls, color=color, label=f'{fold}-fold, {lce/100:.02f} LCE per channel')
        m_limit, x_limit = np.loadtxt(f'results_gaas_oi_scan_NR_100days_{fold}fold_lce{lce:02d}/HeRALD_FC_100d_{fold}device_{fold}fold_100mus.txt').transpose()
        ax.plot(m_limit*1e3, x_limit, lw=2, ls='--', color=color, alpha=0.5)

ax.set_xscale('log')
ax.set_xlim([xmin, xmax])
ax.set_yscale('log')
ax.set_ylim([ymin, ymax])
ax.set_xlabel('DM Mass [MeV]')
ax.set_ylabel('DM-nucleon [cm2]')
ax.legend(loc='lower left', fontsize=14)

fig.tight_layout()
fig.savefig('gaas_NR.png')

