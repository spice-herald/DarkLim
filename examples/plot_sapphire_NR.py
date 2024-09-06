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
ymin = 1e-48; ymax = 1e-28;

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
m_limit, x_limit = np.loadtxt('sapphire/results_sapphire_oi_phonon_massless_001_days/HeRALD_FC_1d_1device_1fold_100mus.txt').transpose()
ax.plot(m_limit*1e3, x_limit, 'b--', lw=4, label='1 day, multiphonon')
m_limit, x_limit = np.loadtxt('sapphire/results_sapphire_oi_phonon_massless_010_days/HeRALD_FC_10d_1device_1fold_100mus.txt').transpose()
ax.plot(m_limit*1e3, x_limit, 'r--', lw=4, label='10 days, multiphonon')
# Also massive
m_limit, x_limit = np.loadtxt('sapphire/results_sapphire_oi_scan_NR_001days/HeRALD_FC_1d_1device_1fold_100mus.txt').transpose()
ax.plot(m_limit*1e3, x_limit, 'b:', lw=4, alpha=0.5, label='1 day, NR')
m_limit, x_limit = np.loadtxt('sapphire/results_sapphire_oi_scan_NR_010days/HeRALD_FC_10d_1device_1fold_100mus.txt').transpose()
ax.plot(m_limit*1e3, x_limit, 'r:', lw=4, alpha=0.5, label='1 day, NR')

ax.set_xscale('log')
ax.set_xlim([xmin, xmax])
ax.set_yscale('log')
ax.set_ylim([ymin, ymax])
ax.set_xlabel('DM Mass [MeV]')
ax.set_ylabel('DM-nucleon [cm2]')
ax.legend(loc='lower left', fontsize=14)

fig.tight_layout()
fig.savefig('sapphire_NR.png')

