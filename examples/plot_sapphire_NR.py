import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math

curves_dir = 'ExistingLimits/'

# Some figure setup
mpl.rcParams.update({'font.size': 20})
mpl.rcParams.update({'axes.linewidth': 2})
fig, ax = plt.subplots(1, 1, figsize=(11, 11))
xmin = 1e-2; xmax = 1e7;
ymin = 1e-48; ymax = 1e-20;

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
m_limit, x_limit = np.loadtxt('results/sapphire_oi_NR_1h_low_bkgd/HeRALD_FC_0d_1device_1fold_100mus.txt').transpose()
ax.plot(m_limit*1e3, x_limit, 'k--', lw=3, label='NR Low Bkgd')

m_limit, x_limit = np.loadtxt('results/sapphire_oi_phonon_massive_1h_low_bkgd/HeRALD_FC_0d_1device_1fold_100mus.txt').transpose()
ax.plot(m_limit*1e3, x_limit, 'g--', lw=3, label='Multiphonon Low Bkgd')

m_limit, x_limit = np.loadtxt('results/sapphire_oi_NR_1h/HeRALD_FC_0d_1device_1fold_100mus.txt').transpose()
ax.plot(m_limit*1e3, x_limit, 'k-', lw=3, label='NR')

m_limit, x_limit = np.loadtxt('results/sapphire_oi_phonon_massive_1h/HeRALD_FC_0d_1device_1fold_100mus.txt').transpose()
ax.plot(m_limit*1e3, x_limit, 'g-', lw=3, label='Multiphonon')

m_limit, x_limit = np.loadtxt('results/limit.txt').transpose()
ax.plot(m_limit*1e3, x_limit, 'r-', lw=3, label='Test')

ax.set_xscale('log')
ax.set_xlim([xmin, xmax])
ax.set_yscale('log')
ax.set_ylim([ymin, ymax])
ax.set_xlabel('DM Mass [MeV]')
ax.set_ylabel('DM-nucleon [cm2]')
ax.legend(loc='lower left', fontsize=14)

fig.tight_layout()
fig.savefig('sapphire_NR.png')

