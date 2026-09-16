import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math

curves_dir = 'ExistingLimits/'

# Some figure setup
mpl.rcParams.update({'font.size': 20})
mpl.rcParams.update({'axes.linewidth': 2})
fig, ax = plt.subplots(1, 1, figsize=(11, 11))
xmin = 1e-2; xmax = 1e4;
ymin = 1e-41; ymax = 1e-28;

# Existing limits
m_limit, x_limit = np.loadtxt(curves_dir + 'DAMIC_M_ER_Massless.txt').transpose()
ax.plot(m_limit, x_limit, '--', lw=1.5, label='DAMIC-M')
m_limit, x_limit = np.loadtxt(curves_dir + 'SENSEI_MINOS_ER_Massless.txt').transpose()
ax.plot(m_limit, x_limit, '--', lw=1.5, label='SENSEI MINOS')
m_limit, x_limit = np.loadtxt(curves_dir + 'SENSEI_SNOLAB_ER_Massless.txt').transpose()
ax.plot(m_limit, x_limit, '--', lw=1.5, label='SENSEI SNOLAB')
#m_limit, x_limit = np.loadtxt(curves_dir + 'SENSEI_SNOLAB_Solar_Reflection_ER_Massless.txt').transpose()
#ax.plot(m_limit, x_limit, '--', lw=1.5, label='SENSEI SNOLAB (Solar Refl.)')
#m_limit, x_limit = np.loadtxt(curves_dir + 'XENON1T_S2Only_Solar_Reflected_ER_Massless.txt').transpose()
#ax.plot(m_limit, x_limit, '--', lw=1.5, label='XENON1T S2 Only (Solar Refl.)')
m_limit, x_limit = np.loadtxt(curves_dir + 'protoSENSEI_MINOS_ER_Massless.txt').transpose()
ax.plot(m_limit, x_limit, '--', lw=1.5, label='protoSENSEI')
m_limit, x_limit = np.loadtxt(curves_dir + 'Freeze_in_ER_Massless.txt').transpose()
ax.plot(m_limit, x_limit, '-', lw=8, label='Freeze-in')


# Simulation
for fold, ls  in zip([2, 3], ['-', ':', '-.']):
    for lce, color in zip([5, 10, 25], ['r', 'b', 'm', 'g', 'y']):
        m_limit, x_limit = np.loadtxt(f'gaas/results_gaas_oi_scan_electron_massless_100days_{fold}fold_lce{lce:02d}/HeRALD_FC_100d_{fold}device_{fold}fold_100mus.txt').transpose()
        ax.plot(m_limit*1e3, x_limit, lw=4, ls=ls, color=color, label=f'{fold}-fold, {lce/100:.02f} LCE per channel')

m_limit, x_limit = np.loadtxt('gaas/results_gaas_oi_scan_electron_massless_300days_3fold_lce10/HeRALD_FC_300d_3device_3fold_100mus.txt').transpose()
ax.plot(m_limit*1e3, x_limit, lw=4, ls=':', color='k', label=f'300 days')

m_limit, x_limit = np.loadtxt('results_gaas_oi_electron_massless_100days_3fold_lce10/HeRALD_FC_100d_3device_3fold_100mus.txt').transpose()
ax.plot(m_limit*1e3, x_limit, lw=6, ls='-', color='k', label=f'New')


ax.set_xscale('log')
ax.set_xlim([xmin, xmax])
ax.set_yscale('log')
ax.set_ylim([ymin, ymax])
ax.set_xlabel('DM Mass [MeV]')
ax.set_ylabel('DM-electron cross-section massless [cm2]')
ax.legend(loc='best', fontsize=14)

fig.tight_layout()
fig.savefig('gaas_ER_massless.png')
