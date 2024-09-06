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
m_limit, x_limit = np.loadtxt(curves_dir + 'DAMIC_M_ER_Massive.txt').transpose()
ax.plot(m_limit, x_limit, '--', lw=1.5, label='DAMIC-M')
m_limit, x_limit = np.loadtxt(curves_dir + 'Darkside50_ER_Massive.txt').transpose()
ax.plot(m_limit, x_limit, '--', lw=1.5, label='Darkside 50')
m_limit, x_limit = np.loadtxt(curves_dir + 'SENSEI_SNOLAB_ER_Massive.txt').transpose()
ax.plot(m_limit, x_limit, '--', lw=1.5, label='SENSEI SNOLAB')
#m_limit, x_limit = np.loadtxt(curves_dir + 'SENSEI_SNOLAB_Solar_Reflection_ER_Massive.txt').transpose()
#ax.plot(m_limit, x_limit, '--', lw=1.5, label='SENSEI SNOLAB (Solar Refl.)')
m_limit, x_limit = np.loadtxt(curves_dir + 'XENON1T_S2Only_ER_Massive.txt').transpose()
ax.plot(m_limit, x_limit, '--', lw=1.5, label='XENON1T S2only')
#m_limit, x_limit = np.loadtxt(curves_dir + 'XENON1T_S2Only_Solar_Reflected_ER_Massive.txt').transpose()
#ax.plot(m_limit, x_limit, '--', lw=1.5, label='XENON1T S2only (Solar Refl.)')

# Simulation
m_limit, x_limit = np.loadtxt('gaas/results_gaas_oi_scan_electron_massive_100days_2fold_lce10/HeRALD_FC_100d_2device_2fold_100mus.txt').transpose()
ax.plot(m_limit*1e3, x_limit, 'r-', lw=4, label=f'100 days, 1 light signal, 10% LCE per channel')
m_limit, x_limit = np.loadtxt('gaas/results_gaas_oi_scan_electron_massive_100days_3fold_lce10/HeRALD_FC_100d_3device_3fold_100mus.txt').transpose()
ax.plot(m_limit*1e3, x_limit, 'b-', lw=4, label=f'100 days, 2 light signal, 10% LCE per channel')
m_limit, x_limit = np.loadtxt('gaas/results_gaas_oi_scan_electron_massive_300days_3fold_lce10/HeRALD_FC_300d_3device_3fold_100mus.txt').transpose()
ax.plot(m_limit*1e3, x_limit, 'm-', lw=4, label=f'300 days, 2 light signal, 10% LCE per channel')

m_limit, x_limit = np.loadtxt('results_gaas_oi_electron_massive_100days_3fold_lce10/HeRALD_FC_100d_3device_3fold_100mus.txt').transpose()
ax.plot(m_limit*1e3, x_limit, lw=6, ls='-', color='k', label=f'New')


ax.set_xscale('log')
ax.set_xlim([xmin, xmax])
ax.set_yscale('log')
ax.set_ylim([ymin, ymax])
ax.set_xlabel('DM Mass [MeV]')
ax.set_ylabel('DM-electron cross-section massive [cm2]')
ax.legend(loc='best', fontsize=12)

fig.tight_layout()
fig.savefig('gaas_ER_massive.png')
