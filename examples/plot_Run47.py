import numpy as np
from datetime import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import math

curves_dir = 'ExistingLimits/'

# Some figure setup
mpl.rcParams.update({'font.size': 17})
mpl.rcParams.update({'axes.linewidth': 2})
fig, ax = plt.subplots(1, 1, figsize=(12,9.5))
xmin = 1e1; xmax = 1e4
ymin = 1e-35; ymax = 1e-20

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
m_limit, x_limit = np.loadtxt('Run47_limit_NR/limit.txt').transpose()
ax.plot(m_limit*1e3, x_limit, 'k-', lw=3, label='Run 47 Si')

m_limit, x_limit = np.loadtxt('Run47_limit_phonon_massive/limit.txt').transpose()
ax.plot(m_limit*1e3, x_limit, '--', color='#00c1ff', lw=3, label='Si multiphonon (massive mediator)')

#m_limit, x_limit = np.loadtxt('Run47_limit_phonon_massless/limit.txt').transpose()
#ax.plot(m_limit[m_limit < 10]*1e3, x_limit[m_limit < 10], 'b-.', lw=3, label='Si multiphonon (massless mediator)')

#m_limit, x_limit = np.loadtxt('Run47_sapphire_samemass_limit_NR/limit.txt').transpose()
#ax.plot(m_limit*1e3, x_limit, 'b--', lw=4, label='Run 47 Al2O3 (same mass as Si)')

m_limit, x_limit = np.loadtxt('Run47_sapphire_limit_NR/limit.txt').transpose()
ax.plot(m_limit*1e3, x_limit, 'm-', lw=3, label='Al2O3 NR')

m_limit, x_limit = np.loadtxt('Run47_sapphire_limit_phonon_massive//limit.txt').transpose()
ax.plot(m_limit*1e3, x_limit, '--', color='#ffaa00', lw=3, label='Al2O3 multiphonon (massive mediator)')

#m_limit, x_limit = np.loadtxt('Run47_sapphire_limit_phonon_massless/limit.txt').transpose()
#ax.plot(m_limit*1e3, x_limit, 'g-.', lw=4, label='Al2O3 multiph (massless)')

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
ax.legend(loc='upper right', fontsize=15)

#ax.text(1e2, 1e-24, 'Al2O3, 0.4 grams', fontsize=25)
#ax.text(1e2, 1e-25, '0.5 eV threshold', fontsize=25)

fig.tight_layout()
fig.savefig('Run47_Limit.png')

