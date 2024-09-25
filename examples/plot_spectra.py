import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import sys
from scipy.optimize import minimize
mpl.rcParams.update({'font.size': 20})
mpl.rcParams.update({'axes.linewidth': 2})



# Define the model: falling power law
def power_law_basic(x, A, alpha):
    return A * x**(-alpha)

def power_law(x, A, alpha, x0, k):
    return A * x**(-alpha) / (1 + np.exp(-(x - x0)*k))

# Define the model: falling exponential
def exponential_basic(x, A, alpha):
    return A * np.exp(-x * alpha)

def exponential(x, A, alpha, x0, k):
    return A * np.exp(-x * alpha) / (1 + np.exp(-(x - x0)*k))

# Negative log likelihood function to minimize
def negative_log_likelihood_power(params, x, counts):
    A, alpha, x0, k = params
    y = power_law(x, A*1e4, alpha, x0, k)
    nll = np.sum(y - counts * np.log(y))
    return nll

# Negative log likelihood function to minimize
def negative_log_likelihood_exp(params, x, counts):
    A, alpha, x0, k = params
    y = exponential(x, A*1e4, alpha, x0, k)
    nll = np.sum(y - counts * np.log(y))
    return nll



energies_eV = np.load('spectra/BigFins_shared_0719.npy')

mass_kg = (1 * 1 * 0.1) * 2.329 * 1e-3
time_days = 3 / 24.

print(f'{len(energies_eV)} events')
print(f'{mass_kg * 1000:.4f} grams of Si')

bins_eV_full = np.arange(1, 6, 0.03)
fit_low = 1.3
fit_high = 4
bins_eV_fit = bins_eV_full[(bins_eV_full > fit_low) * (bins_eV_full < fit_high)]

event_weights_DRU = np.full(len(energies_eV), (1/mass_kg/time_days/np.diff(bins_eV_fit*1e-3)[0]))
print(f'Event weight: {event_weights_DRU[0]:.3e}')

dRdE_DRU_arr, E_eV_arr = np.histogram(energies_eV, bins_eV_fit)
E_eV_arr = 0.5 * (E_eV_arr[1:] + E_eV_arr[:-1])

# Perform the optimization to minimize the negative log likelihood
result = minimize(negative_log_likelihood_power, [4, 5, 1.5, 1], args=(E_eV_arr, dRdE_DRU_arr), method='L-BFGS-B', bounds=[(1e-10, None), (1e-10, None), (1, 2), (1e-10, None)])
A_fit_power, alpha_fit_power, Eth_power, inv_width_power = result.x
A_fit_power *= event_weights_DRU[0] * 1e4
print(f"Power law parameters: A = {A_fit_power:.3e} DRU, exponent = -{alpha_fit_power:.4f}")
print(f'Threshold = {Eth_power:.3f} eV, width = {1 / inv_width_power:.5f} eV')

# Perform the optimization to minimize the negative log likelihood
result = minimize(negative_log_likelihood_exp, [4, 2., 1.5, 1.], args=(E_eV_arr, dRdE_DRU_arr), method='L-BFGS-B', bounds=[(1e-10, None), (1e-10, None), (1, 2), (1e-10, None)])
A_fit_exp, inv_E0_exp, Eth_exp, inv_width_exp = result.x
A_fit_exp *= event_weights_DRU[0] * 1e4
print(f"Exponential parameters: A = {A_fit_exp:.3e} DRU, E0 = {1 / inv_E0_exp:.4f} eV")
print(f'Threshold = {Eth_exp:.3f} eV, width = {1 / inv_width_exp:.5f} eV')

####### Plotting #######

fig, ax = plt.subplots(2, 1, figsize=(15, 12))

for i in range(2):
    ax[i].hist(energies_eV, bins_eV_full, weights=event_weights_DRU, alpha=0.4, color='b')
    ax[i].set_yscale('log')
    ymin, ymax = ax[i].get_ylim()

    if i == 0:
#        ax[i].plot(bins_eV_full, exponential(bins_eV_full, A_fit_exp, inv_E0_exp, Eth_exp, inv_width_exp), 'r-')
        ax[i].plot(bins_eV_full, exponential_basic(bins_eV_full, A_fit_exp, inv_E0_exp), 'r-')
        ax[i].plot(bins_eV_full, exponential_basic(bins_eV_full, 2592000000., 1 / 20), 'g-', lw=3)
        ax[i].text(0.98, 0.95, f'dRdE [DRU] = \n{A_fit_exp:.3e} * e^[-E / ({1/inv_E0_exp:.4f} eV)]', ha='right', va='top', fontsize=16, color='r', transform=ax[i].transAxes)
    else:
        ax[i].set_xscale('log')
        ax[i].plot(bins_eV_full, power_law(bins_eV_full, A_fit_power, alpha_fit_power, Eth_power, inv_width_power), 'r-')
        ax[i].text(0.98, 0.95, f'dRdE [DRU] = \n{A_fit_power:.3e} * [E (eV)]^-({alpha_fit_power:.4f})', ha='right', va='top', fontsize=16, color='r', transform=ax[i].transAxes)

    ax[i].set_xlabel('Energy [eV]')
    ax[i].set_ylabel('Event Rate [DRU]')

    ax[i].plot([fit_low, fit_low], [ymin, ymax], 'k--', lw=2)
    ax[i].plot([fit_high, fit_high], [ymin, ymax], 'k--', lw=2)
    ax[i].set_ylim([ymin, ymax])


fig.tight_layout()
fig.savefig('Si_background_spectra.png')
