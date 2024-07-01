import numpy as np
import math
from darklim import constants
import time
from scipy import integrate, interpolate
from darklim import elf
import darklim.sensitivity._sens_est as sens_est
import matplotlib.pyplot as plt
import matplotlib as mpl

def get_deposited_energy_gaas(E_recoil_eV, pce, lce_per_channel, res, n_coincidence_light, threshold_eV, n_samples=1):

    E_light_eV = constants.GaAs_light_fraction * E_recoil_eV

    n_photons_generated_average = np.floor(E_light_eV / constants.bandgap_GaAs_eV)
    if n_photons_generated_average == 0:
        if n_samples == 1:
            return 0.
        else:
            return np.full(n_samples, 0.)

    n_photons_detected_ch1 = np.random.binomial(n_photons_generated_average, lce_per_channel, n_samples)
    n_photons_detected_ch2 = np.random.binomial(n_photons_generated_average, lce_per_channel, n_samples)

    E_ch1_eV = n_photons_detected_ch1 * constants.bandgap_GaAs_eV * np.random.normal(1, res, n_samples)
    E_ch2_eV = n_photons_detected_ch2 * constants.bandgap_GaAs_eV * np.random.normal(1, res, n_samples)

    E_heat_eV = (1 - constants.GaAs_light_fraction) * E_recoil_eV
    E_ch0_eV = np.full(n_samples, E_heat_eV * pce)

    if n_coincidence_light == 1:
        E_det_eV = (E_ch0_eV + E_ch1_eV + E_ch2_eV) * (E_ch0_eV > threshold_eV) * ((E_ch1_eV > threshold_eV) + (E_ch2_eV > threshold_eV))
    elif n_coincidence_light == 2:
        E_det_eV = (E_ch0_eV + E_ch1_eV + E_ch2_eV) * (E_ch0_eV > threshold_eV) * (E_ch1_eV > threshold_eV) * (E_ch2_eV > threshold_eV)

    if n_samples == 1:
        return E_det_eV[0]
    else:
        return E_det_eV



def convert_dRdE_dep_to_obs(E_dep_keV, dRdE_dep_DRU, pce=0.40, lce_per_channel=0.10, res=0.10, n_coincidence_light=1, calorimeter_threshold_eV=0.37, E_min_keV=None, E_max_keV=None, n_samples=int(1e6)):

    # Reduce data to the appropriate energy range
    if E_min_keV is None or E_min_keV < E_dep_keV[0]:
        E_min_keV = E_dep_keV[0]
    if E_max_keV is None or E_max_keV > E_dep_keV[-1]:
        E_max_keV = E_dep_keV[-1]

    E_pdf = E_dep_keV[(E_dep_keV >= E_min_keV) * (E_dep_keV <= E_max_keV)]
    dRdE_pdf = dRdE_dep_DRU[(E_dep_keV >= E_min_keV) * (E_dep_keV <= E_max_keV)]

    # Draw samples from the distribution
    cdf = integrate.cumtrapz(dRdE_pdf, x=E_pdf, initial=0.0)
    cdf /= cdf[-1]

    inv_cdf = interpolate.interp1d(cdf, E_pdf)

    samples = np.random.rand(n_samples)

    energies_sim_keV = inv_cdf(samples)
    energies_obs_keV = np.zeros_like(energies_sim_keV)
    energies_obs_keV = np.copy(energies_sim_keV)
    for i, E in enumerate(energies_sim_keV):
        energies_obs_keV[i] = get_deposited_energy_gaas(E * 1000, pce, lce_per_channel, res, n_coincidence_light, calorimeter_threshold_eV) / 1000

    # Convert to E vs dRdE that we can later interpolate from
    # Normalize to the number of events that are detected
    bins = np.geomspace(min(energies_obs_keV[energies_obs_keV > 0]) * 0.95, max(energies_obs_keV) * 1.05, 10000)
    counts, bin_edges = np.histogram(energies_obs_keV, bins)
    counts = counts * 1.0 / np.diff(bin_edges)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    integral_original = sum(0.5 * (dRdE_pdf[1:] + dRdE_pdf[:-1]) * np.diff(E_pdf))
    fraction_surviving = sum(energies_obs_keV > 0) / len(energies_obs_keV)
    integral_desired = integral_original * fraction_surviving
    integral_observed = sum(counts * np.diff(bin_edges))

    E_obs_keV = np.copy(bin_centers)
    dRdE_obs_DRU = counts * integral_desired / integral_observed

    E_obs_keV = E_obs_keV[dRdE_obs_DRU > 0]
    dRdE_obs_DRU = dRdE_obs_DRU[dRdE_obs_DRU > 0]

    # Return arrays and the list of energies
    return E_obs_keV, dRdE_obs_DRU, energies_obs_keV


