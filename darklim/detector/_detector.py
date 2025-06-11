import numpy as np
import math
from darklim import constants
import time
from scipy import integrate, interpolate
from darklim import elf
import darklim.sensitivity._sens_est as sens_est
from darklim.limit import drde
import matplotlib.pyplot as plt
import matplotlib as mpl
import qetpy
import detprocess
from multiprocessing import Pool
import time
import pickle
from time import localtime, strftime



def get_deposited_energy_gaas(E_recoil_eV, pce, lce_per_channel, res, n_coincidence_light, threshold_eV, coincidence_window_us, phonon_tau_us, n_samples=1):

    # Get the light signal in each channel
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

    # Get the heat signal observed within coincidence window
    E_heat_eV = (1 - constants.GaAs_light_fraction) * E_recoil_eV
    
    n_phonons_generated_average = int(E_heat_eV / constants.GaAs_average_phonon_energy_eV)
    phonon_arrival_times_us = np.random.exponential(phonon_tau_us, (n_phonons_generated_average, n_samples))
    n_phonons_detected = np.sum(phonon_arrival_times_us < coincidence_window_us, axis=0)

    E_ch0_eV = n_phonons_detected * pce * constants.GaAs_average_phonon_energy_eV * np.random.normal(1, res, n_samples)

    if n_coincidence_light == 1:
        E_det_eV = (E_ch0_eV + E_ch1_eV + E_ch2_eV) * (E_ch0_eV > threshold_eV) * ((E_ch1_eV > threshold_eV) + (E_ch2_eV > threshold_eV))
    elif n_coincidence_light == 2:
        E_det_eV = (E_ch0_eV + E_ch1_eV + E_ch2_eV) * (E_ch0_eV > threshold_eV) * (E_ch1_eV > threshold_eV) * (E_ch2_eV > threshold_eV)

    if n_samples == 1:
        return E_det_eV[0]
    else:
        return E_det_eV



def convert_dRdE_dep_to_obs_gaas(E_dep_keV, dRdE_dep_DRU, pce=0.40, lce_per_channel=0.10, res=0.10, n_coincidence_light=1,
    calorimeter_threshold_eV=0.37, coincidence_window_us=100., phonon_tau_us=100., E_min_keV=None, E_max_keV=None, n_samples=int(1e6)):

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
        energies_obs_keV[i] = get_deposited_energy_gaas(E * 1000, pce, lce_per_channel, res, n_coincidence_light, calorimeter_threshold_eV,
            coincidence_window_us, phonon_tau_us) / 1000

    # Perhaps no energy is ever observed
    if sum(energies_obs_keV > 0) == 0:
        return E_pdf, np.zeros_like(dRdE_pdf), np.array([])

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




def generalized_binomial_pmf(k, N, p):
    """Generalized binomial PMF using the Gamma function."""
    coeff = gamma(N + 1) / (gamma(k + 1) * gamma(N - k + 1))
    return coeff * (p ** k) * ((1 - p) ** (N - k))

def generalized_binomial_sample(N, p, size=1):
    """
    Generate random samples from a generalized binomial distribution
    with real-valued N by interpreting it as the sum of Bernoulli(p)
    trials from N independent sources, where a fraction of one trial
    is included probabilistically.
    """
    samples = []
    N_int = int(np.floor(N))
    N_frac = N - N_int
    
    for _ in range(size):
        # Integer part: standard binomial sample
        count = np.random.binomial(N_int, p)
        
        # Fractional part: simulate one extra trial with probability N_frac
        if np.random.random() < N_frac:
            if np.random.random() < p:
                count += 1
                
        samples.append(count)
        
    if size == 1:
        return samples[0]
    else:
        return np.array(samples)


def Lindhard(E_keV):

    Y = np.zeros(2)
    for i, (element, Z, A) in enumerate(zip(['Ga', 'As'], [31., 33.], [69.7, 74.9])):

        eta = 11.5 * E_keV * Z**(-7/3)
        k = 0.133 * Z**(2/3) * A**(-1/2)
        G = 3 * eta**(0.15) + 0.7 * eta**(0.6) + eta
        Y[i] = (k * G) / (1 + k * G)

    return np.mean(Y)



def calculate_electron_hole_pairs(E_recoil,
                                  GaAs_band_gap=1.5e-3,
                                  GaAs_Fano=0.12,
                                  GaAs_gamma_energy=0.9e-3,
                                  pt='ER',
                                  rng=np.random.default_rng(98765),
                                  verbose=False):
    """
    Calculate the number of electron-hole pairs generated based on the recoil energy.

    Parameters:
        E_recoil (float): The recoil energy in keV.

    Returns:
        int: The number of electron-hole pairs generated.
    """
    # Constants
    if pt == 'ER':
        GaAs_Fano = 0.12

    if E_recoil < GaAs_band_gap:
        # If energy is less than the band gap, no electron-hole pairs are generated
        return 0
    elif GaAs_band_gap <= E_recoil <= 3 * GaAs_band_gap and pt == 'ER':
        # If energy is between the band gap and 3 times the band gap, return 1
        return 1
    else:
        # If energy is greater than 3 times the band gap, calculate the expected number of pairs
        if pt == 'NR':
            E_recoil_quenched = E_recoil * Lindhard(E_recoil)
            N = E_recoil_quenched / (3 * GaAs_band_gap)
            if verbose:
                print(f'Expect mean number of pairs: {N:.03f}, variance: {N:.03f}.')
            return rng.poisson(N)
        elif pt == 'ER':
            N = E_recoil / (3 * GaAs_band_gap) / (1 - GaAs_Fano)
            p = 1 - GaAs_Fano
            if verbose:
                print(f'Expect mean number of pairs: {N*p:.03f}, variance: {N*p*(1-p):.03f}. This corresponds to N = {N:.03f}, p = {p:.03f}.')
            return generalized_binomial_sample(N, p)



def DM_spectrum_GaAs(m_DM_eV, sigma, mediator='NR', e_high=100.,
                     N_PDs = 1, E_th_PD = 0.9e-3, E_res_PD = 0.18e-3, E_th_GaAs = 0.9e-3, E_res_GaAs = 0.18e-3,
                     collection_efficiency=0.34, GaAs_QE=0.6, GaAs_gamma_energy=0.9e-3,
                     rng=np.random.default_rng(12345), n_sim = 1_000_000, verbose=False, plot=False,
                     ): 
    """
    Generate the DM spectrum for GaAs based on the given mass and cross-section.

    Parameters:
        m_DM_eV (float): Mass of the dark matter particle in eV.
        sigma (float): Cross-section for nuclear or electron scattering in cm^2.

    Returns:
        tuple: Energy array in keV and corresponding differential rate.
    """

    if mediator == 'NR':
        E_DM_keV = np.geomspace(0.1e-3, e_high, 600)
        dRdE_DM_DRU = drde(E_DM_keV, m_DM_eV/1e9, sigma, 'GaAs')

    else:
        E_DM_keV = np.append(np.linspace(0.1e-3, 3e-3, 100), np.linspace(3e-3, 100e-3, 400))
        fun = elf.get_dRdE_lambda_GaAs_electron(mX_eV=m_DM_eV, mediator=mediator, sigmae=sigma)
        dRdE_DM_DRU = [fun(E) for E in E_DM_keV]

    R_integrated = np.trapz(dRdE_DM_DRU, E_DM_keV)
    if R_integrated == 0:
        return E_DM_keV, np.zeros_like(dRdE_DM_DRU)

    pdf_DM = np.array(dRdE_DM_DRU) / R_integrated  # Normalize the differential rate to get a PDF
    cdf_DM = np.cumsum(pdf_DM * np.append(np.array([0.]), np.diff(E_DM_keV)))
    cdf_DM /= cdf_DM[-1]  # Normalize to end at 1

    ########################

    E_recoil = np.zeros(n_sim, dtype=float)
    for i in range(len(E_recoil)):
        p = rng.random()
        E_recoil[i] = np.interp(p, cdf_DM, E_DM_keV)

    N_eh = np.zeros_like(E_recoil)
    N_photons = np.zeros_like(N_eh)
    N_photons_PD_A = np.zeros_like(N_eh)
    N_photons_PD_B = np.zeros_like(N_eh)
    N_photons_sink = np.zeros_like(N_eh)
    E_PD_A_keV = np.zeros_like(N_eh)
    E_PD_B_keV = np.zeros_like(N_eh)
    E_PD_keV = np.zeros_like(N_eh)
    E_GaAs_keV = np.zeros_like(N_eh)
    E_GaAs_obs_keV = np.zeros_like(N_eh)
    fraction_in_GaAs = np.zeros_like(N_eh)
    detected = np.zeros_like(N_eh, dtype=bool)

    for i in range(len(N_eh)):
        if mediator == 'NR':
            N_eh[i] = calculate_electron_hole_pairs(E_recoil[i], pt='NR', rng=rng)
        else:
            N_eh[i] = calculate_electron_hole_pairs(E_recoil[i], rng=rng)

        N_photons[i] = rng.binomial(N_eh[i], GaAs_QE)
        if N_PDs == 1:
            N_photons_PD_A[i] = rng.binomial(N_photons[i], collection_efficiency)
            N_photons_sink[i] = N_photons[i] - N_photons_PD_A[i]

            E_PD_A_keV[i] = rng.normal(N_photons_PD_A[i] * GaAs_gamma_energy, E_res_PD)
            E_PD_keV[i] = E_PD_A_keV[i]
            E_GaAs_keV[i] = E_recoil[i] - (N_photons_PD_A[i] + N_photons_sink[i]) * GaAs_gamma_energy
            E_GaAs_obs_keV[i] = rng.normal(E_GaAs_keV[i], E_res_GaAs)

            fraction_in_GaAs[i] = E_GaAs_keV[i] / E_recoil[i] if E_recoil[i] > 0 else 0
            detected[i] = (E_PD_A_keV[i] > E_th_PD) * (E_GaAs_obs_keV[i] > E_th_GaAs)
        elif N_PDs == 2:
            
            # Prevent more than N_photons from being detected
            # Also don't give preference to either PD
            if rng.random() < 0.5:
                N_photons_PD_A[i] = rng.binomial(N_photons[i], collection_efficiency/2)
                N_photons_PD_B[i] = rng.binomial(N_photons[i], collection_efficiency/2)
                if N_photons_PD_A[i] + N_photons_PD_B[i] > N_photons[i]:
                    N_photons_PD_B[i] = N_photons[i] - N_photons_PD_A[i]
            else:
                N_photons_PD_B[i] = rng.binomial(N_photons[i], collection_efficiency/2)
                N_photons_PD_A[i] = rng.binomial(N_photons[i], collection_efficiency/2)
                if N_photons_PD_A[i] + N_photons_PD_B[i] > N_photons[i]:
                    N_photons_PD_A[i] = N_photons[i] - N_photons_PD_B[i]

            N_photons_sink[i] = N_photons[i] - (N_photons_PD_A[i] + N_photons_PD_B[i])

            E_PD_A_keV[i] = rng.normal(N_photons_PD_A[i] * GaAs_gamma_energy, E_res_PD)
            E_PD_B_keV[i] = rng.normal(N_photons_PD_B[i] * GaAs_gamma_energy, E_res_PD)
            E_PD_keV[i] = (E_PD_A_keV[i] + E_PD_B_keV[i])
            
            E_GaAs_keV[i] = E_recoil[i] - (N_photons_PD_A[i] + N_photons_PD_B[i] + N_photons_sink[i]) * GaAs_gamma_energy
            E_GaAs_obs_keV[i] = rng.normal(E_GaAs_keV[i], E_res_GaAs)

            fraction_in_GaAs[i] = E_GaAs_keV[i] / E_recoil[i] if E_recoil[i] > 0 else 0
            detected[i] = (E_PD_A_keV[i] > E_th_PD) * (E_PD_B_keV[i] > E_th_PD) * (E_GaAs_obs_keV[i] > E_th_GaAs)

    fraction_detected = sum(detected) / n_sim
    if sum(detected) == 0:
        return E_DM_keV, np.zeros_like(dRdE_DM_DRU)

    # Convert simulation to dRdE spectrum for GaAs observed energy
    E_GaAs_bins = np.linspace(min(E_GaAs_obs_keV[detected]) * 0.8, max(E_GaAs_obs_keV[detected]) * 1.2, 300)
    dRdE_GaAs_obs, _ = np.histogram(E_GaAs_obs_keV[detected], bins=E_GaAs_bins, density=True)
    E_GaAs_bins = (E_GaAs_bins[:-1] + E_GaAs_bins[1:]) / 2  # Convert to bin centers
    dRdE_GaAs_obs *= fraction_detected * R_integrated

    if verbose:
        print(f'Mean recoil energy: {np.mean(E_recoil)*1e3:.3f} eV, std {np.std(E_recoil)*1e3:.3f} eV')
        print(f'Mean number of electron-hole pairs: {np.mean(N_eh):.3f}, variance {np.var(N_eh):.3f}')
        print(f'Fraction of events with at least {E_th_PD*1e3:.2f} eV in Photodetector: {fraction_detected:.3f}')

        # Plot the results in counts
        fig, ax = plt.subplots(1, 1, figsize=(9, 6))

        h = ax.hist2d(E_PD_keV*1e3, E_GaAs_obs_keV*1e3, bins=(100, 100), norm=mpl.colors.LogNorm(), cmap='viridis', cmin=1)
        fig.colorbar(h[3], ax=ax, label='Counts')

        ax.set_xlabel('Energy observed in Photodetector [eV]')
        ax.set_ylabel('Energy observed in GaAs [eV]')
        ax.set_title(f'{mediator} Scattering\n m_DM = {m_DM_eV/1e6:.0f} MeV, Collection Efficiency = {collection_efficiency:.2f}', fontsize=18)

        fig.tight_layout()
        fig.show()

        # Plot the distribution of recoil energies that trigger the photodetector
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        ax[0].hist(E_recoil, bins=100, alpha=0.7, histtype='step', color='black', label='Simulated Events [True E_R]')
        ax[0].hist(E_recoil[detected], bins=100, alpha=0.7, histtype='step', color='blue', label='Observed Events [True E_R]')
        ax[0].hist(E_GaAs_obs_keV[detected], bins=100, alpha=0.7, histtype='step', color='red', label='Observed Events [E_GaAs_obs]')
        ax[0].set_xscale('log')
        ax[0].set_yscale('log')
        xmin, xmax = ax[0].get_xlim()
        ax[0].set_xlim([max(1e-4, xmin), xmax])
        ax[0].set_xlabel('Energy [keV]')
        ax[0].set_ylabel('Counts')
        ax[0].legend(loc='best', fontsize=13)

        # Plot the distribution of energy in GaAs if there is at least 0.9 eV in the Photodetector
        ax[1].hist(E_GaAs_obs_keV[detected] * 1e3, bins=100, density=True, alpha=0.7, color='green')
        ax[1].set_xlabel('Energy in GaAs [eV]')
        ax[1].set_ylabel('Probability Density')
        fig.tight_layout()
        fig.show()

        # Plot the differential rate in GaAs observed energy
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(E_DM_keV, dRdE_DM_DRU, label=f'Recoil Rate for m_DM = {m_DM_eV/1e6:.0f} MeV', color='black')
        ax.plot(E_GaAs_bins, dRdE_GaAs_obs, label=f'Observed GaAs Energy Spectrum', color='red')
        ax.set_xlabel('Energy [keV]')
        ax.set_ylabel('Differential Rate [DRU]')
        ax.set_xscale('log')
        ax.set_yscale('log')
        xmin, xmax = ax.get_xlim()
        ax.set_xlim([max(5e-4, xmin), xmax])
        ymin, ymax = ax.get_ylim()
        ax.set_ylim([min(dRdE_GaAs_obs[dRdE_GaAs_obs > 0]) / 10, ymax])
        ax.legend(loc='best', fontsize=13)
        fig.tight_layout()
        fig.show()

    return E_GaAs_bins, dRdE_GaAs_obs




def monoenergetic_sim_GaAs(E_keV, recoil = 'ER',
                     N_PDs = 1, E_th_PD = 0.9e-3, E_res_PD = 0.18e-3, E_th_GaAs = 0.9e-3, E_res_GaAs = 0.18e-3,
                     collection_efficiency=0.34, GaAs_QE=0.6, GaAs_gamma_energy=0.9e-3,
                     rng=np.random.default_rng(12345), n_sim = 1_000_000, verbose=False,
                     ): 
    """
    Generate simulated events for a monoenergetic signal in GaAs.

    Parameters:
        E_keV (float): Energy of the signal in keV.
        N_PDs (int): Number of photodetectors.
        E_th_PD (float): Energy threshold for the photodetector in eV.
        E_res_PD (float): Energy resolution of the photodetector in eV.
        E_th_GaAs (float): Energy threshold for GaAs in eV.
        E_res_GaAs (float): Energy resolution of GaAs in eV.
        collection_efficiency (float): Collection efficiency of the detector.
        GaAs_QE (float): Quantum efficiency of GaAs.
        GaAs_gamma_energy (float): Energy of gamma photons in eV.
        rng (np.random.Generator): Random number generator.
        n_sim (int): Number of simulated events.
        verbose (bool): If True, print additional information.
    
    Returns:
        E0_arr, E1_arr, [E2_arr]: Arrays of energies observed in the GaAs (E0) and photodetectors (E1 and maybe E2).
        detected: Array of boolean values indicating if the event was detected.
    """

    ########################

    E_recoil = np.full(n_sim, E_keV, dtype=float)

    N_eh = np.zeros_like(E_recoil)
    N_photons = np.zeros_like(N_eh)
    N_photons_PD_A = np.zeros_like(N_eh)
    N_photons_PD_B = np.zeros_like(N_eh)
    N_photons_sink = np.zeros_like(N_eh)
    E_PD_A_keV = np.zeros_like(N_eh)
    E_PD_B_keV = np.zeros_like(N_eh)
    E_GaAs_keV = np.zeros_like(N_eh)
    E_GaAs_obs_keV = np.zeros_like(N_eh)
    fraction_in_GaAs = np.zeros_like(N_eh)
    detected = np.zeros_like(N_eh, dtype=bool)

    for i in range(len(N_eh)):
        if recoil == 'NR':
            N_eh[i] = calculate_electron_hole_pairs(E_recoil[i], pt='NR', rng=rng)
        else:
            N_eh[i] = calculate_electron_hole_pairs(E_recoil[i], rng=rng)

        N_photons[i] = rng.binomial(N_eh[i], GaAs_QE)
        if N_PDs == 1:
            N_photons_PD_A[i] = rng.binomial(N_photons[i], collection_efficiency)
            N_photons_sink[i] = N_photons[i] - N_photons_PD_A[i]

            E_PD_A_keV[i] = rng.normal(N_photons_PD_A[i] * GaAs_gamma_energy, E_res_PD)
            E_GaAs_keV[i] = E_recoil[i] - (N_photons_PD_A[i] + N_photons_sink[i]) * GaAs_gamma_energy
            E_GaAs_obs_keV[i] = rng.normal(E_GaAs_keV[i], E_res_GaAs)

            fraction_in_GaAs[i] = E_GaAs_keV[i] / E_recoil[i] if E_recoil[i] > 0 else 0
            detected[i] = (E_PD_A_keV[i] > E_th_PD) * (E_GaAs_obs_keV[i] > E_th_GaAs)
            
        elif N_PDs == 2:
            
            # Prevent more than N_photons from being detected
            # Also don't give preference to either PD
            if rng.random() < 0.5:
                N_photons_PD_A[i] = rng.binomial(N_photons[i], collection_efficiency/2)
                N_photons_PD_B[i] = rng.binomial(N_photons[i], collection_efficiency/2)
                if N_photons_PD_A[i] + N_photons_PD_B[i] > N_photons[i]:
                    N_photons_PD_B[i] = N_photons[i] - N_photons_PD_A[i]
            else:
                N_photons_PD_B[i] = rng.binomial(N_photons[i], collection_efficiency/2)
                N_photons_PD_A[i] = rng.binomial(N_photons[i], collection_efficiency/2)
                if N_photons_PD_A[i] + N_photons_PD_B[i] > N_photons[i]:
                    N_photons_PD_A[i] = N_photons[i] - N_photons_PD_B[i]

            N_photons_sink[i] = N_photons[i] - (N_photons_PD_A[i] + N_photons_PD_B[i])

            E_PD_A_keV[i] = rng.normal(N_photons_PD_A[i] * GaAs_gamma_energy, E_res_PD)
            E_PD_B_keV[i] = rng.normal(N_photons_PD_B[i] * GaAs_gamma_energy, E_res_PD)
            
            E_GaAs_keV[i] = E_recoil[i] - (N_photons_PD_A[i] + N_photons_PD_B[i] + N_photons_sink[i]) * GaAs_gamma_energy
            E_GaAs_obs_keV[i] = rng.normal(E_GaAs_keV[i], E_res_GaAs)

            fraction_in_GaAs[i] = E_GaAs_keV[i] / E_recoil[i] if E_recoil[i] > 0 else 0
            detected[i] = (E_PD_A_keV[i] > E_th_PD) * (E_PD_B_keV[i] > E_th_PD) * (E_GaAs_obs_keV[i] > E_th_GaAs)

    return E_GaAs_obs_keV, E_PD_A_keV, E_PD_B_keV, detected



def sim_collect_mono(sim_number, template, csd, fs, E, rate_Hz, sim_trace_length_s, n_sigma, rng, n_cores):
    
    f_frequencies = template.shape[-1]

    sim_trace_length_s = 10.
    sim_trace_length = fs * sim_trace_length_s
    n_traces_per_sim = int(sim_trace_length / f_frequencies)
    
    pretrig = int(f_frequencies/2)

    # Simulate noise in the pixel
    simulated_waveform_2D = qetpy.gen_noise(csd, fs, n_traces_per_sim, rng=rng)
    simulated_waveform_td = np.zeros(f_frequencies * n_traces_per_sim)
    for trace_i in range(n_traces_per_sim):
        simulated_waveform_td[trace_i * f_frequencies:(trace_i + 1) * f_frequencies] = simulated_waveform_2D[trace_i, 0, :]

    # Simulate arrival times
    expected_events = int(rate_Hz * sim_trace_length_s * 1.5)  # Overestimate to be safe
    arrival_times = np.cumsum(rng.exponential(1.0 / rate_Hz, expected_events)) * fs
    arrival_times = arrival_times[arrival_times < (sim_trace_length - f_frequencies)].astype(int)

    # Add pulses to pixel 0 waveform
    for i, arrival_time in enumerate(arrival_times):
        simulated_waveform_td[arrival_time:arrival_time + f_frequencies] += E * template[0,0]

    # Process the data with a 1x1 trigger
    oftrigger = detprocess.OptimumFilterTrigger('MyTrigger0', fs, template, csd, pretrigger_samples=pretrig)
    oftrigger.update_trace(simulated_waveform_td)
    dynamic_threshold_function = lambda amp: max(100, 2 * 78 * np.log(amp / 13))
    oftrigger.find_triggers(n_sigma, dynamic=True, dynamic_threshold_function=dynamic_threshold_function)

    # Extract trigger amplitudes the 1x1 trigger
    trigger_times = np.array(oftrigger.get_trigger_data()['MyTrigger0']['trigger_index'])
    trigger_amplitudes = np.array(oftrigger.get_trigger_data()['MyTrigger0']['trigger_amplitude_0'])
    
    # Check if each simulated event was detected
    detected = np.zeros(len(arrival_times), dtype=bool)
    detection_window = 50 # 50 samples = 40 us
    e_resolution = oftrigger.get_resolution()[0]
    
    for i, arrival_time in enumerate(arrival_times):
        # Check for any coincident triggers
        t_adjusted = arrival_time + pretrig
        idxs = np.where((trigger_times >= t_adjusted - detection_window) * (trigger_times < t_adjusted + detection_window))[0]
        if len(idxs) > 0:
            # Check if the trigger amplitude is the correct energy
            energies = trigger_amplitudes[idxs]
            if np.any(np.abs(energies - E) < 4 * e_resolution):
                detected[i] = True

    return trigger_times, trigger_amplitudes, arrival_times, detected, np.full(len(trigger_times), sim_number, dtype=int)



def monoenergetic_sim_OF(template, csd, fs, E, n_sims=10_000, rate_Hz = 50., n_sigma=5., seed=12345, n_cores=3):
    """
    Simulate a single energy many times using the Optimum Filter method.
    
    Parameters:
        template (ndarray): The 1x1 template to use for simulation.
        csd (ndarray): The 1x1 CSD to use for simulation.
        E (float): The energy of the signal in the same units as the template.
        n_sims (int): Number of simulations to run.
        rate_Hz (float): Rate of events in Hz.
        n_sigma (float): Number of standard deviations for the trigger.
        rng (np.random.Generator): Random number generator.
        n_cores (int): Number of cores to use for parallel processing.
        
    Returns:
        E_arr (ndarray): Array of energies observed in the OF.
    """
    
    # Set up the simulation parameters
    f_frequencies = template.shape[-1]
    data_length_s = n_sims / rate_Hz

    sim_trace_length_s = 10.
    sim_trace_length = fs * sim_trace_length_s

    n_simulations = int(np.floor(data_length_s / sim_trace_length_s))
    
    results = []
    for batch_start in range(0, n_simulations, n_cores):
        batch_indices = list(range(batch_start, min(batch_start + n_cores, n_simulations)))
        batch_args = [(sim_number, template, csd, fs, E, rate_Hz, sim_trace_length_s, n_sigma, np.random.default_rng(seed + sim_number * 1000), n_cores) for sim_number in batch_indices]
        with Pool(n_cores) as pool:
            batch_results = pool.starmap(sim_collect_mono, batch_args)
        results.extend(batch_results)
        
    trigger_times = np.array([e for result in results for e in result[0]])
    trigger_amplitudes = np.array([e for result in results for e in result[1]])
    true_arrival_times = np.array([e for result in results for e in result[2]])
    detected = np.array([e for result in results for e in result[3]])
    sim_numbers = np.array([e for result in results for e in result[4]])
    
    return trigger_times, trigger_amplitudes, true_arrival_times, detected, sim_numbers






def sim_collect_DM(sim_number, template, csd, fs, rate_Hz, mass_GeV, tm, sim_trace_length_s, n_sigma, rng, n_cores):
    
    f_frequencies = template.shape[-1]

    sim_trace_length_s = 10.
    sim_trace_length = fs * sim_trace_length_s
    n_traces_per_sim = int(sim_trace_length / f_frequencies)
    
    pretrig = int(f_frequencies/2)

    # Simulate noise in the pixel
    simulated_waveform_2D = qetpy.gen_noise(csd, fs, n_traces_per_sim, rng=rng)
    simulated_waveform_td = np.zeros(f_frequencies * n_traces_per_sim)
    for trace_i in range(n_traces_per_sim):
        simulated_waveform_td[trace_i * f_frequencies:(trace_i + 1) * f_frequencies] = simulated_waveform_2D[trace_i, 0, :]

    # Simulate arrival times
    expected_events = int(rate_Hz * sim_trace_length_s * 1.5)  # Overestimate to be safe
    arrival_times = np.cumsum(rng.exponential(1.0 / rate_Hz, expected_events)) * fs
    arrival_times = arrival_times[arrival_times < (sim_trace_length - f_frequencies)].astype(int)
    
    E_eV_arr = np.geomspace(1e-3, 100e3, 10_000)
    E_keV_arr = E_eV_arr / 1e3 
    dRdE_DRU_arr = drde(E_keV_arr, mass_GeV, 1e-40, tm)
    R_integrated = np.trapz(dRdE_DRU_arr, E_keV_arr)
    pdf_arr = dRdE_DRU_arr / R_integrated
    cdf_arr = np.cumsum(pdf_arr * np.append(np.array([0.]), np.diff(E_keV_arr)))
    simulated_energies = np.zeros(len(arrival_times), dtype=float)
    for i in range(len(arrival_times)):
        p = rng.random()
        simulated_energies[i] = np.interp(p, cdf_arr, E_keV_arr) * 1e3

    # Add pulses to pixel 0 waveform
    for i, arrival_time in enumerate(arrival_times):
        simulated_waveform_td[arrival_time:arrival_time + f_frequencies] += simulated_energies[i] * template[0,0]

    # Process the data with a 1x1 trigger
    oftrigger = detprocess.OptimumFilterTrigger('MyTrigger0', fs, template, csd, pretrigger_samples=pretrig)
    oftrigger.update_trace(simulated_waveform_td)
    dynamic_threshold_function = lambda amp: max(100, 2 * 78 * np.log(amp / 13))
    oftrigger.find_triggers(n_sigma, dynamic=True, dynamic_threshold_function=dynamic_threshold_function)

    # Extract trigger amplitudes the 1x1 trigger
    trigger_times = np.array(oftrigger.get_trigger_data()['MyTrigger0']['trigger_index'])
    trigger_amplitudes = np.array(oftrigger.get_trigger_data()['MyTrigger0']['trigger_amplitude_0'])
    
    return trigger_times, trigger_amplitudes, arrival_times, np.full(len(trigger_times), sim_number, dtype=int)



def DM_sim_OF(template, csd, fs, n_sims=10_000, rate_Hz = 50., n_sigma=5., mass_GeV=1., tm='Si', seed=12345, n_cores=3):
    """
    Simulate a single energy many times using the Optimum Filter method.
    
    Parameters:
        template (ndarray): The 1x1 template to use for simulation.
        csd (ndarray): The 1x1 CSD to use for simulation.
        E (float): The energy of the signal in the same units as the template.
        n_sims (int): Number of simulations to run.
        rate_Hz (float): Rate of events in Hz.
        n_sigma (float): Number of standard deviations for the trigger.
        rng (np.random.Generator): Random number generator.
        n_cores (int): Number of cores to use for parallel processing.
        
    Returns:
        E_arr (ndarray): Array of energies observed in the OF.
    """
    
    # Set up the simulation parameters
    f_frequencies = template.shape[-1]
    data_length_s = n_sims / rate_Hz

    sim_trace_length_s = 10.
    sim_trace_length = fs * sim_trace_length_s

    n_simulations = int(np.floor(data_length_s / sim_trace_length_s))
    
    results = []
    for batch_start in range(0, n_simulations, n_cores):
        batch_indices = list(range(batch_start, min(batch_start + n_cores, n_simulations)))
        batch_args = [(sim_number, template, csd, fs, rate_Hz, mass_GeV, tm, sim_trace_length_s, n_sigma, np.random.default_rng(seed + sim_number * 1000), n_cores) for sim_number in batch_indices]
        with Pool(n_cores) as pool:
            batch_results = pool.starmap(sim_collect_DM, batch_args)
        results.extend(batch_results)
        
    trigger_times = np.array([e for result in results for e in result[0]])
    trigger_amplitudes = np.array([e for result in results for e in result[1]])
    true_arrival_times = np.array([e for result in results for e in result[2]])
    sim_numbers = np.array([e for result in results for e in result[3]])
    
    return trigger_times, trigger_amplitudes, true_arrival_times, sim_numbers



