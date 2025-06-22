from IPython.utils import io
import numpy as np
import sys
sys.path.insert(0, '/home/vvelan/Test/DarkELF/')
from darkelf import darkelf
from darklim import constants
import time

__all__ = [
    "get_dRdE_lambda_Al2O3_electron",
    "get_dRdE_lambda_GaAs_electron",
    "get_dRdE_lambda_Si_electron",
    "get_dRdE_lambda_Al2O3_phonon",
    "get_dRdE_lambda_GaAs_phonon",
    "get_dRdE_lambda_Si_phonon",
    "get_dRdE_lambda_Al2O3_absorption",
    "convert_sigmae_to_sigman",
    "convert_sigman_to_sigmae",
]

def get_dRdE_lambda_Al2O3_electron(mX_eV=1e8, mediator='massless', sigmae=1e-31, kcut=0, method='grid', withscreening=True, suppress_darkelf_output=False, gain=1.):
    """
    Function to get an anonymous lambda function, which calculates dRdE
    for DM-electron scattering in Al2O3 given only deposited energy.

    Parameters
    ----------
    mX_eV : float
        Dark matter mass in eV
    mediator : str
        Dark photon mediator mass. Must be "massive" (infinity) or
        "massless" (zero).
    sigmae : float
        DM-electron scattering cross section in cm^2
    kcut : float
        Maximum k value in the integration, in eV. If kcut=0 (default), the
        integration is cut off at the highest k-value of the grid at hand.
    method : str
        Must be "grid" or "Lindhard". Choice to use interpolated grid of
        epsilon, or Lindhard analytic epsilon
    withscreening : bool
        Whether to include the 1/|epsilon|^2 factor in the scattering rate
    suppress_darkelf_output : bool
        Whether to suppress the (useful but long) output that DarkELF gives
        when loading a material's properties.

    Returns
    -------
    fun : lambda function
        A function to calculate dRdE in DRU given E 

    """

    # Set up DarkELF GaAs object
    if suppress_darkelf_output:
        print('WARNING: You are suppressing DarkELF output')
        with io.capture_output() as captured:
            sapphire = darkelf(target='Al2O3', filename="Al2O3_mermin.dat")
    else:
        sapphire = darkelf(target='Al2O3', filename="Al2O3_mermin.dat")

    # Create anonymous function to get rate with only deposited energy
    # Note DarkELF expects recoil energies and WIMP masses in eV, and returns rates in counts/kg/yr/eV
    # But DarkLim expects recoil energies in keV, WIMP masses in GeV, and rates in counts/kg/day/keV (DRU)
    sapphire.update_params(mX=mX_eV, mediator=mediator)
    fun = lambda keV : np.heaviside(keV * 1000 / gain - constants.bandgap_Al2O3_eV, 1) * \
            sapphire.dRdomega_electron(keV * 1000 / gain, method=method, sigmae=sigmae, kcut=kcut, withscreening=withscreening) * \
            (1000 / 365.25) / gain

    return fun




def get_dRdE_lambda_Al2O3_phonon(mX_eV=1e8, mediator='massless', sigma=1e-31, dark_photon=False, suppress_darkelf_output=False):
    """
    Function to get an anonymous lambda function, which calculates dRdE
    for DM-nuclear scattering via phonons in Al2O3 given only deposited energy.

    Parameters
    ----------
    mX_eV : float
        Dark matter mass in eV
    mediator : str
        Dark photon mediator mass. Must be "massive" (infinity) or
        "massless" (zero).
    sigma : float
        If dark photon: DM-electron scattering cross section in cm^2
        If scalar: DM-nucleon scattering cross section in cm^2
    dark_photon : bool
        Whether to treat this as a dark photon
    suppress_darkelf_output : bool
        Whether to suppress the (useful but long) output that DarkELF gives
        when loading a material's properties.

    Returns
    -------
    fun : lambda function
        A function to calculate dRdE in DRU given E 

    """

    # Set up DarkELF GaAs object
    if suppress_darkelf_output:
        print('WARNING: You are suppressing DarkELF output')
        with io.capture_output() as captured:
            sapphire = darkelf(target='Al2O3', filename="Al2O3_mermin.dat", phonon_filename='Al2O3_epsphonon_o.dat')
    else:
        sapphire = darkelf(target='Al2O3', filename="Al2O3_mermin.dat", phonon_filename='Al2O3_epsphonon_o.dat')

    # Create anonymous function to get rate with only deposited energy
    # Note DarkELF expects recoil energies and WIMP masses in eV, and returns rates in counts/kg/yr/eV
    # But DarkLim expects recoil energies in keV, WIMP masses in GeV, and rates in counts/kg/day/keV (DRU)
    
    # Scalar nucleon interaction with massive mediator (the "standard" NR interaction)
    if mediator == 'massive' and not dark_photon:
        sapphire.update_params(mX=mX_eV, mediator=mediator)
        fun = lambda keV : (sapphire._dR_domega_multiphonons_SI(keV * 1000, sigma, dark_photon) + 
                            sapphire._dR_domega_coherent_single(keV * 1000, sigma, dark_photon)) * (1000 / 365.25)
        
    # Scalar nucleon interaction with massless mediator
    elif mediator == 'massless' and not dark_photon:
        sapphire.update_params(mX=mX_eV, mediator=mediator)
        fun = lambda keV : (sapphire._dR_domega_multiphonons_SI(keV * 1000, sigma, dark_photon) + 
                            sapphire._dR_domega_coherent_single(keV * 1000, sigma, dark_photon)) * (1000 / 365.25)
        
    # Dark photon interaction with massive mediator
    elif mediator == 'massive' and dark_photon:
        sapphire.update_params(mX=mX_eV, mediator=mediator)
        sigmae = sigma
        sigman = convert_sigmae_to_sigman(sigmae, mX_eV, mediator)

        # Only use the multiphonon part above the single-phonon maximum energy
        E_cutoff_single_phonon_eV = 0.199
        fun = lambda keV : sapphire._dR_domega_multiphonons_SI(keV * 1000, sigman, dark_photon) * (1000 / 365.25) * \
            np.heaviside(keV * 1000 - E_cutoff_single_phonon_eV, 1)
    
    # Dark photon interaction with massless mediator
    elif mediator == 'massless' and dark_photon:
        sapphire.update_params(mX=mX_eV, mediator=mediator)
        sigmae = sigma
        sigman = convert_sigmae_to_sigman(sigmae, mX_eV, mediator)

        # Add both single-phonon and multiphonon, but only use the
        # multiphonon part above the single-phonon maximum energy
        E_cutoff_single_phonon_eV = 0.199
        fun = lambda keV : (1000 / 365.25) * \
            (sapphire._dR_domega_multiphonons_SI(keV * 1000, sigman, dark_photon) * np.heaviside(keV * 1000 - E_cutoff_single_phonon_eV, 1) + 
             sapphire.dRdomega_phonon(keV * 1000, sigmae))
    
    return fun

    
def get_dRdE_lambda_GaAs_electron(mX_eV=1e8, mediator='massless', sigmae=1e-31, kcut=0, method='grid', withscreening=True, suppress_darkelf_output=False, gain=1.):
    """
    Function to get an anonymous lambda function, which calculates dRdE
    for DM-electron scattering in GaAs given only deposited energy.

    Parameters
    ----------
    mX_eV : float
        Dark matter mass in eV
    mediator : str
        Dark photon mediator mass. Must be "massive" (infinity) or
        "massless" (zero).
    sigmae : float
        DM-electron scattering cross section in cm^2
    kcut : float
        Maximum k value in the integration, in eV. If kcut=0 (default), the
        integration is cut off at the highest k-value of the grid at hand.
    method : str
        Must be "grid" or "Lindhard". Choice to use interpolated grid of
        epsilon, or Lindhard analytic epsilon
    withscreening : bool
        Whether to include the 1/|epsilon|^2 factor in the scattering rate
    suppress_darkelf_output : bool
        Whether to suppress the (useful but long) output that DarkELF gives
        when loading a material's properties.

    Returns
    -------
    fun : lambda function
        A function to calculate dRdE in DRU given E 

    """

    # Set up DarkELF GaAs object
    if suppress_darkelf_output:
        print('WARNING: You are suppressing DarkELF output')
        with io.capture_output() as captured:
            gaas = darkelf(target='GaAs', filename="GaAs_mermin.dat")
    else:
        gaas = darkelf(target='GaAs', filename="GaAs_mermin.dat")

    # Create anonymous function to get rate with only deposited energy
    # Note DarkELF expects recoil energies and WIMP masses in eV, and returns rates in counts/kg/yr/eV
    # But DarkLim expects recoil energies in keV, WIMP masses in GeV, and rates in counts/kg/day/keV (DRU)
    gaas.update_params(mX=mX_eV, mediator=mediator)
    fun = lambda keV : np.heaviside(keV * 1000 / gain - constants.bandgap_GaAs_eV, 1) * \
            gaas.dRdomega_electron(keV * 1000 / gain, method=method, sigmae=sigmae, kcut=kcut, withscreening=withscreening) * \
            (1000 / 365.25) / gain

    return fun




def get_dRdE_lambda_GaAs_phonon(mX_eV=1e8, mediator='massless', sigman=1e-31, dark_photon=False, suppress_darkelf_output=False, gain=1.):
    """
    Function to get an anonymous lambda function, which calculates dRdE
    for DM-nuclear scattering via GaAs in Al2O3 given only deposited energy.

    Parameters
    ----------
    mX_eV : float
        Dark matter mass in eV
    mediator : str
        Dark photon mediator mass. Must be "massive" (infinity) or
        "massless" (zero).
    sigman : float
        DM-nucleon scattering cross section in cm^2
    dark_photon : bool
        Whether to treat this as a dark photon
    suppress_darkelf_output : bool
        Whether to suppress the (useful but long) output that DarkELF gives
        when loading a material's properties.

    Returns
    -------
    fun : lambda function
        A function to calculate dRdE in DRU given E 

    """

    # Set up DarkELF GaAs object
    if suppress_darkelf_output:
        print('WARNING: You are suppressing DarkELF output')
        with io.capture_output() as captured:
            gaas = darkelf(target='GaAs', filename="GaAs_mermin.dat", phonon_filename='GaAs_epsphonon_data10K.dat')
    else:
        gaas = darkelf(target='GaAs', filename="GaAs_mermin.dat", phonon_filename='GaAs_epsphonon_data10K.dat')

    # Create anonymous function to get rate with only deposited energy
    # Note DarkELF expects recoil energies and WIMP masses in eV, and returns rates in counts/kg/yr/eV
    # But DarkLim expects recoil energies in keV, WIMP masses in GeV, and rates in counts/kg/day/keV (DRU)
    gaas.update_params(mX=mX_eV, mediator=mediator)
    fun = lambda keV : gaas._dR_domega_multiphonons_no_single(keV * 1000 / gain, sigman=sigman, dark_photon=dark_photon) * \
            (1000 / 365.25) / gain

    return fun


def get_dRdE_lambda_Si_electron(mX_eV=1e8, mediator='massless', sigmae=1e-31, kcut=0, method='grid', withscreening=True, suppress_darkelf_output=False, gain=1.):
    """
    Function to get an anonymous lambda function, which calculates dRdE
    for DM-electron scattering in Si given only deposited energy.

    Parameters
    ----------
    mX_eV : float
        Dark matter mass in eV
    mediator : str
        Dark photon mediator mass. Must be "massive" (infinity) or
        "massless" (zero).
    sigmae : float
        DM-electron scattering cross section in cm^2
    kcut : float
        Maximum k value in the integration, in eV. If kcut=0 (default), the
        integration is cut off at the highest k-value of the grid at hand.
    method : str
        Must be "grid" or "Lindhard". Choice to use interpolated grid of
        epsilon, or Lindhard analytic epsilon
    withscreening : bool
        Whether to include the 1/|epsilon|^2 factor in the scattering rate
    suppress_darkelf_output : bool
        Whether to suppress the (useful but long) output that DarkELF gives
        when loading a material's properties.

    Returns
    -------
    fun : lambda function
        A function to calculate dRdE in DRU given E 

    """

    # Set up DarkELF GaAs object
    if suppress_darkelf_output:
        print('WARNING: You are suppressing DarkELF output')
        with io.capture_output() as captured:
            Si = darkelf(target='Si', filename="Si_mermin.dat")
    else:
        Si = darkelf(target='Si', filename="Si_mermin.dat")

    # Create anonymous function to get rate with only deposited energy
    # Note DarkELF expects recoil energies and WIMP masses in eV, and returns rates in counts/kg/yr/eV
    # But DarkLim expects recoil energies in keV, WIMP masses in GeV, and rates in counts/kg/day/keV (DRU)
    Si.update_params(mX=mX_eV, mediator=mediator)
    fun = lambda keV : np.heaviside(keV * 1000 / gain - constants.bandgap_Si_eV, 1) * \
            Si.dRdomega_electron(keV * 1000 / gain, method=method, sigmae=sigmae, kcut=kcut, withscreening=withscreening) * \
            (1000 / 365.25) / gain

    return fun



def get_dRdE_lambda_Si_phonon(mX_eV=1e8, mediator='massless', sigman=1e-31, dark_photon=False, suppress_darkelf_output=False, gain=1.):
    """
    Function to get an anonymous lambda function, which calculates dRdE
    for DM-nuclear scattering via Si given only deposited energy.

    Parameters
    ----------
    mX_eV : float
        Dark matter mass in eV
    mediator : str
        Dark photon mediator mass. Must be "massive" (infinity) or
        "massless" (zero).
    sigman : float
        DM-nucleon scattering cross section in cm^2
    dark_photon : bool
        Whether to treat this as a dark photon
    suppress_darkelf_output : bool
        Whether to suppress the (useful but long) output that DarkELF gives
        when loading a material's properties.

    Returns
    -------
    fun : lambda function
        A function to calculate dRdE in DRU given E 

    """

    # Set up DarkELF Si object
    if suppress_darkelf_output:
        print('WARNING: You are suppressing DarkELF output')
        with io.capture_output() as captured:
            silicon = darkelf(target='Si', filename="Si_mermin.dat", phonon_filename='Si_epsphonon_data6K.dat')
    else:
        silicon = darkelf(target='Si', filename="Si_mermin.dat", phonon_filename='Si_epsphonon_data6K.dat')

    # Create anonymous function to get rate with only deposited energy
    # Note DarkELF expects recoil energies and WIMP masses in eV, and returns rates in counts/kg/yr/eV
    # But DarkLim expects recoil energies in keV, WIMP masses in GeV, and rates in counts/kg/day/keV (DRU)
    silicon.update_params(mX=mX_eV, mediator=mediator)
    fun = lambda keV : silicon._dR_domega_multiphonons_no_single(keV * 1000 / gain, sigman=sigman, dark_photon=dark_photon) * \
            (1000 / 365.25) / gain

    return fun



def get_dRdE_lambda_Al2O3_absorption(mX_eV=1., kappa=1e-15, res_eV=0.1, suppress_darkelf_output=False):
    """
    Function to get an anonymous lambda function, which calculates dRdE
    for DM absorption in Al2O3 given only deposited energy.

    Parameters
    ----------
    mX_eV : float
        Dark matter mass in eV
    kappa : float
        DM absorption strength (unitless)
    res_eV : float
        Energy resolution in eV (default 0.1 eV)
    suppress_darkelf_output : bool
        Whether to suppress the (useful but long) output that DarkELF gives
        when loading a material's properties.

    Returns
    -------
    fun : lambda function
        A function to calculate dRdE in DRU given E in keV

    """

    # Set up DarkELF Al2O3 object for absorption
    if suppress_darkelf_output:
        print('WARNING: You are suppressing DarkELF output')
        with io.capture_output() as captured:
            sapphire = darkelf(target='Al2O3', filename="Al2O3_mermin.dat")
    else:
        sapphire = darkelf(target='Al2O3', filename="Al2O3_mermin.dat")

    sapphire.update_params(mX=mX_eV)
    R_kgday = sapphire.R_absorption(kappa) / 365.25
    
    res_keV = res_eV / 1000.
    E0_keV = mX_eV / 1000.
    fun = lambda keV: R_kgday / np.sqrt(2 * np.pi * res_keV**2) * np.exp(-(keV - E0_keV)**2 / (2 * res_keV**2))
        
    return fun




def get_dRdE_lambda_GaAs_absorption(mX_eV=1., kappa=1e-15, res_eV=0.1, suppress_darkelf_output=False):
    """
    Function to get an anonymous lambda function, which calculates dRdE
    for DM absorption in GaAs given only deposited energy.

    Parameters
    ----------
    mX_eV : float
        Dark matter mass in eV
    kappa : float
        DM absorption strength (unitless)
    res_eV : float
        Energy resolution in eV (default 0.1 eV)
    suppress_darkelf_output : bool
        Whether to suppress the (useful but long) output that DarkELF gives
        when loading a material's properties.

    Returns
    -------
    fun : lambda function
        A function to calculate dRdE in DRU given E in keV

    """

    # Set up DarkELF Al2O3 object for absorption
    if suppress_darkelf_output:
        print('WARNING: You are suppressing DarkELF output')
        with io.capture_output() as captured:
            gaas = darkelf(target='GaAs',filename="GaAs_mermin.dat",phonon_filename="GaAs_epsphonon_data10K.dat")

    else:
        gaas = darkelf(target='GaAs',filename="GaAs_mermin.dat",phonon_filename="GaAs_epsphonon_data10K.dat")

    gaas.update_params(mX=mX_eV)
    R_kgday = gaas.R_absorption(kappa) / 365.25
    
    res_keV = res_eV / 1000.
    E0_keV = mX_eV / 1000.
    fun = lambda keV: R_kgday / np.sqrt(2 * np.pi * res_keV**2) * np.exp(-(keV - E0_keV)**2 / (2 * res_keV**2))
        
    return fun



def convert_sigmae_to_sigman(sigmae, mX_eV, mediator):
    """
    Convert DM-electron cross section to DM-nucleon cross section.
    
    Parameters
    ----------
    sigmae : float
        DM-electron scattering cross section in cm^2
    mX_eV : float
        Dark matter mass in eV
    mediator : str
        Dark photon mediator mass. Must be "massive" (infinity) or
        "massless" (zero).
        
    Returns
    -------
    sigman : float
        DM-nucleon scattering cross section in cm^2
    """

    mu_chi_p_eV = (mX_eV * constants.m_proton_GeV * 1e9) / (mX_eV + constants.m_proton_GeV * 1e9)
    mu_chi_e_eV = (mX_eV * constants.m_electron_GeV * 1e9) / (mX_eV + constants.m_electron_GeV * 1e9)
    
    if mediator == 'massive':
        sigman = sigmae * (mu_chi_p_eV / mu_chi_e_eV)**2
    elif mediator == 'massless':
        q0_e_eV = constants.alpha_fine_structure * constants.m_electron_GeV * 1e9
        q0_p_eV = constants.v0_sun / constants.speed_of_light * mu_chi_p_eV
        sigman = sigmae * (mu_chi_p_eV / mu_chi_e_eV)**2 * (q0_e_eV / q0_p_eV)**4
        
    return sigman


def convert_sigman_to_sigmae(sigman, mX_eV, mediator):
    """
    Convert DM-nucleon cross section to DM-electron cross section.
    
    Parameters
    ----------
    sigman : float
        DM-nucleon scattering cross section in cm^2
    mX_eV : float
        Dark matter mass in eV
    mediator : str
        Dark photon mediator mass. Must be "massive" (infinity) or
        "massless" (zero).
        
    Returns
    -------
    sigmae : float
        DM-electron scattering cross section in cm^2
        
    """
    
    mu_chi_p_eV = (mX_eV * constants.m_proton_GeV * 1e9) / (mX_eV + constants.m_proton_GeV * 1e9)
    mu_chi_e_eV = (mX_eV * constants.m_electron_GeV * 1e9) / (mX_eV + constants.m_electron_GeV * 1e9)

    if mediator == 'massive':
        sigmae = sigman * (mu_chi_e_eV / mu_chi_p_eV)**2
    elif mediator == 'massless':
        q0_e_eV = constants.alpha_fine_structure * constants.m_electron_GeV * 1e9
        q0_p_eV = constants.v0_sun / constants.speed_of_light * mu_chi_p_eV
        sigmae = sigman * (mu_chi_e_eV / mu_chi_p_eV)**2 * (q0_p_eV / q0_e_eV)**4

    return sigmae
