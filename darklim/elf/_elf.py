from IPython.utils import io
import numpy as np
import sys
sys.path.append('/Users/vetri/GitRepos/DarkELF/')
from darkelf import darkelf
from darklim import constants

__all__ = [
    "get_dRdE_lambda_Al2O3_electron",
    "get_dRdE_lambda_GaAs_electron",
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
    fun = lambda keV : np.heaviside(keV * 1000 / gain - 2 * constants.bandgap_Al2O3_eV, 1) * \
            sapphire.dRdomega_electron(keV * 1000 / gain, method=method, sigmae=sigmae, kcut=kcut, withscreening=withscreening) * \
            (1000 / 365.25) / gain

    return fun



if False:

    def get_dRdE_lambda_Al2O3_electron(mX_eV=1e8, mediator='massless', sigmae=1e-31, kcut=0, method='grid', withscreening=True, suppress_darkelf_output=False, gain=1):
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

        # Set up DarkELF Al2O3 (sapphire) object
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
        fun = lambda keV : np.heaviside(keV - 2 * constants.bandgap_Al2O3_eV, 1) * \
                (1000 / 365.25) * \
                sapphire.dRdomega_electron(keV * 1000, method=method, sigmae=sigmae, kcut=kcut, withscreening=withscreening)

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
    fun = lambda keV : np.heaviside(keV * 1000 / gain - 2 * constants.bandgap_GaAs_eV, 1) * \
            gaas.dRdomega_electron(keV * 1000 / gain, method=method, sigmae=sigmae, kcut=kcut, withscreening=withscreening) * \
            (1000 / 365.25) / gain

    return fun





def get_dRdE_lambda_GaAs_phonon(mX_eV=1e8, mediator='massless', sigmae=1e-31, suppress_darkelf_output=False, gain=1.):
    """
    Function to get an anonymous lambda function, which calculates dRdE
    for DM-electron scattering in GaAs via phonons given only deposited
    energy. We return the total rate from both single-photon and
    multi-phonon interactions (by adding these together).

    Parameters
    ----------
    mX_eV : float
        Dark matter mass in eV
    mediator : str
        Dark photon mediator mass. Must be "massive" (infinity) or
        "massless" (zero).
    sigmae : float
        DM-electron scattering cross section in cm^2
    suppress_darkelf_output : bool
        Whether to suppress the (useful but long) output that DarkELF gives
        when loading a material's properties.

    Returns
    -------
    fun : lambda function
        A function to calculate dRdE in DRU given E 

    """

    # Set up DarkELF Al2O3 (sapphire) object
    if suppress_darkelf_output:
        print('WARNING: You are suppressing DarkELF output')
        with io.capture_output() as captured:
            gaas_single = darkelf(target='GaAs',phonon_filename="GaAs_epsphonon_data10K.dat")
            gaas_multi = darkelf(target='GaAs')
    else:
        gaas_single = darkelf(target='GaAs',phonon_filename="GaAs_epsphonon_data10K.dat")
        gaas_multi = darkelf(target='GaAs')

    # Create anonymous function to get rate with only deposited energy
    # Note DarkELF expects recoil energies and WIMP masses in eV, and returns rates in counts/kg/yr/eV
    # But DarkLim expects recoil energies in keV, WIMP masses in GeV, and rates in counts/kg/day/keV (DRU)
    if mediator == 'massless':
        gaas_single.update_params(mX=mX_eV, mMed=0)
        gaas_multi.update_params(mX=mX_eV, mMed=0)
    elif mediator == 'massive':
        gaas_single.update_params(mX=mX_eV, mMed=1e11)
        gaas_multi.update_params(mX=mX_eV, mMed=1e11)
    fun = lambda keV : (1000 / 365.25) * \
            (gaas_single.dRdomega_phonon(keV * 1000, sigmae=sigmae)
             + gaas_multi._dR_domega_multiphonons_no_single(keV * 1000, ))

    return fun



