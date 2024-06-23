import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, special

import mendeleev
from darklim import constants
from darklim.limit._limit import drde, optimuminterval, fc_limits
from darklim.sensitivity._random_sampling import pdf_sampling
from darklim.sensitivity._plotting import RatePlot


__all__ = [
    "calculate_substrate_mass",
    "trigger_pdf",
    "n_fold_lee",
    "edep_to_eobs",
    "eobs_to_edep",
    "drde_obs",
    "drde_obs_test",
    "drde_wimp_obs",
    "SensEst"
]


def calculate_substrate_mass(vol, tm):
    """
    Helper function for calculating the mass of substrate given its
    volume.

    Parameters
    ----------
    vol : float
        The volume of the substrate that we want to know the mass of,
        in units of meters^3.
    tm : str, int
        The target material of the detector. Can be passed as either
        the atomic symbol, the atomic number, or the full name of the
        element.

    Returns
    -------
    mass : float
        The mass, in kg, of the substrate based on the inputted volume
        and target material.

    """

    conv_factor = constants.centi**3 * constants.kilo
    # density in kg/m^3
    rho = mendeleev.element(tm).density / conv_factor
    print('material density is {:0.1f} kg/m3'.format(rho))
    mass = rho * vol

    return mass


def trigger_pdf(x, sigma, n_win):
    """
    Function for calculating the expected PDF due to a finite trigger
    window, based on the Optimum Filter method. Outputs in units of
    1 / [units of `sigma`].

    Parameters
    ----------
    x : ndarray
        The values at which the PDF will be evaluated. Units are
        arbitrary, but note that this function is usually used with
        regards to energy.
    sigma : float
        The detector resolution in the same units of `x`.
    n_win : float
        The number of independent samples in a trigger search window.
        If there is correlated noise, this value can be shorter than
        the number of samples in the trigger window.

    Returns
    -------
    pdf : ndarray
        The PDF for noise triggers from the given inputs.

    """

    normal_dist = lambda xx: stats.norm.pdf(xx, scale=sigma)
    erf_scale = lambda xx: special.erf(xx / np.sqrt(2 * sigma**2))

    pdf = n_win * normal_dist(x) * (0.5 * (1 + erf_scale(x)))**(n_win - 1)

    return pdf

def n_fold_lee(x,m=1,n=1,e0=0.020,R=0.12,w=100e-6):
    '''
    e0 has units keV
    R has units 1/seconds
    w, coincidence window, units are seconds
    '''
    if n>m:
        raise ValueError(
            "Coincidence level (n) cannot exceed the number of devices (m)."
        )
    term1 = special.factorial(m)/special.factorial(m-n)
    term2 = R**n * w**(n-1)
    pile_up_rate = term1*term2
    
    dist = pile_up_rate * 3600 * 24 * stats.erlang.pdf(x,a=n,loc=0,scale=e0)
    
    return dist

def edep_to_eobs(e_dep,gain):
    return e_dep*gain

def eobs_to_edep(e_obs,gain):
    return e_obs/gain

#def drde_obs(e_obs, drde_func, gain):
def drde_obs(drde_func, gain):
    """
    given an observed energy value (e_obs) and a signal spectrum,
    differential in the *deposited energy* (drde_func), return the 
    spectrum function differential in the observed energy.
    assumed the obs and dep energies are related by a gain factor.
    """
    return lambda x : (1/gain) * drde_func( eobs_to_edep(x,gain) )
    #return (1/gain) * drde_func( get_e_dep(e_obs,gain) )

def drde_wimp_obs(eobs, m, sigma0, tm, gain):
    """
    given an observed energy value (e_obs) return the WIMP NR 
    differential rate, now differential in the observed energy.
    """
    return (1/gain) * drde( eobs_to_edep(eobs,gain) ,m, sigma0, tm )
    
def drde_obs_test(e_obs, drde_func, gain):
    """
    test for debugging/understanding
    """
    return (1/gain) * drde_func( eobs_to_edep(e_obs,gain) )

class SensEst(object):
    """
    Class for setting up and running an estimate of the sensitivity of
    a device, given expected backgrounds.

    Attributes
    ----------
    m_det : float
        Mass of the detector in kg.
    tm : str, int
        The target material of the detector. Can be passed as either
        the atomic symbol, the atomic number, or the full name of the
        element.
    exposure : float
        The total exposure of the detector in units of kg*days.

    """

    def __init__(self, m_det, time_elapsed, eff=1, tm="Si", gain=1):#, signal_name='SI-NR'):
        """
        Initialization of the SensEst class.

        Parameters
        ----------
        m_det : float
            Mass of the detector in kg.
        time_elapsed : float
            The time elapsed for the simulated experiment, in days.
        eff : float, optional
            The estimated efficiency due to data selection, live time
            losses, etc. Default is 1.
        tm : str, int, optional
            The target material of the detector. Can be passed as
            either the atomic symbol, the atomic number, or the full
            name of the element. Default is 'Si'.

        """

        self.m_det = m_det
        self.tm = tm
        self.exposure = m_det * time_elapsed * eff
        self.gain = gain
        
        self._backgrounds = []
        self._background_labels = []
        
        #self.signal_name = signal_name
        
        #self.signal = None
        #if self.signal_name=='SI-NR':
        #    self.add_signal_model
        
    def add_flat_bkgd(self, flat_rate):
        """
        Method for adding a flat background to the simulation.

        Parameters
        ----------
        flat_rate : float
            The flat background rate, in units of events/kg/kev/day
            (DRU).

        """

        flat_bkgd = lambda x: flat_rate * np.ones(len(x))
        self._backgrounds.append(flat_bkgd)
        self._background_labels.append('{:0.2f} DRU Bkg'.format(flat_rate))

    def add_noise_bkgd(self, sigma, n_win, fs):
        """
        Method for adding a noise background to the simulation.

        Parameters
        ----------
        sigma : float
            The detector resolution in units of keV.
        n_win : float
            The number of independent samples in a trigger search
            window. If there is correlated noise, this value can be
            shorter than the number of samples in the trigger window.
        fs : float
            The digitization rate of the data being used in the trigger
            algorithm, units of Hz.

        """

        norm = self.m_det * n_win / fs / constants.day
        noise_bkgd = lambda x: trigger_pdf(x, sigma, n_win) / norm
        self._backgrounds.append(noise_bkgd)
        self._background_labels.append('Noise Bkg')

    def add_dm_bkgd(self, m_dm, sig0, res=None):
        """
        Method for adding a DM background to the simulation.

        Parameters
        ----------
        m_dm : float
            The dark matter mass at which to calculate the expected
            differential event rate. Expected units are GeV/c^2.
        sig0 : float
            The dark matter cross section at which to calculate the
            expected differential event rate. Expected units are cm^2.
        res : float
            Energy resolution if one desires to smear the DM spectrum.
        """

        dm_bkgd = lambda x: drde(x, m_dm, sig0, tm=self.tm)
        self._backgrounds.append(dm_bkgd)
        self._background_labels.append('{:0.2f} GeV DM'.format(m_dm))


    def add_arb_bkgd(self, function, label='Arb. Bkg'):
        """
        Method for adding an arbitrary background to the simulation.

        Parameters
        ----------
        function : FunctionType
            A function that returns a background rate in units of
            events/kg/keV/day when inputted energy, where the energies
            are in units of keV.

        """

        self._backgrounds.append(function)
        self._background_labels.append(label)
    
    def add_nfold_lee_bkgd(self,m=1,n=1,e0=0.020,R=0.12,w=100e-6):
        """
        Method for adding a flat background to the simulation.

        Parameters
        ----------
        m : int
            Total number of devices
        n : int
            Coincidence level
        e0 : float
        R : float
        w : float
            Coincidence window length in seconds. 
        """
    
        nfold_lee = lambda x: n_fold_lee(x,m=m,n=n,e0=e0,R=R,w=w) / self.m_det
        self._backgrounds.append(nfold_lee)
        self._background_labels.append('{:d}-fold LEE in {:d} devices'.format(n,m))

    def reset_sim(self):
        """Method for resetting the simulation to its initial state."""

        self._backgrounds = []


    def run_sim(self, threshold, e_high, e_low=1e-6, m_dms=None, nexp=1, npts=1000,
                plot_bkgd=False, res=None, verbose=False, sigma0=1e-41):
        """
        Method for running the simulation for getting the sensitivity
        estimate.

        Parameters
        ----------
        threshold : float
            The energy threshold of the experiment, units of keV.
        e_high : float
            The high energy cutoff of the analysis range, as we need
            some cutoff to the event energies that we generate.
        m_dms : ndarray, optional
            Array of dark matter masses (in GeV/c^2) to run the Optimum
            Interval code. Default is 50 points from 0.05 to 2 GeV/c^2.
        nexp : int, optional
            The number of experiments to run - the median of the
            outputs will be taken. Recommended to set to 1 for
            diagnostics, which is default.
        npts : int, optional
            The number of points to use when interpolating the
            simulated background rates. Default is 1000.
        plot_bkgd : bool, optional
            Option to plot the background being used on top of the
            generated data, for diagnostic purposes. If `nexp` is
            greater than 1, then only the first generated dataset is
            plotted.

        Returns
        -------
        m_dms : ndarray
            The dark matter masses in GeV/c^2 that upper limit was set
            at.
        sig : ndarray
            The cross section in cm^2 that the upper limit was
            determined to be.

        """

        sigs = []

        if m_dms is None:
            m_dms = np.geomspace(0.5, 2, num=50)

        en_interp = np.geomspace(e_low, e_high, num=npts)
        
        #drdefunction = [drde_obs(en_interp,lambda x: drde(x,m,sigma0,tm=self.tm),gain)*np.heaviside(en_interp-threshold,1) for m in m_dms]
        
        for ii in range(nexp):
            evts_sim = self._generate_background(
                en_interp, plot_bkgd=plot_bkgd and ii==0,
            )
            
            sig_temp, _, _ = optimuminterval(
                evts_sim[evts_sim >= threshold], # evt energies
                en_interp, # efficiency curve energies
                np.heaviside(en_interp - threshold, 1), # efficiency curve values
                m_dms, # mass list
                self.exposure, #exposure
                tm=self.tm, # target material
                cl=0.9, # C.L.
                res=res, # include smearing of DM spectrum
                gauss_width=10, # if smearing, number of sigma to go out to
                verbose=verbose, # print outs
                drdefunction=None, # 
                hard_threshold=threshold,
                sigma0=sigma0
            )

            sigs.append(sig_temp)

        sig = np.median(np.stack(sigs, axis=1), axis=1)

        return m_dms, sig
    
    def run_sim_fc(self, known_bkgs_list, threshold, e_high, e_low=1e-6, m_dms=None, nexp=1, npts=1000,
                plot_bkgd=False, res=None, verbose=False, sigma0=1e-41,use_drdefunction=False):
        """
        Method for running the simulation for getting the sensitivity
        estimate using FC intervals.

        Parameters
        ----------
        known_bkg_index : ndarray, int
            Indeces of the bkg components to be treated as 'known'.
        threshold : float
            The energy threshold of the experiment, units of keV.
        e_high : float
            The high energy cutoff of the analysis range, as we need
            some cutoff to the event energies that we generate.
        m_dms : ndarray, optional
            Array of dark matter masses (in GeV/c^2) to run the Optimum
            Interval code. Default is 50 points from 0.05 to 2 GeV/c^2.
        nexp : int, optional
            The number of experiments to run - the median of the
            outputs will be taken. Recommended to set to 1 for
            diagnostics, which is default.
        npts : int, optional
            The number of points to use when interpolating the
            simulated background rates. Default is 1000.
        plot_bkgd : bool, optional
            Option to plot the background being used on top of the
            generated data, for diagnostic purposes. If `nexp` is
            greater than 1, then only the first generated dataset is
            plotted.

        Returns
        -------
        m_dms : ndarray
            The dark matter masses in GeV/c^2 that upper limit was set
            at.
        sig : ndarray
            The cross section in cm^2 that the upper limit was
            determined to be.

        """

        sigs = []
        uls = []

        if m_dms is None:
            m_dms = np.geomspace(0.5, 2, num=50)
        
        #if verbose:
        #    print('Running over the following masses:',m_dms)
        
        en_interp = np.geomspace(e_low, e_high, num=npts)
        
        # created summed 'known' background function:
        known_bkgd_func = lambda x: np.stack([bkgd(x) for ind,bkgd in enumerate(self._backgrounds) if ind in known_bkgs_list], axis=1,).sum(axis=1)
        print('Treating the following as known bkgs for FC limits: ')
        for idx in known_bkgs_list:
            print('   ',self._background_labels[idx])
        
        # create signal model functions at each mass
        drdefunction = None
        if use_drdefunction:
            drdefunction = [ lambda x,m: drde_wimp_obs( x, m, sigma0, self.tm, self.gain ) for m in m_dms ]
            #drdefunction = [ drde_obs( lambda x: drde(x,m,sigma0,tm=self.tm), self.gain ) for m in m_dms ]
            #drdefunction = [drde_obs(en_interp,lambda x: drde(x,m,sigma0,tm=self.tm),self.gain) for m in m_dms]
            
        for ii in range(nexp):
            if ii%10==0:
                print('\n Running toy number {}...'.format(ii))
            
            # generate a toy:
            evts_sim = self._generate_background(
                en_interp, plot_bkgd=plot_bkgd and ii==0,
            )
            
            # sig_temp has length = number of DM masses
            # ul_temp is a scalar - UL in number of evts is same for all DM masses
            
            sig_temp, ul_temp = fc_limits(
                known_bkgd_func,
                evts_sim[evts_sim >= threshold], # evt energies
                en_interp, # efficiency curve energies
                np.heaviside(en_interp - threshold, 1), # efficiency curve values
                m_dms, # mass list
                self.exposure, #exposure
                tm=self.tm, # target material
                res=res, # include smearing of DM spectrum
                gauss_width=10, # if smearing, number of sigma to go out to
                verbose=verbose, # print outs
                drdefunction=drdefunction, # 
                hard_threshold=threshold,
                sigma0=sigma0
            )
            
            #print(sig_temp)
            #print(ul_temp)
            
            sigs.append(sig_temp)
            uls.append(ul_temp)

        sig = np.median(np.stack(sigs, axis=1), axis=1)
        ul = np.median(np.asarray(uls))
        return m_dms, sig, ul
    
    def generate_background(self, e_high, e_low=1e-6, npts=1000,
                            plot_bkgd=False,verbose=False):
        """
        Method for generating events based on the inputted background.

        Parameters
        ----------
        e_high : float
            The high energy cutoff of the analysis range, as we need
            some cutoff to the event energies that we generate.
        e_low : float, optional
            The low energy cutoff of the analysis range, default is 1e-6.
        npts : int, optional
            The number of points to use when interpolating the
            simulated background rates. Default is 1000.
        plot_bkgd : bool, optional
            Option to plot the background being used on top of the
            generated data, for diagnostic purposes. If `nexp` is
            greater than 1, then only the first generated dataset is
            plotted.

        Returns
        -------
        evts_sim : ndarray
            The array of all the simulated events based on the inputted
            backgrounds. Units are keV.

        Raises
        ------
        ValueError
            If `self._backgrounds` is an empty list (no backgrounds
            have been added).

        """

        en_interp = np.geomspace(e_low, e_high, num=npts)
        evts_sim = self._generate_background(en_interp, plot_bkgd=plot_bkgd,verbose=verbose)

        return evts_sim

    def _generate_background(self, en_interp, plot_bkgd=False, verbose=False, nbins=100):
        """
        Hidden method for generating events based on the inputted
        background.

        Parameters
        ----------
        en_interp : ndarray
            The energies at which the total simulated background rate
            will be interpolated, in units of keV.
        plot_bkgd : bool, optional
            Option to plot the background being used on top of the
            generated data, for diagnostic purposes. If `nexp` is
            greater than 1, then only the first generated dataset is
            plotted.

        Returns
        -------
        evts_sim : ndarray
            The array of all the simulated events based on the inputted
            backgrounds. Units are keV.

        Raises
        ------
        ValueError
            If `self._backgrounds` is an empty list (no backgrounds
            have been added).

        """

        if len(self._backgrounds) == 0:
            raise ValueError(
                "No backgrounds have been added, "
                "add some using the methods of SensEst."
            )

        e_high = en_interp.max()
        e_low = en_interp.min()
        npts = len(en_interp)

        tot_bkgd = np.zeros(npts)

        for bkgd in self._backgrounds:
            tot_bkgd += bkgd(en_interp)

        tot_bkgd_func = lambda x: np.stack(
            [bkgd(x) for bkgd in self._backgrounds], axis=1,
        ).sum(axis=1)

        rtot = np.trapz(tot_bkgd_func(en_interp), x=en_interp)

        nevts_exp = rtot * self.exposure
        nevts_sim = np.random.poisson(nevts_exp)
        if verbose:
            print('expect {:0.1f} evts'.format(nevts_exp))
            print('created {:0.1f} evts'.format(nevts_sim))
        
        evts_sim = pdf_sampling(
            tot_bkgd_func, (e_low, e_high), npoints=npts, nsamples=nevts_sim,
        )

        if plot_bkgd:
            self._plot_bkgd(evts_sim, en_interp, tot_bkgd_func, nbins=nbins)

        return evts_sim

    def _plot_bkgd(self, evts, en_interp, tot_bkgd_func, nbins=100):
        """
        Hidden Method for plotting the generated events on top of the
        inputted backgrounds.

        """

        ratecomp = RatePlot(
            (en_interp.min(), en_interp.max()), figsize=(9, 6),
        )
        ratecomp.add_data(
            evts,
            self.exposure,
            nbins=nbins,
            label="Simulated Events Spectrum",
        )

        if len(self._backgrounds) > 1:
            for ii, bkgd in enumerate(self._backgrounds):
                ratecomp.ax.plot(
                    en_interp,
                    bkgd(en_interp),
                    linestyle='--',
                    label=self._background_labels[ii],
                )

        ratecomp.ax.plot(
            en_interp,
            tot_bkgd_func(en_interp),
            linestyle='--',
            label="Total Background",
        )

        ratecomp._update_colors('--')

        ratecomp.ax.set_xlabel("Event Energy [keV]")
        ratecomp.ax.set_title("Spectrum of Simulated Events")
        ratecomp.ax.set_ylim(
            tot_bkgd_func(en_interp).min() * 0.1,
            tot_bkgd_func(en_interp).max() * 10,
        )

        ratecomp.ax.legend(fontsize=14)
        list_of_text = [
            ratecomp.ax.title,
            ratecomp.ax.xaxis.label,
            ratecomp.ax.yaxis.label,
        ]

        for item in list_of_text:
            item.set_fontsize(14)

        ratecomp.fig.tight_layout()

