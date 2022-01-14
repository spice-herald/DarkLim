import numpy as np
import matplotlib.pyplot as plt
import types

from darklim.limit._limit import drde, gauss_smear


__all__ = [
    "RatePlot",
]


class RatePlot(object):
    """
    Class for making a plot of different dark matter spectra with the ability of comparing
    different data and with theoretical dR/dE curves.

    Attributes
    ----------
    fig : matplotlib.figure.Figure
        Matplotlib Figure object
    ax : matplotlib.axes.AxesSubplot
        Matplotlib Axes object
    _energy_range : array_like
        The energy range of the plot in keV.
    _spectrum_cmap : str
        The colormap to use for plotting each spectra from data.
    _drde_cmap : str
        The colormap to use for plotting each theoretical drde curve using the WIMP model.

    """

    def __init__(self, energy_range, spectrum_cmap="inferno", drde_cmap="viridis", figsize=(10, 6)):
        """
        Initialization of the RatePlot class for plotting dark matter spectra.

        Parameters
        ----------
        energy_range : array_like
            The energy range of the plot in keV.
        spectrum_cmap : str, optional
            The colormap to use for plotting each spectra from data. Default is "inferno".
        drde_cmap : str, optional
            The colormap to use for plotting each theoretical drde curve using the WIMP
            model. Default is "viridis".
        figsize : tuple, optional
            Width and height of the figure in inches. Default is (10, 6).

        Returns
        -------
        None

        """

        self._energy_range = energy_range
        self._spectrum_cmap = spectrum_cmap
        self._drde_cmap = drde_cmap


        self.fig, self.ax = plt.subplots(figsize=figsize)

        self.ax.grid()
        self.ax.grid(which="minor", axis="both", linestyle="dotted")
        self.ax.tick_params(which="both", direction="in", right=True, top=True)

        self.ax.set_yscale('log')
        self.ax.set_xlim(self._energy_range)
        self.ax.set_ylabel("$\partial R/\partial E_r$ [evts/keV/kg/day]")
        self.ax.set_xlabel("Energy [keV]")
        self.ax.set_title(
            f"Spectrum of Events from {self._energy_range[0]:.1f} to "
            f"{self._energy_range[1]:.1f} keV"
        )


    def _update_colors(self, linestyle):
        """
        Helper method for updating the line colors whenever a new line is added.

        Parameters
        ----------
        linestyle : str
            The linestyle to update all the plot colors for. Should be "-" or "--".

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If `linestyle` is not "-" or "--".

        Notes
        -----
        Given a linestyle, this method checks how many lines have the specified style.
        This method assumes that data curves have a solid linestyle and theoretical
        DM curves have a dashed linestyle.

        """

        if linestyle == "-":
            cmap = self._spectrum_cmap
        elif linestyle == "--":
            cmap = self._drde_cmap
        else:
            raise ValueError("The inputted linestyle is not supported.")

        n_lines = sum(line.get_linestyle() == linestyle for line in self.ax.lines)
        line_colors = plt.cm.get_cmap(cmap)(np.linspace(0.1, 0.9, n_lines))
        ii = 0

        for line in self.ax.lines:
            if line.get_linestyle() == linestyle:
                line.set_color(line_colors[ii])
                ii += 1

        self.ax.legend(loc="upper right")

    def add_data(self, energies, exposure, efficiency=None, label=None, nbins=100, **kwargs):
        """
        Method for plotting a single spectrum in evts/keV/kg/day from inputted data.

        Parameters
        ----------
        energies : array_like
            The energies of the events that will be used when plotting, in units of keV.
        exposure : float
            The exposure of DM search with respect to the inputted spectrum, in units
            of kg-days.
        efficiency : float, FunctionType, NoneType, optional
            The efficiency of the cuts that were applied to the inputted energies. This
            can be passed as a float, as a function of energy in keV, or left as None if
            no efficiency correction will be done.
        label : str, optional
            The label for this data to be used in the plot. If left as None, no label is added.
        nbins : int, optional
            The number of bins to use in the plot.
        kwargs
            The keyword arguments to pass to `matplotlib.pyplot.step`.

        Returns
        -------
        None

        """

        hist, bin_edges = np.histogram(energies, bins=nbins, range=self._energy_range)
        bin_cen = (bin_edges[:-1]+bin_edges[1:])/2
        rate = hist / np.diff(bin_cen).mean() / exposure

        if np.isscalar(efficiency):
            rate /= efficiency
        elif isinstance(efficiency, types.FunctionType):
            rate /= efficiency(bin_cen)

        self.ax.step(bin_cen, rate, where='mid', label=label, linestyle='-', **kwargs)
        self._update_colors("-")
        
    def add_drde(self, masses, sigmas, tm="Si", npoints=1000, res=None, gauss_width=10, **kwargs):
        """
        Method for plotting the expected dark matter spectrum for specified masses and
        cross sections in evts/keV/kg/day.

        Parameters
        ----------
        masses : float, array_like
            The dark matter mass at which to calculate/plot the expected differential
            scattering rate. Expected units are GeV.
        sigmas : float, array_like
            The dark matter cross section at which to calculate/plot the expected
            differential scattering rate. Expected units are cm^2.
        tm : str, int, optional
            The target material of the detector. Can be passed as either the atomic symbol, the
            atomic number, or the full name of the element. Default is 'Si'.
        npoints : int, optional
            The number of points to use in the dR/dE plot. Default is 1000.
        res : float, NoneType, optional
            The width of the gaussian (1 standard deviation) to be used to smear differential
            scatter rate in the plot. Should have units of keV. None by default, in which no
            smearing is done.
        gauss_width : float, optional
            If `res` is not None, this is the number of standard deviations of the Gaussian
            distribution that the smearing will go out to. Default is 10.
        kwargs
            The keyword arguments to pass to `matplotlib.pyplot.plot`.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If `masses` and `sigmas` are not the same length.

        """

        if np.isscalar(masses):
            masses = [masses]
        if np.isscalar(sigmas):
            sigmas = [sigmas]
        if len(masses) != len(sigmas):
            raise ValueError("masses and sigmas must be the same length.")

        xvals = np.linspace(self._energy_range[0], self._energy_range[1], npoints)

        for m, sig in zip(masses, sigmas):
            drde = drde(xvals, m, sig, tm=tm)
            label = f"DM Mass = {m:.2f} GeV, σ = {sig:.2e} cm$^2$"
            if res is not None:
                drde = gauss_smear(xvals, drde, res, gauss_width=gauss_width)
                label += f"\nWith {gauss_width}$\sigma_E$ Smearing"

            self.ax.plot(
                xvals,
                drde,
                linestyle='--',
                label=label,
                **kwargs,
            )

        self._update_colors("--")

