import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

import darklim
from darklim import constants

import scanparser

from multihist import Hist1d
import time
import datetime


##################################################################

def plot_dm_rates(m_dms,dm_rates,raw_dm_rates,sigma0,savename=None):
    
    #print('Signal events at m={:0.3f} GeV & {:0.1e} cm2: {:0.3e} evts'.format(mass,sigma0,signal_rates[ii]))
    
    # plot the evt rate vs mass:
    fig, ax = plt.subplots(1,figsize=(6,4))
    plt.plot(m_dms,dm_rates)
    #plt.plot(en_interp,curr_exp(en_interp),ls='--')
    #ax.axvline(threshold,ls='--',color='red')
    ax.set_ylabel('Events')
    ax.set_xlabel('Dark Matter Mass [GeV]')
    ax.set_xlim(m_dms[0],m_dms[-1])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Expected WIMP events at {:0.1e} cm2'.format(sigma0))
    
    if savename is not None:
        plt.savefig(savename+'_rate.png',facecolor='white',bbox_inches='tight')
    
    # plot the acceptance vs mass:
    fig, ax = plt.subplots(1,figsize=(6,4))
    plt.plot(m_dms,dm_rates/raw_dm_rates)
    ax.set_ylabel('Signal Acceptance')
    ax.set_xlabel('Dark Matter Mass [GeV]')
    ax.set_xlim(m_dms[0],m_dms[-1])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(1e-5,1)
    #ax.set_title('Signal Acceptance')
    
    if savename is not None:
        plt.savefig(savename+'_acceptance.png',facecolor='white',bbox_inches='tight')
    
    return

def sapphire_scan():
    
    args = scanparser.get_scan_parameters()
    
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    f = open(args.results_dir + 'info.txt', 'w')
    f.write(datetime.datetime.now().strftime('%m/%d/%Y, %H:%M:%S') + '\n\n')
    f.write(f'Detector material: {args.target}\n')
    f.write(f'Exposure time: {args.t_days} days\n')
    f.write(f'Detector volume: {args.volume_cm3} cm^3\n')
    f.write(f'Detector mass: {args.target_mass_kg} kg\n')
    f.write(f'Number of devices: {args.n_sensors}\n')
    f.write(f'Coincidence level: {args.coincidence}\n')
    f.write(f'Time window (s): {args.window_s}\n')
    f.write(f'Baseline energy resolution (eV): {args.baseline_res_eV}\n')
    f.write(f'Sigma above baseline for detection per sensor: {args.nsigma}\n')
    f.write(f'Dark matter masses (GeV/c2): ' + str(args.masses_GeV) + '\n')
    f.write(f'Default cross section (cm2): {args.sigma0:.4f}\n')
    f.write(f'Detector gain: 1\n')
    f.write('ELF model: ' + str(args.elf_model) + '\n')
    f.write('ELF params: ' + str(args.elf_params) + '\n') 
    f.close()
    
    ##################
    
    per_device_threshold_keV = args.nsigma * args.baseline_res_eV * 1e-3
    threshold_keV = args.coincidence * per_device_threshold_keV
    
    SE = darklim.sensitivity.SensEst(args.target_mass_kg, args.t_days, tm=args.target, eff=1., gain=1.)
    SE.reset_sim()
    SE.add_flat_bkgd(1) # flat background of 1 DRU

    SE.add_nfold_lee_bkgd(m=args.n_sensors, n=args.coincidence, w=args.window_s)
    
    sigma_out = np.zeros_like(args.masses_GeV)

    for i, mass in enumerate(args.masses_GeV):

        # First, figure out what the maximum energy from this dRdE is
        e_high_keV = 100. # keV
        e_low_keV = 1e-6 # keV
        drdefunction = SE.run_sim(
            threshold_keV,
            e_high=e_high_keV,
            e_low=e_low_keV,
            m_dms=[mass],
            sigma0=args.sigma0,
            elf_model=args.elf_model,
            elf_target=args.target,
            elf_params=args.elf_params,
            return_only_drde=True,
#            gaas_params=None
            )
        drdefunction = drdefunction[0]

        e_high_guesses = np.geomspace(e_low_keV, e_high_keV, 3000)
        skip = False
        try:
            drdefunction_guesses = drdefunction(e_high_guesses)
        except ValueError:
            drdefunction_guesses = np.array([drdefunction(en) for en in e_high_guesses])
        indices = np.where(drdefunction_guesses > 0)
        if len(indices[0]) == 0:
            e_high_keV = threshold_keV * 1.1
        else:
            j = int(indices[0][-1])
            e_high_keV = e_high_guesses[j] * 1.1
            if e_high_keV < threshold_keV:
                e_high_keV = threshold_keV * 1.1

        if skip:
            sigma_out[i] = np.inf
        else:
            _, sigma_out[i] = SE.run_sim(
                threshold_keV,
                e_high=e_high_keV,
                e_low=1e-6,
                m_dms=[mass],
                nexp=args.nexp,
                npts=100000,
                plot_bkgd=False,
                res=None,
                verbose=True,
                sigma0=args.sigma0,
                elf_model=args.elf_model,
                elf_target=args.target,
                elf_params=args.elf_params,
                return_only_drde=False,
    #            gaas_params=None
            )

        print(f'Done mass = {mass}, sigma = {sigma_out[i]}')

    # save results to txt file
    outname = args.results_dir + 'limit.txt'
    tot = np.column_stack( (args.masses_GeV, sigma_out) )
    np.savetxt(outname, tot, fmt=['%.5e','%0.5e'], delimiter=' ')
    
    return
    

# ------------------------------------------------------
# ------------------------------------------------------
    

if __name__ == "__main__":

    t_start = time.time()
    sapphire_scan()
    t_end = time.time()
    print(f'Full scan took {(t_end - t_start)/60:.2f} minutes.')

