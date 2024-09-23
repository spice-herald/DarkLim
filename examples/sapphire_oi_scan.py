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

import multiprocessing as mp


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



def process_mass(mass, args):
    # All the code that processes the mass value goes here, extracted from the original loop.

    SE = darklim.sensitivity.SensEst(args.target_mass_kg, args.t_days, tm=args.target, eff=1., gain=1.)
    SE.reset_sim()
    SE.add_flat_bkgd(1) # flat background of 1 DRU
    SE.add_nfold_lee_bkgd(m=args.n_sensors, n=args.coincidence, w=args.window_s)

    per_device_threshold_keV = args.nsigma * args.baseline_res_eV * 1e-3
    threshold_keV = args.coincidence * per_device_threshold_keV

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
        sigma = np.inf
    else:
        _, sigma = SE.run_sim(
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

    print(f'Done mass = {mass}, sigma = {sigma}')

    return mass, sigma


    
def sapphire_scan():
    
    # Read command-line arguments
    args = scanparser.get_scan_parameters()

    # Write input parameters to a text file
    scanparser.write_info(args)

    # Main parallel execution block
    with mp.Pool(processes=min(args.max_cpus, mp.cpu_count())) as pool:
        results = pool.starmap(process_mass, [(mass, args) for mass in args.masses_GeV])

    # save results to txt file
    sigma = np.zeros_like(args.masses_GeV)
    for i, result in enumerate(results):
        sigma[i] = result[1][0]

    outname = args.results_dir + 'limit.txt'
    tot = np.column_stack( (args.masses_GeV, sigma) )
    np.savetxt(outname, tot, fmt=['%.5e','%0.5e'], delimiter=' ')
    
    return
    

# ------------------------------------------------------
# ------------------------------------------------------
    

if __name__ == "__main__":

    t_start = time.time()
    sapphire_scan()
    t_end = time.time()
    print(f'Full scan took {(t_end - t_start)/60:.2f} minutes.')

