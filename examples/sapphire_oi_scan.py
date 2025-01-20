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

    SE = darklim.sensitivity.SensEst(args.target_mass_kg, args.t_days, tm=args.target, eff=1., gain=1., seed=(int(time.time() + mass*1e6)))
    SE.reset_sim()
    #SE.add_nfold_lee_bkgd(m=args.n_sensors, n=args.coincidence, w=args.window_s, e0=0.41e-3, R=33.)
    #SE.add_nfold_lee_bkgd(m=args.n_sensors, n=args.coincidence, w=args.window_s, e0=3.81e-3, R=0.0226)
    SE.add_power_bkgd(1.4e-8, 5.77)
    SE.add_power_bkgd(0.107, 2.72)

    per_device_threshold_keV = args.nsigma * args.baseline_res_eV * 1e-3
    threshold_keV = args.coincidence * per_device_threshold_keV

    _, sigma = SE.run_sim(
            threshold_keV,
            e_high=50e-3,
            #e_low=1e-6,
            m_dms=[mass],
            nexp=args.nexp,
            #npts=100000,
            plot_bkgd=False,
            res=args.baseline_res_eV*1e-3,
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
    
    # Force some parameters
    args.target = 'Al2O3'

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

