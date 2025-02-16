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

he_gain = 0.15


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

    if args.coincidence==1: # if coinc is 1, LEE is 'unknown'
        known_bkgs = [0]
    else:
        known_bkgs = [0,1]
    
    SE = darklim.sensitivity.SensEst(args.target_mass_kg, args.t_days, tm=args.target, eff=1., gain=he_gain, seed=(int(time.time() + mass*1e6)))
    SE.reset_sim()
    #SE.add_nfold_lee_bkgd(m=args.n_sensors, n=args.coincidence, w=args.window_s, e0=0.41e-3, R=33.)
    #SE.add_nfold_lee_bkgd(m=args.n_sensors, n=args.coincidence, w=args.window_s, e0=3.81e-3, R=0.0226)
    #SE.add_power_bkgd(1.4e-8, 5.77)
    #SE.add_power_bkgd(0.107, 2.72)

    SE.add_flat_bkgd(1) # flat background of 1 DRU
    SE.add_nfold_powerlaw_lee_bkgd(m=args.n_sensors,n=args.coincidence,w=args.window_s)
    
    per_device_threshold_keV = args.nsigma * args.baseline_res_eV * 1e-3
    threshold_keV = args.coincidence * per_device_threshold_keV

    # set max energy considered based on DM recoil spectrum:
    ehigh = darklim.sensitivity.edep_to_eobs(darklim.limit.drde_max_q(mass, tm=args.target),he_gain) + 10*np.sqrt(args.n_sensors)*args.baseline_res_eV * 1e-3
    print('ROI max: {:0.3f} keV for max DM mass of {:0.3f} GeV.'.format(ehigh,mass))
    
    """
    _, sigma = SE.run_sim(
            threshold_keV,
            e_high=50e-3,
            #e_low=1e-6,
            m_dms=[mass],
            nexp=args.nexp,
            #npts=100000,
            plot_bkgd=False,
            #res=args.baseline_res_eV*1e-3,
            res=np.sqrt(args.n_sensors)*args.baseline_res_eV*1e-3,
            verbose=True,
            sigma0=args.sigma0,
            elf_model=args.elf_model,
            elf_target=args.target,
            elf_params=args.elf_params,
            return_only_drde=False,
#            gaas_params=None
    )
    """
    # run
    m_dm, sig, ul, dm_rates, raw_dm_rates, exp_bkg = SE.run_fast_fc_sim(
        known_bkgs,
        threshold_keV,
        ehigh,
        e_low=1e-6, #threshold,
        m_dms=[mass],
        nexp=args.nexp,
        #npts=int(1e4),
        plot_bkgd=False,
        res=np.sqrt(args.n_sensors)*args.baseline_res_eV*1e-3,
        verbose=False,
        sigma0=args.sigma0,
        use_drdefunction=True,
        pltname='ULs_{:0.0f}d_{:d}device_{:d}fold_{:0.0f}mus'.format(args.t_days,args.n_sensors,args.coincidence,args.window_s/1e-6),
        savedir=args.results_dir
        #pltname=None
    )

    # plot acceptance and DM evt rate:
    #savename = \
    #'{:s}dmrate_{:0.0f}d_{:d}device_{:d}fold_{:0.0f}mus'.format(args.results_dir,args.t_days,args.n_sensors,args.coincidence,args.window_s/1e-6)
    #plot_dm_rates(m_dms,dm_rates,raw_dm_rates,sigma0,savename=savename)
    
    print(f'Done mass = {mass}, sigma = {sig}')

    return mass, sig, dm_rates/raw_dm_rates


    
def helium_scan():

    save = True
    
    # Read command-line arguments
    args = scanparser.get_scan_parameters()
    
    # Force some parameters
    args.target = 'He'
    args.elf_model = None
    # Write input parameters to a text file
    scanparser.write_info(args)

    print('running over masses:',args.masses_GeV)
    
    # Main parallel execution block
    with mp.Pool(processes=min(args.max_cpus, mp.cpu_count())) as pool:
        results = pool.starmap(process_mass, [(mass, args) for mass in args.masses_GeV])
     
    # save results to txt file
    sigma = np.zeros_like(args.masses_GeV)
    acceptance = np.zeros_like(args.masses_GeV)
    for i, result in enumerate(results):
        sigma[i] = result[1][0]
        acceptance[i] = result[2][0]
    
    if save:
        outname = args.results_dir + 'limit.txt'
        tot = np.column_stack( (args.masses_GeV, sigma, acceptance) )
        np.savetxt(outname, tot, fmt=['%.5e','%0.5e','%.5e'], delimiter=' ')
    
    return
    

# ------------------------------------------------------
# ------------------------------------------------------
    

if __name__ == "__main__":

    t_start = time.time()
    helium_scan()
    t_end = time.time()
    print(f'Full scan took {(t_end - t_start)/60:.2f} minutes.')

