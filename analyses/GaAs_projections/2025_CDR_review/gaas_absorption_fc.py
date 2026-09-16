import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import math

import darklim
from darklim import constants

import scanparser

from multihist import Hist1d
import time
import datetime

import multiprocessing as mp

##################################################################


def process_mass(mass, args):
    # All the code that processes the mass value goes here, extracted from the original loop.

    # Load background template
    E_arr_template, dRdE_bkgd_arr_template, _ = np.loadtxt(args.LEE_filename, unpack=True)
    
    # Get energy range for this mass
    E_th_keV = args.nsigma * args.baseline_res_eV * 1e-3  # threshold in keV
    E_min_sig_keV = mass * 1e6 - 3 * args.baseline_res_eV * 1e-3
    E_max_sig_keV = mass * 1e6 + 3 * args.baseline_res_eV * 1e-3
    E_th_updated_keV = max(E_th_keV, E_min_sig_keV, 0.)
    
    if E_max_sig_keV < E_th_updated_keV:
        print(f"Mass {mass*1e12:.3e} meV is below threshold for detection. Skipping...")
        return mass, [np.inf]
    else:
        E_arr = np.linspace(E_th_updated_keV, E_max_sig_keV, 10_000)
    
    # Scale background template to exposure time and target mass
    dRdE_bkgd_arr = np.interp(E_arr, E_arr_template, dRdE_bkgd_arr_template)  # interpolate to new energy range
    dRdE_bkgd_arr *= (args.volume_cm3 / 0.1) # scale to 0.1 cm3 (Run 57 Si)
    dRdE_bkgd_arr /= args.LEE_improvement # Improvement factor for gaas
    dRdE_bkgd_arr *= args.t_days # scale to exposure time; new units are cts/keV

    # Load signal template for bosonic absorption
    dRdE_signal_fun = darklim.elf.get_dRdE_lambda_GaAs_absorption(mX_eV=mass*1e9, kappa=args.sigma0, res_eV=args.baseline_res_eV, suppress_darkelf_output=False)
    dRdE_signal_arr = dRdE_signal_fun(E_arr)
    dRdE_signal_arr *= (args.t_days * args.target_mass_kg) # Scale to exposure; new units are cts/keV
    
    R_signal = np.trapz(dRdE_signal_arr, E_arr)  # total signal counts
    R_bkgd = np.trapz(dRdE_bkgd_arr, E_arr)      # total background counts
    
    if R_bkgd > 1000:
        R_desired = R_bkgd + np.sqrt(R_bkgd) * 1.28
    elif R_bkgd > 10:
        R_desired = darklim.feldman_cousins.FC_ints(int(R_bkgd), 0.)[1]        
    else:
        lower = math.floor(R_bkgd)
        upper = lower + 1
        fraction = R_bkgd - lower
        R_desired_a = darklim.feldman_cousins.FC_ints(lower, 0.)[1]
        R_desired_b = darklim.feldman_cousins.FC_ints(upper, 0.)[1]
        R_desired = (R_desired_a + fraction * (R_desired_b - R_desired_a)) 
    
    x_lim = np.sqrt(R_desired / R_signal) * args.sigma0
   
    return mass, [x_lim]


    
def gaas_scan():
    
    # Read command-line arguments
    args = scanparser.get_scan_parameters()
    
    # Force some parameters
    args.target = 'GaAs'
    args.masses_GeV = np.append(np.geomspace(1e-12, 6e-10, 72), np.geomspace(1e-9, 150e-9, 72))

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
    gaas_scan()
    t_end = time.time()
    print(f'Full scan took {(t_end - t_start)/60:.2f} minutes.')

