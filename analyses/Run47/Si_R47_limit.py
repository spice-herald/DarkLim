import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy import interpolate

import darklim
from darklim import constants

import scanparser

from multihist import Hist1d
import time
import datetime

import multiprocessing as mp

##################################################################

def process_mass(mass, args, energies):
    # All the code that processes the mass value goes here, extracted from the original loop.

    # Calculate the dRdE vs E function
    drdefunction = None

    if args.elf_model is None:
        drdefunction = lambda x, mass=mass: darklim.limit.drde(x, mass, args.sigma0, args.target)
        
    elif args.elf_model == 'phonon' and args.target == 'Si':
        
        elf_mediator = args.elf_params['mediator']
        elf_suppress = args.elf_params['suppress_darkelf_output']
        elf_darkphoton = args.elf_params['dark_photon']

        drdefunction = \
            darklim.elf.get_dRdE_lambda_Si_phonon(mX_eV=mass*1e9, sigman=args.sigma0, mediator=elf_mediator,
                                                dark_photon=elf_darkphoton,
                                                suppress_darkelf_output=elf_suppress, gain=1.)

    elif args.elf_model == 'electron' and args.target == 'Si':

        elf_mediator = args.elf_params['mediator']
        elf_kcut = args.elf_params['kcut']
        elf_method = args.elf_params['method']
        elf_screening = args.elf_params['withscreening']
        elf_suppress = args.elf_params['suppress_darkelf_output']

        drdefunction = \
            darklim.elf.get_dRdE_lambda_Si_electron(mX_eV=mass*1e9, sigmae=args.sigma0, mediator=elf_mediator,
                                                kcut=elf_kcut, method=elf_method, withscreening=elf_screening,
                                                suppress_darkelf_output=elf_suppress, gain=1.)

    en_interp = np.geomspace(max(min(energies), args.threshold_keV), max(energies), int(1e4))
    eff_interp = np.ones_like(en_interp) * 0.7955
    
    try:
        rate_interp_DRU = drdefunction(en_interp)
    except ValueError:
        rate_interp_DRU = np.array([drdefunction(en) for en in en_interp])

    real_exposure_interp = eff_interp * args.exposure_kgd
    rate_interp_smeared = real_exposure_interp * \
        darklim.limit.gauss_smear(en_interp, rate_interp_DRU, args.baseline_res_eV*1e-3, gauss_width=10)
        
    sigma, _, _ = darklim.limit.optimuminterval(
        energies,
        en_interp, # efficiency curve energies
        eff_interp, # efficiency curve values
        [mass],
        args.exposure_kgd,
        tm=args.target,
        cl=0.9,
        verbose=True,
        hard_threshold=args.threshold_keV,
        sigma0=args.sigma0,
        en_interp=en_interp,
        rate_interp=[rate_interp_smeared],
    )
    
    return mass, sigma


    
def silicon_scan():
    
    # Read command-line arguments
    args = scanparser.get_scan_parameters()

    # Write input parameters to a text file
    scanparser.write_info(args)

    # Force some parameters based on R47 parameters
    args.target = 'Si'
    args.volume_cm3 = 0.1
    args.t_days = 3 / 24.
    args.exposure_kgd = 0.00003

    args.baseline_res_eV = 0.37
    args.nsigma = 5
    args.per_device_threshold_keV = args.nsigma * args.baseline_res_eV * 1e-3
    args.threshold_keV = args.coincidence * args.per_device_threshold_keV

    # Get real data
    energies_keV = np.load('spectra/BigFins_shared_0719.npy') / 1000

    # Main parallel execution block
    with mp.Pool(processes=min(args.max_cpus, mp.cpu_count())) as pool:
        results = pool.starmap(process_mass, [(mass, args, energies_keV) for mass in args.masses_GeV])

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
    silicon_scan()
    t_end = time.time()
    print(f'Full scan took {(t_end - t_start)/60:.2f} minutes.')

