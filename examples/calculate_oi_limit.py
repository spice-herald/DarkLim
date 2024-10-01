import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

import darklim
from darklim import constants
import darklim.elf as elf

import scanparser

from multihist import Hist1d
import time
import datetime

import multiprocessing as mp


##################################################################


def process_mass(mass, args, shared_energies_eV):
    # All the code that processes the mass value goes here, extracted from the original loop.

    eventenergies_keV = sorted(np.random.choice(shared_energies_eV, size=30, replace=False) * 1e-3)
    print('The chosen energies are:', eventenergies_keV)

    effenergies_keV = np.linspace(min(eventenergies_keV), max(eventenergies_keV), 10)
    effs = np.full(len(effenergies_keV), 0.7955)
    
    elf_mediator = 'massive'
    elf_kcut = 0
    elf_method = 'grid'
    elf_screening = True
    elf_suppress = False

    drdefunction = \
        [elf.get_dRdE_lambda_Si_electron(mX_eV=m*1e9, sigmae=args.sigma0, mediator=elf_mediator,
                                            kcut=elf_kcut, method=elf_method, withscreening=elf_screening,
                                            suppress_darkelf_output=elf_suppress, gain=1.)
         for m in mass]
        
    E_deposited_keV_arr = np.geomspace(min(eventenergies_keV), max(eventenergies_keV), 1000)
    for j, m in enumerate(mass):
        #try:
        #    dRdE_DRU_arr = drdefunction[j](E_deposited_keV_arr)
        #except ValueError:
        dRdE_DRU_arr = np.array([drdefunction[j](en) for en in E_deposited_keV_arr])

        check = sum(dRdE_DRU_arr > 0)
        if check == 0:
            print(f'No observed rate for mass {j}, {m} GeV')

        drdefunction[j] = lambda E: np.interp(E, E_deposited_keV_arr, dRdE_DRU_arr, left=0., right=0.)

    sigma, _, _ = darklim.limit.optimuminterval(eventenergies_keV, effenergies_keV, effs, mass, args.exposure_kgd,
                    tm=args.target, cl=0.9, res=args.baseline_res_eV*1e-3, gauss_width=3, verbose=True,
                    drdefunction=drdefunction, hard_threshold=args.baseline_res_eV*5*1e-3, sigma0=1e-41,
                    en_interp=None, rate_interp=None)

    print(f'Done mass = {mass}, sigma = {sigma}')

    return mass, sigma


    
def sapphire_scan():
    
    # Read command-line arguments
    args = scanparser.get_scan_parameters()

    # Write input parameters to a text file
    scanparser.write_info(args)

    # Main parallel execution block
    shared_energies_eV = np.load('spectra/BigFins_shared_0719.npy')
    #process_mass(args.masses_GeV, args, shared_energies_eV)
    
    if True:
        with mp.Pool(processes=min(args.max_cpus, mp.cpu_count())) as pool:
            results = pool.starmap(process_mass, [([mass], args, shared_energies_eV) for mass in args.masses_GeV])

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

