import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

import darklim
from darklim import constants

from multihist import Hist1d
import time
import datetime

##################################################################

efficiency = 1.0
tm = 'Al2O3' # target name
energy_res = 0.100e-3 # energy resolution in keV
det_gain = 1.

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

def run_scan_point(nexp,time_elapsed,mass_det,n_devices,coinc,window,var_threshold=False,save=True,savedir=None,
    m_dms=np.geomspace(0.0001,5,50),sigma0=1e-36,elf_model=None, elf_target=None, elf_params=None):
    
    if var_threshold:
        nsigma = stats.norm.isf(stats.norm.sf(5)**(1/coinc))
    else:
        nsigma = 5
    
    per_device_threshold = nsigma * energy_res # threshold
    threshold = coinc*per_device_threshold
    
    SE = darklim.sensitivity.SensEst(mass_det, time_elapsed, eff=efficiency, tm=tm, gain=det_gain)
    SE.reset_sim()
    SE.add_flat_bkgd(1) # flat background of 1 DRU

    SE.add_nfold_lee_bkgd(m=n_devices,n=coinc,w=window)
    
    print('\nRunning with the following settings:')
    print('Mass: {:0.4f} kg; Time: {:0.3f} d => Exposure: {:0.3f} kg-d'.format(SE.m_det,time_elapsed,SE.exposure))
    print('Coincidence: {:d}-fold in {:d} devices; {:0.1f} microsecond window'.format(coinc,n_devices,window/1e-6))
    print('Energy threshold: {:0.3f} eV; {:0.2f} sigma in each device'.format(threshold*1e3,nsigma))

    # run
    sig = np.zeros_like(m_dms)
    ul = np.zeros_like(m_dms)
    dm_rates = np.zeros_like(m_dms)
    raw_dm_rates = np.zeros_like(m_dms)
    exp_bkg = np.zeros_like(m_dms)

    if np.isscalar(sigma0):
        sigma0_arr = np.full(len(m_dms), sigma0)
    else:
        sigma0_arr = np.copy(sigma0)

    for i, mass in enumerate(m_dms):

        sigma0_i = sigma0_arr[i]
        if sigma0_i  == np.inf:
            print(f'Infinite, skipping mass {mass}')
            continue

        # First, figure out what the maximum energy from this dRdE is
        ehigh = 1. # keV
        drdefunction = SE.run_sim(
            threshold,
            ehigh,
            e_low=1e-6,
            m_dms=[mass],
            nexp=1,
            npts=100000,
            plot_bkgd=False,
            res=None,
            verbose=False,
            sigma0=sigma0_i,
            elf_model=elf_model,
            elf_target=elf_target,
            elf_params=elf_params,
            return_only_drde=True,
#            gaas_params=None
            )
        drdefunction = drdefunction[0]

        ehigh_guesses = np.geomspace(1e-6, 1e3, 3000)
        try:
            drdefunction_guesses = drdefunction(ehigh_guesses)
        except ValueError:
            drdefunction_guesses = np.array([drdefunction(en) for en in ehigh_guesses])
        indices = np.where(drdefunction_guesses > 0)
        if len(indices[0]) == 0:
            ehigh = 1.
        else:
            j = int(indices[0][-1])
            ehigh = ehigh_guesses[j] * 1.1
            if ehigh < threshold:
                ehigh = 1.

        _, sig[i] = SE.run_sim(
            threshold,
            ehigh,
            e_low=1e-6,
            m_dms=[mass],
            nexp=nexp,
            npts=100000,
            plot_bkgd=False,
            res=None,
            verbose=True,
            sigma0=sigma0_i,
            elf_model=elf_model,
            elf_params=elf_params,
            elf_target=elf_target,
            return_only_drde=False,
#            gaas_params=None
        )

        print(f'Done mass = {mass}, sigma = {sig[i]}')

    # save results to txt file
    if save and savedir is not None:
        outname = './{:s}/HeRALD_FC_{:0.0f}d_{:d}device_{:d}fold_{:0.0f}mus.txt'.format(savedir,time_elapsed,n_devices,coinc,window/1e-6)
        tot = np.column_stack( (m_dms, sig) )
        np.savetxt(outname,tot,fmt=['%.5e','%0.5e'] ,delimiter=' ')
    
    return
    
def sapphire_scan(results_dir):
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    nexp = 100 # number of toys
    
    var_threshold = True # vary 5sigma requirement based on coinc level
    
    times = np.array([1/24]) # days
    mass_det = 1. * constants.Al2O3_density * 1e-3 # mass in kg, = (0.1 cm)^3
    exposures = times*mass_det
    
    n_devices = 1
    coinc = np.array([1])
    window = 100e-6 # s

    m_dms = np.geomspace(3e-4, 1e3, 40)
    sigma0 = np.full_like(m_dms, 1e-35)

#    elf_model='electron'
#    elf_params={'mediator': 'massless', 'kcut': 0, 'method': 'grid', 'withscreening': True, 'suppress_darkelf_output': False}
#    elf_model='phonon'
#    elf_params={'mediator': 'massive', 'suppress_darkelf_output': False, 'dark_photon': False}
    elf_model = None
    elf_params = {}

    if var_threshold:
        nsigma = stats.norm.isf(stats.norm.sf(5)**(1/coinc))
    else:
        nsigma = 5

    f = open(results_dir + '/info.txt', 'w')
    f.write(datetime.datetime.now().strftime('%m/%d/%Y, %H:%M:%S') + '\n\n')
    f.write('Detector material: ' + tm + '\n')
    f.write('Exposure time(s) (days): ' + str(times) + '\n')
    f.write('Detector mass (kg): ' + '%.4e' % mass_det + '\n')
    f.write('Number of devices: ' + str(n_devices) + '\n')
    f.write('Coincidence level: ' + str(coinc) + '\n')
    f.write('Time window (s): ' + str(window) + '\n')
    f.write('Dark matter masses (GeV): ' + str(m_dms) + '\n')
    f.write('Cross section (cm2): ' + str(sigma0) + '\n')
    f.write('Baseline resolution (keV): ' + str(energy_res) + '\n')
    f.write('Gain: ' + str(det_gain) + '\n')
    f.write('Variable resolution: ' + str(var_threshold) + '\n')
    f.write('ELF model: ' + str(elf_model) + '\n')
    f.write('ELF params: ' + str(elf_params) + '\n') 
    f.close()
    
    
    for t in times:
        for n in coinc:
            run_scan_point(
                nexp,
                t,
                mass_det,
                n_devices,
                n,
                window,
                var_threshold=var_threshold,
                save=True,
                savedir=results_dir,
                m_dms=m_dms,
                sigma0=sigma0,
                elf_target=tm,
                elf_model=elf_model,
                elf_params=elf_params,
            )
        
    return

# ------------------------------------------------------
# ------------------------------------------------------
def main():
    
    t_start = time.time()
    try: 
        results_dir = sys.argv[1]
    except:
        print("Check inputs.\n")
        return 1
    
    sapphire_scan(results_dir)
    t_end = time.time()
    print(f'Full scan took {(t_end - t_start)/60:.2f} minutes.')

    return 0

if __name__ == "__main__":
    main()
