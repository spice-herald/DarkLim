import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

import darklim

from multihist import Hist1d

##################################################################

efficiency = 1.0
tm = 'He' # target name
energy_res = 0.373e-3 # energy resolution in keV

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
    ax.grid(ls='--',lw=0.3)
    
    if savename is not None:
        plt.savefig(savename+'_acceptance.png',facecolor='white',bbox_inches='tight')
    
    return

def run_scan_point(nexp,time_elapsed,mass_det,n_devices,coinc,window,change_box=True,var_threshold=False,save=True,savedir=None):
    
    if coinc==1: # if coinc is 1, LEE is 'unknown'
        known_bkgs = [0]
    else:
        known_bkgs = [0,1]
    
    #m_dms = np.geomspace(0.005, 2, num=25)
    m_dms = np.concatenate((np.geomspace(0.08, 0.280, num=15),np.array([0.300]),np.geomspace(0.320, 2, num=12)))
    #print(m_dms)
    sigma0 = 1e-36
    
    if change_box:
        ehigh = darklim.sensitivity.edep_to_eobs(darklim.limit.drde_max_q(max(m_dms), tm=tm),he_gain) + 10*np.sqrt(n_devices)*energy_res
    else:
        ehigh = 10 # keV
    
    if var_threshold:
        nsigma = stats.norm.isf(stats.norm.sf(5)**(1/coinc))
    else:
        nsigma = 5
    
    per_device_threshold = nsigma * energy_res # threshold
    threshold = coinc*per_device_threshold
    
    SE = darklim.sensitivity.SensEst(mass_det, time_elapsed, eff=efficiency, tm=tm, gain=he_gain)
    SE.reset_sim()
    SE.add_flat_bkgd(1) # flat background of 1 DRU
    SE.add_nfold_lee_bkgd(m=n_devices,n=coinc,w=window)
    
    print('\nRunning with the following settings:')
    print('Mass: {:0.4f} kg; Time: {:0.3f} d => Exposure: {:0.3f} kg-d'.format(SE.m_det,time_elapsed,SE.exposure))
    print('Coincidence: {:d}-fold in {:d} devices; {:0.1f} microsecond window'.format(coinc,n_devices,window/1e-6))
    print('Energy threshold: {:0.3f} eV; {:0.2f} sigma in each device'.format(threshold*1e3,nsigma))
    print('ROI max: {:0.3f} keV for max DM mass of {:0.3f} GeV.'.format(ehigh,max(m_dms)))
    
    # run
    m_dm, sig, ul, dm_rates, raw_dm_rates, exp_bkg = SE.run_fast_fc_sim(
        known_bkgs,
        threshold,
        ehigh,
        e_low=1e-6, #threshold,
        m_dms=m_dms,
        nexp=nexp,
        npts=int(1e4),
        plot_bkgd=False,
        res=np.sqrt(n_devices)*energy_res,
        verbose=False,
        sigma0=sigma0,
        use_drdefunction=True,
        pltname='ULs_{:0.0f}d_{:d}device_{:d}fold_{:0.0f}mus'.format(time_elapsed,n_devices,coinc,window/1e-6)
        #pltname=None
    )
    
    # save results to txt file
    if save and savedir is not None:
        outname = './{:s}/HeRALD_FC_{:0.0f}d_{:d}device_{:d}fold_{:0.0f}mus.txt'.format(savedir,time_elapsed,n_devices,coinc,window/1e-6)
        tot = np.column_stack( (m_dm, sig, dm_rates/raw_dm_rates, exp_bkg) )
        np.savetxt(outname,tot,fmt=['%.5e','%0.5e','%0.5e','%0.5e'] ,delimiter=' ')
    
    # plot acceptance and DM evt rate:
    savename = \
    './{:s}/dmrate_{:0.0f}d_{:d}device_{:d}fold_{:0.0f}mus'.format(savedir,time_elapsed,n_devices,coinc,window/1e-6)
    plot_dm_rates(m_dms,dm_rates,raw_dm_rates,sigma0,savename=savename)
    
    return
    
def helium_scan(results_dir):
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    nexp = 200 # number of toys
    
    var_threshold = True # vary 5sigma requirement based on coinc level
    change_box = True # vary upper limit of energy ROI based on DM mass
    
    #times = np.array([1,2,5,10,20,50,75,100,200,500]) # d
    #times = np.linspace(1,250,num=20, endpoint=True) # d
    
    times = np.concatenate( (np.array([1,2,5,10]), np.linspace(15,400,num=20, endpoint=True), np.array([30,183,365])) ) #d
    times = np.sort(times)
    #times = np.array([30,183,365]) # 1, 6, and 12 months
    
    #mass_det = 8.*0.14*1e-3 # mass in kg, = 8cc * 0.14g/cc
    
    mass_det = 10*1e-3 # mass in kg
    
    exposures = times*mass_det
    
    n_devices = 4
    coinc = np.arange(2,5)
    #coinc = np.array([1])
    window = 100e-6 # s
    
    for t in times:
        for n in coinc:
            run_scan_point(
                nexp,
                t,
                mass_det,
                n_devices,
                n,
                window,
                change_box=change_box,
                var_threshold=var_threshold,
                save=True,
                savedir=results_dir
            )
        
    return

# ------------------------------------------------------
# ------------------------------------------------------
def main():
    
    try: 
        results_dir = sys.argv[1]
    except:
        print("Check inputs.\n")
        return 1
    
    helium_scan(results_dir)
    return 0

if __name__ == "__main__":
         main()