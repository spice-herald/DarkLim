import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import darklim

from multihist import Hist1d

##################################################################

efficiency = 1.0
tm = 'He' # target name
energy_res = 0.373e-3 # energy resolution in keV
per_device_threshold = 5 * energy_res # threshold
he_gain = 0.15
##################################################################

def run_scan_point(time_elapsed,mass_det,n_devices,coinc,window,save=True,savedir=None):
    
    known_bkgs = [0,1]
    
    nexp = 250 # number of toys
    m_dms = np.geomspace(0.005, 1, num=20)
    
    ehigh = 10 # keV
    threshold = coinc*per_device_threshold
    
    SE = darklim.sensitivity.SensEst(mass_det, time_elapsed, eff=efficiency, tm=tm, gain=he_gain)
    SE.reset_sim()
    SE.add_flat_bkgd(1) # flat background of 1 DRU
    SE.add_nfold_lee_bkgd(m=n_devices,n=coinc,w=window)
    
    print('\nRunning with the following settings:')
    print('Mass: {:0.4f} kg; Time: {:0.3f} d => Exposure: {:0.3f} kg-d'.format(SE.m_det,time_elapsed,SE.exposure))
    print('Coincidence: {:d}-fold in {:d} devices; {:0.1f} microsecond window'.format(coinc,n_devices,window/1e-6))
    print('Energy threshold: {:0.3f} eV'.format(threshold*1e3))
    
    # run
    m_dm, sig, ul = SE.run_fast_fc_sim( #SE.run_sim_fc(
        known_bkgs,
        threshold,
        ehigh,
        e_low=1e-6, #threshold,
        m_dms=m_dms,
        nexp=nexp,
        npts=int(1e4),
        plot_bkgd=True,
        res=np.sqrt(n_devices)*energy_res,
        verbose=False,
        sigma0=1e-36,
        use_drdefunction=True,
        pltname='ULs_{:0.0f}d_{:d}device_{:d}fold_{:0.0f}mus'.format(time_elapsed,n_devices,coinc,window/1e-6)
        #pltname=None
    )

    #print('ULs (evts) = ',ul)
    #print('ULs (xsec) = ',sig)
    
    # save results to txt file
    if save and savedir is not None:
        outname = './{:s}/HeRALD_FC_{:0.0f}d_{:d}device_{:d}fold_{:0.0f}mus.txt'.format(savedir,time_elapsed,n_devices,coinc,window/1e-6)
        tot = np.column_stack( (m_dm, sig) )
        np.savetxt(outname,tot,fmt=['%.5e','%0.5e'] ,delimiter=' ')
    
    return
    
def helium_scan(results_dir):
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    #exposures = np.array([0.1,1,10]) # kg-d
    #exposures = np.array([1]) # kg-d
    
    times = np.array([10])#1,10,50,100,200,500]) # d
    mass_det = 8.*0.14*1e-3 # mass in kg, = 8cc * 0.14g/cc
    exposures = times*mass_det
    #print(times)
    
    n_devices = 4
    coinc = np.arange(2,5)
    #coinc = np.array([2])
    window = 100e-6 # s
    
    for t in times:
        for n in coinc:
            run_scan_point(t,mass_det,n_devices,n,window,save=True,savedir=results_dir)
        
        
    return

# ------------------------------------------------------
# ------------------------------------------------------
def main():
    results_dir = sys.argv[1]
    #try: 
    #    file = sys.argv[1]
    #except:
    #    print("Check inputs.\n")
    #    return 1
    
    helium_scan(results_dir)
    return 0

if __name__ == "__main__":
         main()