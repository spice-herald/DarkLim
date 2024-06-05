import numpy as np
import matplotlib.pyplot as plt

import darklim

from multihist import Hist1d

##################################################################

efficiency = 1.0
tm = 'He' # target name
energy_res = 0.373e-3 # energy resolution in keV
per_device_threshold = 5 * energy_res # threshold

##################################################################

def run_scan_point(time_elapsed,mass_det,n_devices,coinc,window,save=True):
    
    known_bkg_index = 0
    
    nexp = 10 # number of toys
    m_dms = np.geomspace(0.03, 2, num=10)
    
    ehigh = 10 # keV
    threshold = coinc*per_device_threshold
    
    SE = darklim.sensitivity.SensEst(mass_det, time_elapsed, eff=efficiency, tm=tm)
    SE.reset_sim()
    SE.add_flat_bkgd(1) # flat background of 1 DRU
    SE.add_nfold_lee_bkgd(m=n_devices,n=coinc,w=window)
    
    print('\nRunning with the following settings:')
    print('Mass: {:0.3f} kg; Time: {:0.3f} d => Exposure: {:0.3f} kg-d'.format(SE.m_det,time_elapsed,SE.exposure))
    print('Coincidence: {:d}-fold in {:d} devices; {:0.1f} microsecond window'.format(coinc,n_devices,window/1e-6))
    print('Energy threshold: {:0.3f} keV'.format(threshold))
    
    # run
    m_dm, sig, ul = SE.run_sim_fc(
        known_bkg_index,
        threshold,
        ehigh,
        e_low=1e-6,
        m_dms=m_dms,
        nexp=nexp,
        npts=1000,
        plot_bkgd=False,
        res=energy_res,
        verbose=False,
        sigma0=1e-36
    )

    #print('ULs (evts) = ',ul)
    #print('ULs (xsec) = ',sig)
    
    # save results to txt file
    if save:
        outname = 'HeRALD_FC_{:0.1f}kgd_{:d}device_{:d}fold_{:0.0f}mus.txt'.format(SE.exposure,n_devices,coinc,window/1e-6)
        tot = np.column_stack( (m_dm, sig) )
        np.savetxt(outname,tot,fmt=['%.5e','%0.5e'] ,delimiter=' ')
    
    return
    
def helium_scan():
    
    #exposures = np.array([0.1,1,10]) # kg-d
    exposures = np.array([10]) # kg-d
    mass_det = 0.010 # mass in kg
    times = exposures/mass_det
    print(times)
    
    n_devices = 4
    coinc = np.arange(2,5)
    #coinc = np.array([2])
    window = 100e-6 # s
    for t in times:
        for n in coinc:
            run_scan_point(t,mass_det,n_devices,n,window,save=True)
        
        
    return

# ------------------------------------------------------
# ------------------------------------------------------
def main():
    #try: 
    #    file = sys.argv[1]
    #except:
    #    print("Check inputs.\n")
    #    return 1
    
    helium_scan()
    return 0

if __name__ == "__main__":
         main()