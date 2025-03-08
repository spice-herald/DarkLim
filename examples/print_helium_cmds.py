import os
import sys
import numpy as np


def print_helium_cmds():

    t_days = 10
    
    #detectors = ['4-1cm2']#,'4-0.5cm2','16-0.5cm2','24-0.5cm2']

    detectors = ['24-0.5cm2'] #,'4-0.5cm2','16-0.5cm2','24-0.5cm2']
    
    he_gains = [0.15, 0.30]
    per_device_thresholds = np.arange(0.2,1.2,0.2) * 1e-3 # keV
    energy_resolutions = [0.200]
    windows = [100e-6] # seconds

    for d in detectors:

        #per_device_thresholds = np.arange(0.2,1.2,0.2) * 1e-3 # keV
        per_device_thresholds = np.arange(0.4,1.2,0.2) * 1e-3 # keV
        
        if d=='4-1cm2': # shovel-ready
            vol = 7.65
            # for this case, pile up rate below 0.6 eV threshold so high
            # the code can't even run... too many bkg evts to generate
            per_device_thresholds = np.arange(0.8,1.2,0.2) * 1e-3 # keV
            windows = [100e-6]
            
        if d=='16-0.5cm2': # same mass as shovel-ready
            vol = 7.65
            windows = [100e-6]
            
        if d=='4-0.5cm2': # lil' guy
            vol = 0.92
            windows = [100e-6]
            
        if d=='24-0.5cm2':
            vol = 102
            windows = [100e-6]
        
        for he_gain in he_gains:
            for thres in per_device_thresholds:
                for energy_res_eV in energy_resolutions:
                    for w in windows:

                        print('')
                        print( "python helium_fc_scan.py --detector={d:s} --volume_cm3={vol:0.2f} --he_gain={he_gain:0.2f} --per_device_threshold_keV={thres:0.2e} --baseline_res_eV={energy_res_eV:0.3f} --window_s={w:0.0e} --t_days={t_days:0.1f} --results_dir='./results-{d:s}-{vol:0.2f}cm3-{he_gain:0.2f}gain-{thres:0.2e}keV-{energy_res_eV:0.3f}eV-{w:0.0e}s-{t_days:0.1f}day'".format(d=d, vol=vol, he_gain=he_gain, thres=thres, energy_res_eV=energy_res_eV, w=w, t_days=t_days) )
                
    
    return
    

# ------------------------------------------------------
# ------------------------------------------------------

if __name__ == "__main__":
    print_helium_cmds()
    
