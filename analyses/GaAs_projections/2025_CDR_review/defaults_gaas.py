import numpy as np
from darklim import constants

class Defaults:
    
    def __init__(self):
        
        self.results_dir = './results/'
        self.max_cpus = 12

        self.nexp = 1000
        self.t_days = 300.
    
        self.target = 'GaAs'
        self.volume_cm3 = (900 * 12) / 1000 * 12

        self.n_sensors = 1
        self.coincidence = 1
        self.window_s = 1e-6

        self.nsigma = np.inf
        
        self.baseline_res_eV = np.inf
        self.PD_energy_resolution = 0.019e-3
        self.GaAs_energy_resolution = 0.164e-3
        self.PD_energy_threshold = np.inf
        self.GaAs_energy_threshold = np.inf
        
        self.masses_GeV = [1e-1, 100, 36]

        self.sigma0 = 1e-43

        self.elf_params_NR = {}
        self.elf_params_electron = {'mediator': 'massless', 'kcut': 0, 'method': 'grid', 'withscreening': True, 'suppress_darkelf_output': False}
        self.elf_params_phonon = {'mediator': 'massive', 'suppress_darkelf_output': False, 'dark_photon': False}

        self.detector = ''
        self.he_gain = 0.
        self.per_device_threshold_keV = 0.

        self.e_high_keV = 100.

        self.LEE_improvement = 1.
        self.LEE_filename = 'backgrounds.txt'

