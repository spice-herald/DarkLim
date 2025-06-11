import numpy as np
from darklim import constants

class Defaults:
    
    def __init__(self):
        
        self.results_dir = './results/'
        self.max_cpus = 1

        self.nexp = 50
        self.t_days = 500

        self.target = 'Al2O3'
        self.volume_cm3 = 0.009375

        self.n_sensors = 1
        self.coincidence = 1
        self.window_s = 1e-6
        self.nsigma = 5
        
        self.baseline_res_eV = 64e-3
        self.PD_energy_resolution = np.inf
        self.GaAs_energy_resolution = np.inf
        self.e_high_keV = 1e-3

        self.masses_GeV = [20e-3, 20e-3, 1]

        self.sigma0 = 1e-31

        self.elf_params_NR = {}
        self.elf_params_electron = {'mediator': 'massless', 'kcut': 0, 'method': 'grid', 'withscreening': True, 'suppress_darkelf_output': False}
        self.elf_params_phonon = {'mediator': 'massive', 'suppress_darkelf_output': False, 'dark_photon': False}
        self.elf_params_absorption = {'suppress_darkelf_output': False}
        self.detector = ''
        self.he_gain = 0.
        self.per_device_threshold_keV = 0.

