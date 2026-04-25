import numpy as np
from darklim import constants

class Defaults:
    
    def __init__(self):
        
        self.results_dir = './results/'
        self.max_cpus = 12

        self.nexp = 30
        self.t_days = 300.

        self.target = 'Al2O3'
        self.volume_cm3 = (25 * 0.375) / 1000

        self.n_sensors = 1
        self.coincidence = 1
        self.window_s = 1e-6
        self.nsigma = 5
        
        self.baseline_res_eV = 64e-3
        self.e_high_keV = 1.

        self.PD_energy_resolution = np.inf
        self.GaAs_energy_resolution = np.inf
        self.PD_energy_threshold = np.inf
        self.GaAs_energy_threshold = np.inf

        self.masses_GeV = [1e-2, 1e0, 24]
        self.sigma0 = 1e-36

        self.elf_params_NR = {}
        self.elf_params_electron = {'mediator': 'massless', 'kcut': 0, 'method': 'grid', 'withscreening': True, 'suppress_darkelf_output': False}
        self.elf_params_phonon = {'mediator': 'massive', 'suppress_darkelf_output': False, 'dark_photon': False}
        self.elf_params_absorption = {'suppress_darkelf_output': False}

        self.detector = ''
        self.he_gain = 0.
        self.per_device_threshold_keV = 0.

        self.LEE_improvement = 1.
        self.LEE_filename = 'Run57_LEE_templates/combined_spectra_shared57_CRESST_extrap.txt'
