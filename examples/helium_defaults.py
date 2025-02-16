import numpy as np
from darklim import constants

class Defaults:
    
    def __init__(self):
        
        #self.results_dir = './results/'
        self.results_dir = '/global/cfs/cdirs/lz/users/haselsco/TESSERACT_Limits/DarkLim_vetriupdate/examples/results/'
        self.max_cpus = 12

        self.nexp = 10
        self.t_days = 1 / 3600 / 24.

        self.target = 'He'
        self.volume_cm3 = 9/0.14 # made to make LHe mass = 9 grams

        self.n_sensors = 1
        self.coincidence = 1
        self.window_s = 100e-6
        self.nsigma = 5
        
        self.baseline_res_eV = 0.373 # eV
        
        #self.masses_GeV = [3e-3, 2e1, 24]
        self.masses_GeV = [0.08, 2, 12] # specifies min, max, number of masses
        #self.masses_GeV = [1, 1, 1] # specifies min, max, number of masses

        self.sigma0 = 1e-36

        self.elf_params_NR = {}
        self.elf_params_electron = {'mediator': 'massless', 'kcut': 0, 'method': 'grid', 'withscreening': True, 'suppress_darkelf_output': False}
        self.elf_params_phonon = {'mediator': 'massive', 'suppress_darkelf_output': False, 'dark_photon': False}
