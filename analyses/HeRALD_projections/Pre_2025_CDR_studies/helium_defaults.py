import numpy as np
from darklim import constants

class Defaults:
    
    def __init__(self):
        
        #self.results_dir = './results/'
        self.results_dir = '/global/cfs/cdirs/lz/users/haselsco/TESSERACT_Limits/DarkLim_vetriupdate/examples/results/'
        
        #self.max_cpus = 1
        #self.masses_GeV = [0.010,0.010,1] # specifies min, max, number of masses
        #self.masses_GeV = [1,1,1] # specifies min, max, number of masses
        
        self.max_cpus = 48
        self.masses_GeV = [0.005, 10, 48] # specifies min, max, number of masses
        
        self.nexp = 250
        self.t_days = 1

        self.detector = '4-1cm2'

        self.he_gain = 0.15
        
        self.target = 'He'
        self.volume_cm3 = 102. #  = 14.3 grams

        self.n_sensors = 4
        self.coincidence = 4
        self.window_s = 100e-6
        self.nsigma = 5

        self.per_device_threshold_keV = 1e-3
        
        self.baseline_res_eV = 0.200 # eV

        #self.masses_GeV = [0.08, 2, 3] # specifies min, max, number of masses
        #self.masses_GeV = [3e-3, 2e1, 24]
        
        #self.masses_GeV = [1, 1, 1] # specifies min, max, number of masses

        self.sigma0 = 1e-36

        self.elf_params_NR = {}
        self.elf_params_electron = {'mediator': 'massless', 'kcut': 0, 'method': 'grid', 'withscreening': True, 'suppress_darkelf_output': False}
        self.elf_params_phonon = {'mediator': 'massive', 'suppress_darkelf_output': False, 'dark_photon': False}
