import argparse
import numpy as np
import os
import scipy.stats as stats
import defaults
from darklim import constants



def mybool(s):
    '''
    Convert string to boolean. String must be
    T, TRUE, F, or FALSE.
    '''
    
    if s.upper() in ['T', 'TRUE']:
        return True
    elif s.upper() in ['F', 'FALSE']:
        return False
    else:
        raise ValueError(f'{s} is not a boolean')

        
        
def convert_scan_parameters(args):
    '''
    Convert CLI parameters that ArgumentParser couldn't handle
    into a usable format (e.g., bool, dict), and do some calculations
    (i.e. volume -> mass and exposure). Returns a modified version
    of the input Namespace.
    '''
    
    df = defaults.Defaults()
    
    if args.results_dir[-1] != '/':
        args.results_dir += '/'
    
    if args.coincidence > 1:
        args.nsigma = stats.norm.isf(stats.norm.sf(args.nsigma)**(1/args.coincidence))

    if args.target == 'Al2O3':
        density_gcm3 = constants.Al2O3_density

    args.target_mass_kg = args.volume_cm3 * density_gcm3 * 1e-3
    args.exposure_kgd = args.target_mass_kg * args.t_days

    m_min, m_max, n_masses = args.masses_GeV
    args.masses_GeV = np.geomspace(m_min, m_max, int(n_masses))

    if args.elf is None or args.elf[0] == 'NR':
        args.elf_model = None
        args.elf_params = df.elf_params_NR
    elif args.elf[0] == 'electron':
        args.elf_model = 'electron'
        args.elf_params = df.elf_params_electron
    elif args.elf[0] == 'phonon':
        args.elf_model = 'phonon'
        args.elf_params = df.elf_params_phonon

    if args.elf_model is not None:
        for i, s in enumerate(args.elf[1:]):
            if s in args.elf_params:
                if s == 'kcut':
                    args.elf_params[s] = int(args.elf[i+2])
                elif s in ['withscreening', 'suppress_darkelf_output', 'dark_photon']:
                    args.elf_params[s] = mybool(args.elf[i+2])
                else:
                    args.elf_params[s] = args.elf[i+2]

    return args
    


def get_scan_parameters():
    '''
    Read command-line string and extract parameters for an OI calculation.
    Note: argv should not include the executing file name, so it should
    typically be sys.argv[1:].
    '''

    df = defaults.Defaults()

    parser = argparse.ArgumentParser(description="Generate a single limit")

    parser.add_argument('--results_dir', type=str, default=df.results_dir,
                        help='Output directory')

    parser.add_argument('--nexp', type=int, default=df.nexp,
                        help='Number of pseudoexperiments')

    parser.add_argument('--t_days', type=float, default=df.t_days,
                        help='Exposure in days')

    parser.add_argument('--target', type=str, default=df.target,
                        choices=['Al2O3'],
                        help='Target material')

    parser.add_argument('--volume_cm3', type=float, default=df.volume_cm3,
                        help='Target volume in cm3')

    parser.add_argument('--n_sensors', type=int, default=df.n_sensors,
                        help='Number of sensors')

    parser.add_argument('--coincidence', type=int, default=df.coincidence,
                        help='Coincidence level between sensors')

    parser.add_argument('--window_s', type=float, default=df.window_s,
                        help='Coincidence window (seconds)')

    parser.add_argument('--nsigma', type=float, default=df.nsigma,
                        help=('Number of sigma for detection above baseline ' + 
                              'in one sensor. For coincidence in multiple ' + 
                              'sensors, we lower the threshold per sensor ' +
                              'so that the total threshold corresponds ' +
                              'to the equivalent p-value from nsigma.'))

    parser.add_argument('--baseline_res_eV', type=float, default=df.baseline_res_eV,
                        help='Baseline energy resolution (eV)')

    parser.add_argument('--masses_GeV', type=float, nargs=3, default=df.masses_GeV,
                        help=('DM masses in GeV. Three arguments: minimum ' + 
                              'mass, maximum mass, and number of masses, ' + 
                              'logarithmically spaced'))

    parser.add_argument('--sigma0', type=float, default=df.sigma0,
                        help='Sigma0 [cm2]')

    parser.add_argument('--elf', type=str, nargs='*', default=None,
                        help=('DarkELF Model and Parameters. First argument ' + 
                             'must be either NR, electron, or phonon. ' + 
                             'Remaining arguments should be the ELF parameters ' +
                             'in the format NAME VALUE NAME VALUE ... ' + 
                             'For the list of possible parameters, ' + 
                             'see _sens_est.py. You dont need to set all ' + 
                             'of them; if you dont, they will use their ' + 
                             'default values.'))

    args = parser.parse_args()
    
    args = convert_scan_parameters(args)
    
    return args
    
