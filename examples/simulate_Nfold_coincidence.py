import numpy as np
import random
from multiprocessing import Pool
import pandas as pd
from scipy.interpolate import interp1d
from scipy.special import gamma
import scipy.stats as stats
from scipy.stats import binom

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'

import sys
import darklim

import qetpy
import detprocess
import h5py
import time
import pickle
from time import localtime, strftime
import glob


lee_file = 'Run57_LEE_templates/combined_spectra_shared57_CRESST_extrap.txt'
lee_scale = 1. / 400.

def run57_lee_bkgd():
    template = np.loadtxt(lee_file, skiprows=1)
    interp_func = interp1d(template[:,0],template[:,1],
                            bounds_error=False,
                            #fill_value='extrapolate',
                            fill_value=0.,
                            assume_sorted=True)

    return lambda x: (interp_func(x) / 86400. * lee_scale)



def get_closest_time(trigger_times, t):
    """
    Find the closest time in trigger_times to t and its index.
    """
    # Find the index in trigger_times closest to t (trigger_times is sorted)
    idx = np.searchsorted(trigger_times, t)
    if idx == 0:
        closest_idx = 0
    elif idx == len(trigger_times):
        closest_idx = len(trigger_times) - 1
    else:
        if abs(trigger_times[idx] - t) < abs(trigger_times[idx - 1] - t):
            closest_idx = idx
        else:
            closest_idx = idx - 1

    try:
        _ = trigger_times[closest_idx]
    except IndexError:
        print(f"IndexError: closest_idx = {closest_idx}, len(trigger_times) = {len(trigger_times)}")
        sys.exit(1)
    
    # Return the closest time and its index
    return trigger_times[closest_idx], closest_idx



### Set the number of desired channels, single-pixel energy resolution, and OF trigger strength in your simulation
n_channels = 4
energy_resolution = 0.0074
n_sigma = 7.0


### Load template
template = detprocess.Template(verbose=True)
template.load_hdf5('/home/vvelan/OF_Development/detprocess/scripts/processing_configs/r47_072324_filter_dm_5.hdf5')

# Load shared 2x1 template to get the energy scale right, but we only use the left-channel template shape and PSD
channel = 'Mv3025pcBigFinsLeft|Mv3025pcBigFinsRight'

templates_td, t_sec_template, metadata = template.get_template(channel, return_metadata=True, tag='shared')
templates_td = np.reshape(templates_td[0,0], (1,1,templates_td.shape[-1]))
_, _, f_frequencies = templates_td.shape
fs = metadata['sample_rate']
pretrig = metadata['nb_pretrigger_samples']

csd, freqs_csd, _ = template.get_csd(channel, fold=False, return_metadata=True, tag='default')
csd = np.reshape(csd[0,0], (1,1,csd.shape[-1]))



### Convert OF to multichannel format

# Rescale template so that the energy resolution per channel is some assumed value
base_energy_resolution = 0.3977 # eV; this is the resolution of the template loaded above

template_OF = np.zeros((n_channels, n_channels, f_frequencies))
csd_OF = np.zeros((n_channels, n_channels, f_frequencies))
for i in range(n_channels):
    template_OF[i, i, :] = np.copy(templates_td)
    csd_OF[i, i, :] = np.copy(csd) * (energy_resolution / base_energy_resolution)**2



### Simulation parameters

# Single-pixel event rate in Hz for the simulation
# Note: this should be as high as possible without inducing pileup to make the simulation run faster
# Note: if the integrated LEE rate is higher than this, we will use the integrated LEE rate
rate_Hz = 50

# Minimum energy to simulate a pulse (keV)
E_cutoff = 0.18e-3

# Calculate the true LEE rate integrated above E_cutoff
E_arr = np.geomspace(E_cutoff, 100e-3, 250)
dRdE_arr = run57_lee_bkgd()(E_arr)
R_LEE = np.trapz(dRdE_arr, E_arr)
if R_LEE > rate_Hz:
    rate_Hz = R_LEE

print(f'LEE background: {R_LEE:.2e} Hz per pixel above {E_cutoff*1e3:.2f} eV')
print(f'Simulated event rate: {rate_Hz:.2e} Hz per pixel above {E_cutoff*1e3:.2f} eV')

# Amount of time to simulate
# Note: this time is referenced to rate_Hz, not to rate_LEE
full_data_length_hours = 6

 # Chunk size
sim_trace_length_s = 10
sim_trace_length = 10 * fs

# Split up the simulation into chunks, which are then split between cores
n_cores = 12
n_simulations = int(np.floor(full_data_length_hours * 3600 / sim_trace_length_s))
n_traces_per_sim = int(sim_trace_length / f_frequencies)
print(f'Running {n_simulations} simulations in parallel with {n_cores} cores')




# Results will be saved in a pickle file 'my_simulation_3fold_{run_id}_{seed_1}_{seed_2}.pkl'
seed_1 = int(sys.argv[1])
seed_2 = int(sys.argv[2])
run_id = int(sys.argv[3])

def simulate_and_collect(sim_number):
    """
    Simulate a single run of the experiment (typically 10 seconds) and collect the results.
    """

    rng = np.random.default_rng(seed_1 + sim_number * 100)

    # Simulate pulses in the two pixels
    expected_events = int(rate_Hz * sim_trace_length_s * 2)  # Overestimate to be safe

    # SensEst object to generate the LEE background
    SE = darklim.sensitivity.SensEst(1, 1/86400 * expected_events / R_LEE, tm='GaAs', eff=1., gain=1., seed=seed_2 + sim_number * 100)
    SE.reset_sim()
    SE.add_run57_lee_bkgd(fin='Run57_LEE_templates/combined_spectra_shared57.txt', scale_by=lee_scale)

    # Simulate noise in the N pixels
    simulated_waveform_joint = qetpy.gen_noise(csd_OF, fs, n_traces_per_sim, rng=rng)
    
    # Reshape so that each pixel is a separate channel
    simulated_waveform_td = np.zeros((n_channels, f_frequencies * n_traces_per_sim))
    for i in range(n_channels):
        for j in range(n_traces_per_sim):
            simulated_waveform_td[i, j * f_frequencies:(j + 1) * f_frequencies] = simulated_waveform_joint[j, i, :]
    
    # Simulate arrival times and energies in each pixel
    for i in range(n_channels):
        arrival_times = np.cumsum(rng.exponential(1.0 / rate_Hz, expected_events)) * fs
        arrival_times = arrival_times[arrival_times < (sim_trace_length - f_frequencies)].astype(int)
        energies_eV = SE.generate_background(100e-3, e_low=E_cutoff) * 1000
    
        # Add pulses to waveform
        for t, atime in enumerate(arrival_times):
            try:
                simulated_waveform_td[i, atime : atime + f_frequencies] += energies_eV[t] * template_OF[i,i]
            except IndexError:
                print(f"IndexError: arrival_times = {len(arrival_times)}, atime = {atime}, t = {t}, expected_events = {expected_events}, simulated_waveform_td.shape = {simulated_waveform_td.shape}")
                sys.exit(1)

    # Process the data with an NxN trigger
    chan_name = '|'.join(f'MyTrigger{i}' for i in range(n_channels))
    oftrigger = detprocess.OptimumFilterTrigger(chan_name, fs, template_OF, csd_OF, pretrigger_samples=pretrig)
    oftrigger.update_trace(simulated_waveform_td)
    dynamic_threshold_function = lambda amp: max(100, 2 * 78 * np.log(amp / 13))
    oftrigger.find_triggers(n_sigma, dynamic=True, dynamic_threshold_function=dynamic_threshold_function)

    # Extract trigger amplitudes and times from the NxN trigger
    trigger_times = oftrigger.get_trigger_data()[chan_name]['trigger_index']
    trigger_delta_chi2 = oftrigger.get_trigger_data()[chan_name]['trigger_delta_chi2']
    n_triggers = len(trigger_times)

    E_matrix = np.zeros((n_channels, n_triggers))
    for i in range(n_channels):
        E_matrix[i, :] = oftrigger.get_trigger_data()[chan_name][f'trigger_amplitude_{i}']

    # For each trigger, compare chi2 values from the three triggers:
    if sim_number % n_cores == 0:
        print(f'In simulation {sim_number}, starting the loop over {n_triggers} times. Currently', strftime("%m-%d %H:%M:%S", localtime()))

    # Reprocess the data with a 1x1 trigger on each pixel
    oftriggers = []
    trigger_times_list = []
    trigger_delta_chi2_list = []

    for i in range(n_channels):
        trig_name = f'MyTrigger{i}'
        oftrig = detprocess.OptimumFilterTrigger(
            trig_name,
            fs,
            template_OF[i, i],
            csd_OF[i, i],
            pretrigger_samples=pretrig
        )
        oftrig.update_trace(simulated_waveform_td[i])
        oftrig.find_triggers(n_sigma, dynamic=True, dynamic_threshold_function=dynamic_threshold_function)
        trigger_times_list.append(oftrig.get_trigger_data()[trig_name]['trigger_index'])
        trigger_delta_chi2_list.append(oftrig.get_trigger_data()[trig_name]['trigger_delta_chi2'])
        oftriggers.append(oftrig)

    # Calculate the chi2 difference between the NxN and sum of 1x1 triggers
    delta_delta = np.zeros(n_triggers)
    associated_trigger_times = np.zeros((n_channels, n_triggers), dtype=int)

    # Loop over the NxN triggers and find the closest 1x1 trigger times
    for t, trigger_time in enumerate(trigger_times):
        delta_chi2_NxN = trigger_delta_chi2[t]
        dchi2_1x1_sum = 0
        for i in range(n_channels):
            times_ch = trigger_times_list[i]
            dchi2s_ch = trigger_delta_chi2_list[i]
            if len(times_ch) == 0:
                dchi2 = 0
            else:
                this_t, idx = get_closest_time(times_ch, trigger_time)
                dchi2 = dchi2s_ch[idx]
                associated_trigger_times[i, t] = this_t
            dchi2_1x1_sum += dchi2
        delta_delta[t] = dchi2_1x1_sum - delta_chi2_NxN

    # Delete the waveform object to free up memory
    del simulated_waveform_td
    
    return E_matrix, trigger_times, delta_delta, associated_trigger_times, [sim_number] * n_triggers




##################################################################
# Run the simulation in parallel
##################################################################

results = []
for batch_start in range(0, n_simulations, n_cores):
    batch_indices = list(range(batch_start, min(batch_start + n_cores, n_simulations)))
    with Pool(n_cores) as pool:
        batch_results = pool.map(simulate_and_collect, batch_indices)
    results.extend(batch_results)

with open(f'my_simulation_4fold_{run_id:02d}_{seed_1}_{seed_2}.pkl', 'wb') as f:
    pickle.dump(results, f)

