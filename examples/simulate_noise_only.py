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
font_manager.fontManager.addfont('/home/vvelan/DarkLim/examples/styles/Times_New_Roman.ttf')

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.style.use('/home/vvelan/DarkLim/examples/styles/Style.mplstyle')

import sys
import darklim
sys.path.insert(0, '/home/vvelan/DarkELF/')
import darkelf

import qetpy
import detprocess
import h5py
import time
import pickle
from time import localtime, strftime


##############################

lee_input_file = 'Run57_LEE_templates/combined_spectra_shared57_CRESST_extrap.txt'
lee_scale = 1 / 1.

def run57_lee_bkgd():
    template = np.loadtxt(lee_input_file, skiprows=1)
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


##############################


### Load input arguments
seed_1 = int(sys.argv[1])
seed_2 = int(sys.argv[2])
run_id = int(sys.argv[3])

### Load template
template = detprocess.Template(verbose=True)
template.load_hdf5('/home/vvelan/OF_Development/detprocess/scripts/processing_configs/r47_072324_filter_dm_5.hdf5')

# Load shared 2x1 template to get the energy scale right, but we only use the left channel template shape and PSD
channel = 'Mv3025pcBigFinsLeft|Mv3025pcBigFinsRight'

templates_td, t_sec_template, metadata = template.get_template(channel, return_metadata=True, tag='shared')
templates_td = np.reshape(templates_td[0,0], (1,1,templates_td.shape[-1]))
_, _, f_frequencies = templates_td.shape
fs = metadata['sample_rate']
pretrig = metadata['nb_pretrigger_samples']

csd, freqs_csd, _ = template.get_csd(channel, fold=False, return_metadata=True, tag='default')
csd = np.reshape(csd[0,0], (1,1,csd.shape[-1]))

### Background parameters
rate_Hz_GaAs = 50  # Event rate in Hz (for the simulation--this should be as high as possible without inducing pileup)
rate_Hz_PD = 50  # Event rate in Hz (for the simulation--this should be as high as possible without inducing pileup)

### Convert to multichannel format

# Rescale template so that the energy resolution per channel is some assumed value
base_energy_resolution = 0.3977  # eV; this is the resolution of the 1x1 shared template from Run 57
GaAs_energy_resolution = 0.138
PD_energy_resolution = 0.138
n_sigma = 5.0  # Number of sigma for the trigger

template_OF = np.zeros((3, 3, f_frequencies))
template_OF[0, 0, :] = np.copy(templates_td)
template_OF[1, 1, :] = np.copy(templates_td) 
template_OF[2, 2, :] = np.copy(templates_td)

csd_OF = np.zeros((3, 3, f_frequencies))
csd_OF[0, 0, :] = np.copy(csd) * (GaAs_energy_resolution / base_energy_resolution)**2
csd_OF[1, 1, :] = np.copy(csd) * (PD_energy_resolution / base_energy_resolution)**2
csd_OF[2, 2, :] = np.copy(csd) * (PD_energy_resolution / base_energy_resolution)**2

##############################


### Set up the simulation parameters
full_data_length_hours = 1
n_cores = 12

sim_trace_length = 1_250_000 * 10
sim_trace_length_s = sim_trace_length / fs  # Convert to seconds

n_simulations = int(np.floor(full_data_length_hours * 3600 / sim_trace_length_s))
n_traces_per_sim = int(sim_trace_length / f_frequencies)
print(f'Running {n_simulations} simulations in parallel with {n_cores} cores')


def simulate_and_collect(sim_number):
    """
    Simulate a single run of the experiment (typically 10 seconds) and collect the results.
    """

    rng = np.random.default_rng(seed_1 + sim_number * 100)

    # Simulate noise in the two pixels
    simulated_waveform_joint = qetpy.gen_noise(csd_OF, fs, n_traces_per_sim, rng=rng)
    simulated_waveform_pix0_td = np.zeros(f_frequencies * n_traces_per_sim)
    simulated_waveform_pix1_td = np.zeros(f_frequencies * n_traces_per_sim)
    simulated_waveform_pix2_td = np.zeros(f_frequencies * n_traces_per_sim)
    for trace_i in range(n_traces_per_sim):
        simulated_waveform_pix0_td[trace_i * f_frequencies:(trace_i + 1) * f_frequencies] = simulated_waveform_joint[trace_i, 0, :]
        simulated_waveform_pix1_td[trace_i * f_frequencies:(trace_i + 1) * f_frequencies] = simulated_waveform_joint[trace_i, 1, :]
        simulated_waveform_pix2_td[trace_i * f_frequencies:(trace_i + 1) * f_frequencies] = simulated_waveform_joint[trace_i, 2, :]

    # Convert waveform, template, and CSD into OF format
    simulated_waveform_td = np.concatenate([[simulated_waveform_pix0_td], [simulated_waveform_pix1_td], [simulated_waveform_pix2_td]])

    # Process the data with a 3x3 trigger
    oftrigger = detprocess.OptimumFilterTrigger('MyTrigger0|MyTrigger1|MyTrigger2', fs, template_OF, csd_OF, pretrigger_samples=pretrig)
    oftrigger.update_trace(simulated_waveform_td)
    dynamic_threshold_function = lambda amp: max(100, 2 * 78 * np.log(amp / 13))
    oftrigger.find_triggers(n_sigma, dynamic=True, dynamic_threshold_function=dynamic_threshold_function)

    # Extract trigger amplitudes and times from the 3x3 trigger
    E0_arr = oftrigger.get_trigger_data()['MyTrigger0|MyTrigger1|MyTrigger2']['trigger_amplitude_0']
    E1_arr = oftrigger.get_trigger_data()['MyTrigger0|MyTrigger1|MyTrigger2']['trigger_amplitude_1']
    E2_arr = oftrigger.get_trigger_data()['MyTrigger0|MyTrigger1|MyTrigger2']['trigger_amplitude_2']
    trigger_times = oftrigger.get_trigger_data()['MyTrigger0|MyTrigger1|MyTrigger2']['trigger_index']
    trigger_delta_chi2 = oftrigger.get_trigger_data()['MyTrigger0|MyTrigger1|MyTrigger2']['trigger_delta_chi2']

    filtered_trace_0, filtered_trace_1, filtered_trace_2 = oftrigger.get_filtered_trace()
    delta_chi2_trace = oftrigger.get_filtered_delta_chi2()
    n_triggers = len(E0_arr)

    # Reprocess the data with a 1x1 trigger on each pixel
    oftrigger_a = detprocess.OptimumFilterTrigger('MyTrigger0', fs, template_OF[0,0], csd_OF[0,0], pretrigger_samples=pretrig)
    oftrigger_a.update_trace(simulated_waveform_td[0])
    oftrigger_a.find_triggers(n_sigma, dynamic=True, dynamic_threshold_function=dynamic_threshold_function)
    trigger_times_a = oftrigger_a.get_trigger_data()['MyTrigger0']['trigger_index']
    trigger_delta_chi2_a = oftrigger_a.get_trigger_data()['MyTrigger0']['trigger_delta_chi2']

    oftrigger_b = detprocess.OptimumFilterTrigger('MyTrigger1', fs, template_OF[1,1], csd_OF[1,1], pretrigger_samples=pretrig)
    oftrigger_b.update_trace(simulated_waveform_td[1])
    oftrigger_b.find_triggers(n_sigma, dynamic=True, dynamic_threshold_function=dynamic_threshold_function)
    trigger_times_b = oftrigger_b.get_trigger_data()['MyTrigger1']['trigger_index']
    trigger_delta_chi2_b = oftrigger_b.get_trigger_data()['MyTrigger1']['trigger_delta_chi2']

    oftrigger_c = detprocess.OptimumFilterTrigger('MyTrigger2', fs, template_OF[2,2], csd_OF[2,2], pretrigger_samples=pretrig)
    oftrigger_c.update_trace(simulated_waveform_td[2])
    oftrigger_c.find_triggers(n_sigma, dynamic=True, dynamic_threshold_function=dynamic_threshold_function)
    trigger_times_c = oftrigger_c.get_trigger_data()['MyTrigger2']['trigger_index']
    trigger_delta_chi2_c = oftrigger_c.get_trigger_data()['MyTrigger2']['trigger_delta_chi2']

    # For each trigger, compare chi2 values from the three triggers:
    if sim_number % n_cores == 0:
        print(f'In simulation {sim_number}, starting the loop over {n_triggers} times. Currently', strftime("%m-%d %H:%M:%S", localtime()))

    delta_delta = np.zeros(n_triggers)
    delta_chi2_1x1_a = np.zeros(n_triggers)
    delta_chi2_1x1_b = np.zeros(n_triggers)
    delta_chi2_1x1_c = np.zeros(n_triggers)

    for i, trigger_time in enumerate(trigger_times):

        # Chi2 from the 3x3 trigger
        delta_chi2_3x3 = trigger_delta_chi2[i]

        # Find the pulse in pixel 0 that is closest to the trigger time
        if len(trigger_times_a) == 0:
            dchi2_1x1_a = 0
        else:
            time_a, idx_a = get_closest_time(trigger_times_a, trigger_time)
            dchi2_1x1_a = trigger_delta_chi2_a[idx_a]

        # Find the pulse in pixel 1 that is closest to the trigger time
        if len(trigger_times_b) == 0:
            dchi2_1x1_b = 0
        else:
            time_b, idx_b = get_closest_time(trigger_times_b, trigger_time)
            dchi2_1x1_b = trigger_delta_chi2_b[idx_b]

        # Find the pulse in pixel 2 that is closest to the trigger time
        if len(trigger_times_c) == 0:
            dchi2_1x1_c = 0
        else:
            time_c, idx_c = get_closest_time(trigger_times_c, trigger_time)
            dchi2_1x1_c = trigger_delta_chi2_c[idx_c]

        # Compute the delta delta chi2
        delta_delta[i] = (dchi2_1x1_a + dchi2_1x1_b + dchi2_1x1_c) - delta_chi2_3x3
        delta_chi2_1x1_a[i] = dchi2_1x1_a
        delta_chi2_1x1_b[i] = dchi2_1x1_b
        delta_chi2_1x1_c[i] = dchi2_1x1_c

    detected = (np.array(E0_arr) > (n_sigma * GaAs_energy_resolution)) * \
            (np.array(E1_arr) > (n_sigma * PD_energy_resolution)) * \
            (np.array(E2_arr) > (n_sigma * PD_energy_resolution))
    waveforms_detected = []
    delta_chi2_detected = []

    idxs = np.where(detected)[0]
    # idxs = np.append(idxs, rng.uniform(0, len(detected), 10).astype(int))  # Add some random indices to the detected ones for testing
    times_recorded = []
    for i in np.where(detected)[0]:
    # for i in idxs:
        t = trigger_times[i]

        w0 = np.copy(filtered_trace_0[t - pretrig:t + pretrig])
        w1 = np.copy(filtered_trace_1[t - pretrig:t + pretrig])
        w2 = np.copy(filtered_trace_2[t - pretrig:t + pretrig])
        w = np.array([w0, w1, w2])
        waveforms_detected.append(w)

        delta_chi2_detected.append((delta_chi2_trace[t - pretrig:t + pretrig], trigger_delta_chi2[i], delta_chi2_1x1_a[i], delta_chi2_1x1_b[i], delta_chi2_1x1_c[i]))
        
        times_recorded.append(t)

    del simulated_waveform_td

    return list(E0_arr), list(E1_arr), list(E2_arr), [sim_number] * n_triggers, list(trigger_times), delta_delta, detected, waveforms_detected, delta_chi2_detected, times_recorded

# with Pool(n_cores) as pool:
#     results = pool.map(simulate_and_collect, range(n_simulations))

results = []
for batch_start in range(0, n_simulations, n_cores):
    batch_indices = list(range(batch_start, min(batch_start + n_cores, n_simulations)))
    with Pool(n_cores) as pool:
        batch_results = pool.map(simulate_and_collect, batch_indices)
    results.extend(batch_results)



with open(f'my_simulation_3fold_{run_id:02d}_{seed_1}_{seed_2}.pkl', 'wb') as f:
    pickle.dump(results, f)

