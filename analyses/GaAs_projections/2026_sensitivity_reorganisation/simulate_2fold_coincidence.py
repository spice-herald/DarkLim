import numpy as np
import random
from multiprocessing import Pool
import pandas as pd
from scipy.interpolate import interp1d
from scipy.special import gamma
import scipy.stats as stats
from scipy.stats import binom


import sys
import darklim
from darklim.utils import get_cache_path

from qetpy import gen_noise
import detprocess
import h5py
import time
import pickle
from time import localtime, strftime
import yaml


##############################

#lee_input_file = 'Run57_LEE_templates/combined_spectra_shared57_CRESST_extrap.txt'
def get_LEE_bkgd(scale_by=1.0, LEE_file = None):
    """
        load a .txt file with LEE spectra, divides it by scale_by.
        lee_file is passed to get_cache_path to allow it to call the module data. 
    """
    fin = get_cache_path(LEE_file)
    template = np.loadtxt(fin,skiprows=1)
    #86400 = number of sec in a day
    interp_func = interp1d(template[:,0],scale_by*template[:,1] / 86400,
                            bounds_error=False,
                            #fill_value='extrapolate',
                            fill_value=0.,
                            assume_sorted=True)
    return interp_func




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
parameter_file = sys.argv[4]
with open(parameter_file) as file:
    constants = yaml.safe_load(file)

lee_input_file = get_cache_path(constants["LEE_file"])

### Load template
template = detprocess.Template(verbose=True)
filter_file = get_cache_path(constants["filter_file"])
print("loading filter file ",filter_file)
template.load_hdf5(filter_file)

# Load shared 2x1 template to get the energy scale right, but we only use the left channel template shape and PSD
channel = constants["template_channel"]

templates_td, t_sec_template, metadata = template.get_template(channel, return_metadata=True, tag='shared')
templates_td = np.reshape(templates_td[0,0], (1,1,templates_td.shape[-1]))
_, _, f_frequencies = templates_td.shape
fs = metadata['sample_rate']
pretrig = metadata['nb_pretrigger_samples']

csd, freqs_csd, _ = template.get_csd(channel, fold=False, return_metadata=True, tag='default')
csd = np.reshape(csd[0,0], (1,1,csd.shape[-1]))

### Background parameters
rate_Hz_GaAs = constants["rate_Hz_GaAs"]  # Event rate in Hz (for the simulation--this should be as high as possible without inducing pileup)
rate_Hz_PD = constants["rate_Hz_PD"]  # Event rate in Hz (for the simulation--this should be as high as possible without inducing pileup)

### Convert to multichannel format

# Rescale template so that the energy resolution per channel is some assumed value
base_energy_resolution = constants["base_energy_resolution"]  # eV; this is the resolution of the 1x1 shared template from Run 57
GaAs_energy_resolution = constants["GaAs_energy_resolution"]
PD_energy_resolution = constants["PD_energy_resolution"]
n_sigma = constants["nsigma"]  # Number of sigma for the trigger

template_OF = np.zeros((2, 2, f_frequencies))
template_OF[0, 0, :] = np.copy(templates_td)
template_OF[1, 1, :] = np.copy(templates_td) 

csd_OF = np.zeros((2, 2, f_frequencies))
csd_OF[0, 0, :] = np.copy(csd) * (GaAs_energy_resolution / base_energy_resolution)**2
csd_OF[1, 1, :] = np.copy(csd) * (PD_energy_resolution / base_energy_resolution)**2

##############################


### Set up the simulation parameters
#full_data_length_hours = 6
#n_cores = 12
full_data_length_hours = constants["run_length_hours"]
n_cores = 1

sim_trace_length = 1_250_000 * 10
sim_trace_length_s = sim_trace_length / fs  # Convert to seconds

n_simulations = int(np.floor(full_data_length_hours * 3600 / sim_trace_length_s))
n_traces_per_sim = int(sim_trace_length / f_frequencies)
print(f'Running {n_simulations} simulations in parallel with {n_cores} cores')
f_LEE_bkgd = get_LEE_bkgd(LEE_file=constants["LEE_file"])
E_cutoff_PD = constants["E_cutoff_PD"]*1e-3 #keV
E_max_PD = constants["E_max_PD"]*1e-3 #keV
E_arr = np.geomspace(E_cutoff_PD, E_max_PD, 1000)
dRdE_arr = f_LEE_bkgd(E_arr)
R_LEE_PD = np.trapz(dRdE_arr, E_arr)
if R_LEE_PD > rate_Hz_PD:
    rate_Hz_PD = R_LEE_PD
print(f'LEE Background (PD): {R_LEE_PD:.2e} Hz per pixel above {E_cutoff_PD*1e3:.2f} eV')
print(f'Event rate (PD): {rate_Hz_PD:.2e} Hz per pixel above {E_cutoff_PD*1e3:.2f} eV')

E_cutoff_GaAs = constants["E_cutoff_GaAs"]*1e-3 #keV
E_max_GaAs = constants["E_max_GaAs"]*1e-3 #keV
E_arr_GaAs = np.geomspace(E_cutoff_GaAs, E_max_GaAs, 1000)
dRdE_arr_GaAs = f_LEE_bkgd(E_arr_GaAs)
R_LEE_GaAs = np.trapz(dRdE_arr_GaAs, E_arr_GaAs)
if R_LEE_GaAs > rate_Hz_GaAs:
    rate_Hz_GaAs = R_LEE_GaAs
print(f'LEE Background (GaAs): {R_LEE_GaAs:.2e} Hz per pixel above {E_cutoff_GaAs*1e3:.2f} eV')
print(f'Event rate (GaAs): {rate_Hz_GaAs:.2e} Hz per pixel above {E_cutoff_GaAs*1e3:.2f} eV')



def simulate_and_collect(sim_number):
    """
    Simulate a single run of the experiment (typically 10 seconds) and collect the results.
    """

    rng = np.random.default_rng(seed_1 + sim_number * 100)

    # Estimate the number of events in the two pixels
    expected_events_PD = int(rate_Hz_PD * sim_trace_length_s * 2.)  # Overestimate to be safe
    expected_events_GaAs = int(rate_Hz_GaAs * sim_trace_length_s * 2.)  # Overestimate to be safe

    # Simulate noise in the two pixels
    simulated_waveform_joint = gen_noise(csd_OF, fs, n_traces_per_sim)# , rng=rng)
    simulated_waveform_pix0_td = np.zeros(f_frequencies * n_traces_per_sim)
    simulated_waveform_pix1_td = np.zeros(f_frequencies * n_traces_per_sim)
    for trace_i in range(n_traces_per_sim):
        simulated_waveform_pix0_td[trace_i * f_frequencies:(trace_i + 1) * f_frequencies] = simulated_waveform_joint[trace_i, 0, :]
        simulated_waveform_pix1_td[trace_i * f_frequencies:(trace_i + 1) * f_frequencies] = simulated_waveform_joint[trace_i, 1, :]

    SE_PD = darklim.sensitivity.SensEst(1, 1/86400 * expected_events_PD / R_LEE_PD, tm='GaAs', eff=1., gain=1., seed=seed_2 + sim_number * 100)
    SE_PD.reset_sim()
    SE_PD.add_lee_bkgd_from_file(lee_input_file, scale_by=constants["LEE_scale"])
    
    SE_GaAs = darklim.sensitivity.SensEst(1, 1/86400 * expected_events_GaAs / R_LEE_GaAs, tm='GaAs', eff=1., gain=1., seed=seed_2 + sim_number * 200)
    SE_GaAs.reset_sim()
    SE_GaAs.add_lee_bkgd_from_file(lee_input_file, scale_by=constants["LEE_scale"])

    # Simulate arrival times and energies for pixel 0
    arrival_times_pix0 = np.cumsum(rng.exponential(1.0 / rate_Hz_GaAs, expected_events_GaAs)) * fs
    arrival_times_pix0 = arrival_times_pix0[arrival_times_pix0 < (sim_trace_length - f_frequencies)].astype(int)
    energies_pix0 = SE_GaAs.generate_background(E_max_GaAs, e_low=E_cutoff_GaAs) * 1000 # Convert to eV

    # Add pulses to pixel 0 waveform
    for i, arrival_time in enumerate(arrival_times_pix0):
        simulated_waveform_pix0_td[arrival_time:arrival_time + f_frequencies] += energies_pix0[i] * template_OF[0,0]

    # Simulate arrival times and energies for pixel 1
    arrival_times_pix1 = np.cumsum(rng.exponential(1.0 / rate_Hz_PD, expected_events_PD)) * fs
    arrival_times_pix1 = arrival_times_pix1[arrival_times_pix1 < (sim_trace_length - f_frequencies)].astype(int)
    energies_pix1 = SE_PD.generate_background(E_max_PD, e_low=E_cutoff_PD) * 1000 # Convert to eV

    # Add pulses to pixel 1 waveform
    for i, arrival_time in enumerate(arrival_times_pix1):
        simulated_waveform_pix1_td[arrival_time:arrival_time + f_frequencies] += energies_pix1[i] * template_OF[1,1]
    
    # Convert waveform, template, and CSD into OF format
    simulated_waveform_td = np.concatenate([[simulated_waveform_pix0_td], [simulated_waveform_pix1_td]])

    # Process the data with a 2x2 trigger
    oftrigger = detprocess.OptimumFilterTrigger('MyTrigger0|MyTrigger1', fs, template_OF, csd_OF, pretrigger_samples=pretrig)
    oftrigger.update_trace(simulated_waveform_td)
    dynamic_threshold_function = lambda amp: max(100, 2 * 78 * np.log(amp / 13))
    oftrigger.find_triggers(n_sigma, dynamic=True, dynamic_threshold_function=dynamic_threshold_function)

    # Extract trigger amplitudes and times from the 2x2 trigger
    E0_arr = oftrigger.get_trigger_data()['MyTrigger0|MyTrigger1']['trigger_amplitude_0']
    E1_arr = oftrigger.get_trigger_data()['MyTrigger0|MyTrigger1']['trigger_amplitude_1']
    trigger_times = oftrigger.get_trigger_data()['MyTrigger0|MyTrigger1']['trigger_index']
    trigger_delta_chi2 = oftrigger.get_trigger_data()['MyTrigger0|MyTrigger1']['trigger_delta_chi2']

    filtered_trace_0, filtered_trace_1 = oftrigger.get_filtered_trace()
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

    # For each 2x2 trigger, compare chi2 values from the three triggers:
    if sim_number % n_cores == 0:
        print(f'In simulation {sim_number}, starting the loop over {n_triggers} times. Currently', strftime("%m-%d %H:%M:%S", localtime()))

    delta_delta = np.zeros(n_triggers)
    delta_chi2_1x1_a = np.zeros(n_triggers)
    delta_chi2_1x1_b = np.zeros(n_triggers)
    
    for i, trigger_time in enumerate(trigger_times):

        # Chi2 from the 2x2 trigger
        delta_chi2_2x2 = trigger_delta_chi2[i]

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

        # Compute the delta delta chi2
        delta_delta[i] = (dchi2_1x1_a + dchi2_1x1_b) - delta_chi2_2x2
        delta_chi2_1x1_a[i] = dchi2_1x1_a
        delta_chi2_1x1_b[i] = dchi2_1x1_b

    detected = (np.array(E0_arr) > (n_sigma * GaAs_energy_resolution)) * \
            (np.array(E1_arr) > (n_sigma * PD_energy_resolution))

    waveforms_detected = []
    delta_chi2_detected = []

    # idxs = np.where(detected)[0]
    # idxs = np.append(idxs, rng.uniform(0, len(detected), 10).astype(int))  # Add some random indices to the detected ones for testing
    times_recorded = []
    for i in np.where(detected)[0]:
    # for i in idxs:
        t = trigger_times[i]

        w0 = np.copy(filtered_trace_0[t - pretrig:t + pretrig])
        w1 = np.copy(filtered_trace_1[t - pretrig:t + pretrig])
        w = np.array([w0, w1])
        waveforms_detected.append(w)

        delta_chi2_detected.append((delta_chi2_trace[t - pretrig:t + pretrig], trigger_delta_chi2[i], delta_chi2_1x1_a[i], delta_chi2_1x1_b[i]))
        
        times_recorded.append(t)

    del simulated_waveform_td

    return list(E0_arr), list(E1_arr), [0. for _ in range(len(E1_arr))], [sim_number] * n_triggers, list(trigger_times), delta_delta, detected, waveforms_detected, delta_chi2_detected, times_recorded


results = []
for batch_start in range(0, n_simulations, n_cores):
    batch_indices = list(range(batch_start, min(batch_start + n_cores, n_simulations)))
    with Pool(n_cores) as pool:
        batch_results = pool.map(simulate_and_collect, batch_indices)
    results.extend(batch_results)



with open(constants["simulation_dir"]+f'my_simulation_2fold_{run_id:02d}_{seed_1}_{seed_2}.pkl', 'wb') as f:
    pickle.dump(results, f)

