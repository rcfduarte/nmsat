__author__ = 'duarte'
import sys
sys.path.insert(0, "../")
from modules.parameters import ParameterSet, ParameterSpace, extract_nestvalid_dict
from modules.input_architect import EncodingLayer, StimulusSet, InputSignalSet, InputNoise
from modules.net_architect import Network
from modules.io import set_storage_locations
from modules.signals import iterate_obj_list, empty, gather_analog_activity
from modules.visualization import set_global_rcParams, InputPlots, ActivityAnimator, plot_input_example
from modules.analysis import characterize_population_activity
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as pl
import time
from auxiliary_fcns import plot_single_neuron_response, spike_triggered_synaptic_responses, PSP_kinetics, PSC_kinetics
import nest


# ######################################################################################################################
# Experiment options
# ======================================================================================================================
plot = True
display = True
save = True

# ######################################################################################################################
# Extract parameters from file and build global ParameterSet
# ======================================================================================================================
params_file = '../parameters/synaptic_response_rest.py'

parameter_set = ParameterSpace(params_file)[0]
parameter_set = parameter_set.clean(termination='pars')

if not isinstance(parameter_set, ParameterSet):
	if isinstance(parameter_set, basestring) or isinstance(parameter_set, dict):
		parameter_set = ParameterSet(parameter_set)
	else:
		raise TypeError("parameter_set must be ParameterSet, string with full path to parameter file or dictionary")

# ######################################################################################################################
# Setup extra variables and parameters
# ======================================================================================================================
if plot:
	set_global_rcParams(parameter_set.kernel_pars['mpl_path'])
paths = set_storage_locations(parameter_set, save)

np.random.seed(parameter_set.kernel_pars['np_seed'])
results = dict()

# ######################################################################################################################
# Set kernel and simulation parameters
# ======================================================================================================================
print '\nRuning ParameterSet {0}'.format(parameter_set.label)
nest.ResetKernel()
nest.set_verbosity('M_WARNING')
nest.SetKernelStatus(extract_nestvalid_dict(parameter_set.kernel_pars.as_dict(), param_type='kernel'))

# ######################################################################################################################
# Build network
# ======================================================================================================================
net = Network(parameter_set.net_pars)

# ######################################################################################################################
# Randomize initial variable values
# ======================================================================================================================
for idx, n in enumerate(list(iterate_obj_list(net.populations))):
	if hasattr(parameter_set.net_pars, "randomize_neuron_pars"):
		randomize = parameter_set.net_pars.randomize_neuron_pars[idx]
		for k, v in randomize.items():
			n.randomize_initial_states(k, randomization_function=v[0], **v[1])

# ######################################################################################################################
# Build and connect input
# ======================================================================================================================
enc_layer = EncodingLayer(parameter_set.encoding_pars)
enc_layer.connect(parameter_set.encoding_pars, net)

# ######################################################################################################################
# Set-up Analysis
# ======================================================================================================================
net.connect_devices()

# ######################################################################################################################
# Simulate
# ======================================================================================================================
if parameter_set.kernel_pars.transient_t:
	net.simulate(parameter_set.kernel_pars.transient_t)
	net.flush_records()

net.simulate(parameter_set.kernel_pars.sim_time + parameter_set.kernel_pars.resolution)

# ######################################################################################################################
# Extract and store data
# ======================================================================================================================
net.extract_population_activity()
net.extract_network_activity()
# net.flush_records()

# ######################################################################################################################
# Analyse / plot data
# ======================================================================================================================
time_window = parameter_set.kernel_pars.window
analysis_interval = [parameter_set.kernel_pars.transient_t,
                     parameter_set.kernel_pars.transient_t + parameter_set.kernel_pars.sim_time]
t_axis = np.arange(analysis_interval[0], analysis_interval[1], parameter_set.net_pars.analog_device_pars[0]['interval'])

E_input_times = np.array(parameter_set.encoding_pars.generator.model_pars[
	                          parameter_set.encoding_pars.generator.labels.index('E_input')]['spike_times'])

if not empty(E_input_times):
	E_input_times = np.intersect1d(E_input_times[E_input_times > analysis_interval[0]], E_input_times[E_input_times
	                                <= analysis_interval[1] - time_window[1]])
I_input_times = np.array(parameter_set.encoding_pars.generator.model_pars[
	parameter_set.encoding_pars.generator.labels.index(
	'I_input')]['spike_times'])
if not empty(I_input_times):
	I_input_times = np.intersect1d(I_input_times[I_input_times > analysis_interval[0]], I_input_times[I_input_times
	                                <= analysis_interval[1] - time_window[1]])

# Look at a single neuron
all_activity = gather_analog_activity(parameter_set, net, t_start=analysis_interval[0], t_stop=analysis_interval[1])

if plot:
	plot_single_neuron_response(parameter_set, all_activity, E_input_times, t_axis, analysis_interval, response_type='E')
	plot_single_neuron_response(parameter_set, all_activity, I_input_times, t_axis, analysis_interval, response_type='I')

final_results = {x: {} for x in all_activity.keys()}
results1 = ['q_ratio', 'psc_ratio']
results2 = ['mean_fit_rise', 'mean_fit_decay', 'mean_amplitude']
if not empty(E_input_times):
	results = spike_triggered_synaptic_responses(parameter_set, all_activity, time_window, E_input_times, t_axis,
	                                             response_type='E', plot=plot, display=display,
	                                             save=paths['figures']+paths['label'])

	for pop in all_activity.keys():
		for k in results1:
			final_results[pop].update({k: results[pop][k]})

	results.update(PSC_kinetics(all_activity, time_window, E_input_times, t_axis, response_type='E', plot=plot,
	                     display=display, save=paths['figures']+paths['label']))

	for pop in all_activity.keys():
		for k in results2:
			final_results[pop].update({'PSC_'+k: results[pop][k]})

	results.update(PSP_kinetics(all_activity, time_window, E_input_times, t_axis, response_type='E', plot=plot,
	                            display=display, save=paths['figures']+paths['label']))

	for pop in all_activity.keys():
		for k in results2:
			final_results[pop].update({'PSP_'+k: results[pop][k]})

if not empty(I_input_times):
	results = spike_triggered_synaptic_responses(parameter_set, all_activity, time_window, I_input_times, t_axis,
	                                              response_type='I', plot=plot, display=display,
	                                             save=paths['figures']+paths['label'])
	for pop in all_activity.keys():
		for k in results1:
			final_results[pop].update({k: results[pop][k]})

	results.update(PSC_kinetics(all_activity, time_window, I_input_times, t_axis, response_type='I', plot=plot,
                            display=display, save=paths['figures']+paths['label']))

	for pop in all_activity.keys():
		for k in results2:
			final_results[pop].update({'PSC_'+k: results[pop][k]})

	results.update(PSP_kinetics(all_activity, time_window, I_input_times, t_axis, response_type='I', plot=plot,
                            display=display, save=paths['figures']+paths['label']))

	for pop in all_activity.keys():
		for k in results2:
			final_results[pop].update({'PSP_'+k: results[pop][k]})


# ######################################################################################################################
# Save data
# ======================================================================================================================
if save:
	with open(paths['results'] + 'Results_' + parameter_set.label, 'w') as f:
		pickle.dump(results, f)
	parameter_set.save(paths['parameters'] + 'Parameters_' + parameter_set.label)














