__author__ = 'duarte'
import sys
sys.path.insert(0, "../")
from modules.parameters import ParameterSet, ParameterSpace, extract_nestvalid_dict
from modules.input_architect import EncodingLayer, StimulusSet, InputSignalSet
from modules.net_architect import Network
from modules.io import set_storage_locations
from modules.signals import iterate_obj_list, empty
from modules.visualization import set_global_rcParams, InputPlots, extract_encoder_connectivity
from modules.analysis import characterize_population_activity
from stimulus_generator import StimulusPattern
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as pl
import time
import nest


# ######################################################################################################################
# Experiment options
# ======================================================================================================================
online = False # strictly False!
plot = False
display = True
save = True
debug = True

# ######################################################################################################################
# Extract parameters from file and build global ParameterSet
# ======================================================================================================================
params_file = '../parameters/dcinput_noise_vs_stimulus.py'
# params_file = '../parameters/spikeinput_ongoing_evoked.py'

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

###################################################################################
# Build network
# =================================================================================
net = Network(parameter_set.net_pars)
net.merge_subpopulations([net.populations[0], net.populations[1]], name='EI')

###################################################################################
# Randomize initial variable values
# =================================================================================
for idx, n in enumerate(list(iterate_obj_list(net.populations))):
	if hasattr(parameter_set.net_pars, "randomize_neuron_pars"):
		randomize = parameter_set.net_pars.randomize_neuron_pars[idx]
		for k, v in randomize.items():
			n.randomize_initial_states(k, randomization_function=v[0], **v[1])

###################################################################################
# Build Stimulus Set
# =================================================================================
stim_set_startbuild = time.time()

# Create or Load StimulusPattern
stim_pattern = StimulusPattern(parameter_set.task_pars)
stim_pattern.generate()

input_sequence, output_sequence = stim_pattern.as_index()

# Convert to StimulusSet object
stim_set = StimulusSet(unique_set=None)
stim_set.load_data(input_sequence, type='full_set_labels')
stim_set.discard_from_set(parameter_set.stim_pars.transient_set_length)
stim_set.divide_set(parameter_set.stim_pars.transient_set_length, parameter_set.stim_pars.train_set_length,
                    parameter_set.stim_pars.test_set_length)

# Specify target and convert to StimulusSet object
target_set = StimulusSet(unique_set=None)
target_set.load_data(output_sequence, type='full_set_labels')
target_set.discard_from_set(parameter_set.stim_pars.transient_set_length)
target_set.divide_set(parameter_set.stim_pars.transient_set_length, parameter_set.stim_pars.train_set_length,
                    parameter_set.stim_pars.test_set_length)

print "- Elapsed Time: {0}".format(str(time.time()-stim_set_startbuild))
stim_set_buildtime = time.time()-stim_set_startbuild

###################################################################################
# Build Input Signal Set
# =================================================================================
input_set_time = time.time()
parameter_set.input_pars.signal.N = len(np.unique(input_sequence))
# Create InputSignalSet
inputs = InputSignalSet(parameter_set, stim_set, online=online)
if not empty(stim_set.transient_set_labels):
	inputs.generate_transient_set(stim_set)
	# parameter_set.kernel_pars.transient_t = inputs.transient_stimulation_time
if not online:
	inputs.generate_full_set(stim_set)
#inputs.generate_unique_set(stim)
inputs.generate_train_set(stim_set)
inputs.generate_test_set(stim_set)

inputs.time_offset(parameter_set.kernel_pars.transient_t)

# # Plot example signal
if plot and debug and not online:
	fig_inp = pl.figure()
	ax1 = fig_inp.add_subplot(211)
	ax2 = fig_inp.add_subplot(212)
	fig_inp.suptitle('Input Stimulus / Signal')
	inp_plot = InputPlots(stim_obj=stim_set, input_obj=inputs.train_set_signal, noise_obj=inputs.train_set_noise)
	inp_plot.plot_stimulus_matrix(set='train', ax=ax1, save=False, display=False)
	inp_plot.plot_input_signal(ax=ax2, save=paths['figures']+paths['label'], display=display)
	inp_plot.plot_input_signal(save=paths['figures']+paths['label'], display=display)
	inp_plot.plot_signal_and_noise(save=paths['figures']+paths['label'], display=display)
#parameter_set.kernel_pars.sim_time = inputs.train_stimulation_time + inputs.test_stimulation_time

if save:
	stim_set.save(paths['inputs'] + paths['label'])
	if debug:
		inputs.save(paths['inputs'] + paths['label'])
		stim_pattern.save(paths['inputs'] + paths['label'])

#######################################################################################
# Encode Input
# =====================================================================================
if not online:
	input_signal = inputs.full_set_signal
else:
	input_signal = inputs.transient_set_signal
enc_layer = EncodingLayer(parameter_set.encoding_pars, signal=input_signal, online=online)
enc_layer.connect(parameter_set.encoding_pars, net)

# Attach decoders to input encoding populations
if not empty(enc_layer.encoders) and hasattr(parameter_set.encoding_pars, "input_decoder"):
	enc_layer.connect_decoders(parameter_set.encoding_pars.input_decoder)

if plot and debug:
	extract_encoder_connectivity(enc_layer, net, display, save=paths['figures']+paths['label'])

#######################################################################################
# Set-up Analysis
# =====================================================================================
net.connect_devices()
net.connect_decoders(parameter_set.decoding_pars)

######################################################################################
# Connect Network
# ====================================================================================
net.connect_populations(parameter_set.connection_pars)

######################################################################################
# Simulate (Ongoing+evoked Epoch)
# ====================================================================================
net.simulate(parameter_set.kernel_pars.transient_t + parameter_set.kernel_pars.sim_time + nest.GetKernelStatus()[
	'min_delay'])
net.extract_population_activity()
net.extract_network_activity()

# sanity check
activity = []
for spikes in net.spiking_activity:
	activity.append(spikes.mean_rate())
if not np.mean(activity) > 0:
	raise ValueError("No activity recorded in main network! Stopping simulation..")

# ######################################################################################################################
# Analyse / plot data
# ======================================================================================================================
# analysis_interval = [100, parameter_set.kernel_pars.transient_t] # discard the first 100 ms
# analysis_pars = {'time_bin': 1.,
#                  'n_pairs': 500,
#                  'tau': 20.,
#                  'window_len': 100,
#                  'summary_only': False,
#                  'complete': True,
#                  'time_resolved': False}
# results['ongoing'] = characterize_population_activity(net, parameter_set, analysis_interval, epochs=None,
#                                                 color_map='Accent', plot=plot,
#                                                 display=display, save=paths['figures']+paths['label']+'Ongoing',
#                                                 **analysis_pars)
#
# analysis_interval = [parameter_set.kernel_pars.transient_t, parameter_set.kernel_pars.transient_t +
#                      parameter_set.kernel_pars.sim_time]
# results['evoked'] = characterize_population_activity(net, parameter_set, analysis_interval, epochs=None,
#                                                 color_map='Accent', plot=plot,
#                                                 display=display, save=paths['figures']+paths['label']+'Evoked',
#                                                 **analysis_pars)

analysis_interval = [parameter_set.kernel_pars.transient_t - 500., parameter_set.kernel_pars.transient_t + 500.]
# TODO @barni moved this to parameter file
# analysis_pars = {'time_bin': 1.,
#                  'n_pairs': 500,
#                  'tau': 20.,
#                  'window_len': 100,
#                  'summary_only': False,
#                  'complete': True,
#                  'time_resolved': True,
#                  'color_subpop': True}
epochs = {'ongoing': (analysis_interval[0], parameter_set.kernel_pars.transient_t),
          'evoked': (parameter_set.kernel_pars.transient_t, analysis_interval[1])}
results['transition'] = characterize_population_activity(net, parameter_set, analysis_interval, epochs=epochs,
                                                color_map='Accent', plot=plot,
                                                display=display, save=paths['figures']+paths['label']+'Evoked',
												color_subpop=True, analysis_pars=parameter_set.analysis_pars)

net.flush_records()
enc_layer.flush_records()

#######################################################################################
# Save data
# =====================================================================================
if save:
	with open(paths['results'] + 'Results_' + parameter_set.label, 'w') as f:
		pickle.dump(results, f)
	parameter_set.save(paths['parameters'] + 'Parameters_' + parameter_set.label)