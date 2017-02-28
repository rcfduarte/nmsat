__author__ = 'duarte'
from modules.parameters import ParameterSet, ParameterSpace, extract_nestvalid_dict
from modules.input_architect import EncodingLayer, StimulusSet, InputSignalSet
from modules.net_architect import Network
from modules.io import set_storage_locations
from modules.signals import iterate_obj_list, empty
from modules.visualization import set_global_rcParams, InputPlots
from modules.analysis import reset_decoders, analyse_state_matrix
from modules.auxiliary import iterate_input_sequence, retrieve_data_set, retrieve_stimulus_timing, \
	update_input_signals, update_spike_template, extract_state_vectors, time_keep, flush, compile_results
import cPickle as pickle
import matplotlib.pyplot as pl
import numpy as np
import time
import itertools
import sys
import nest
# specific to this project..
sys.path.append('../')
from stimulus_generator import StimulusPattern

# ######################################################################################################################
# Experiment options
# ======================================================================================================================
plot = True
display = True
save = True
debug = True
online = True

# ######################################################################################################################
# Extract parameters from file and build global ParameterSet
# ======================================================================================================================
params_file = '../../encoding_decoding/parameters/dc_stimulus_input.py'
# params_file = '../parameters/spike_pattern_input.py'

parameter_set = ParameterSpace(params_file)[0]
parameter_set = parameter_set.clean(termination='pars')

# parameter_set.encoding_pars.encoder.N = 0 # avoid connecting the parrots..
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
results = dict(rank={}, performance={}, dimensionality={})

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
net.merge_subpopulations([net.populations[0], net.populations[1]], name='EI')

# ######################################################################################################################
# Randomize initial variable values
# ======================================================================================================================
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
stim_set = StimulusSet(unique_set=False)
stim_set.load_data(input_sequence, type='full_set_labels')
stim_set.discard_from_set(parameter_set.stim_pars.transient_set_length)
stim_set.divide_set(parameter_set.stim_pars.transient_set_length, parameter_set.stim_pars.train_set_length,
                    parameter_set.stim_pars.test_set_length)

# Specify target and convert to StimulusSet object
target_set = StimulusSet(unique_set=False)
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
inputs = InputSignalSet(parameter_set, stim_set, online=online, rng=parameter_set.kernel_pars['np_seed'])
inputs.generate_datasets(stim_set)
print "- Elapsed Time: {0}".format(str(time.time() - input_set_time))

# Plot example signal
if plot and debug and not online:
	fig_inp = pl.figure()
	ax1 = fig_inp.add_subplot(211)
	ax2 = fig_inp.add_subplot(212)
	fig_inp.suptitle('Input Stimulus / Signal')
	inp_plot = InputPlots(stim_obj=stim_set, input_obj=inputs.test_set_signal, noise_obj=inputs.test_set_noise)
	inp_plot.plot_stimulus_matrix(set='test', ax=ax1, save=False, display=False)
	inp_plot.plot_input_signal(ax=ax2, save=paths['figures']+paths['label'], display=display)
	inp_plot.plot_input_signal(save=paths['figures']+paths['label'], display=display)
	inp_plot.plot_signal_and_noise(save=paths['figures']+paths['label'], display=display)
parameter_set.kernel_pars.sim_time = inputs.train_stimulation_time + inputs.test_stimulation_time

if save:
	stim_pattern.save(paths['inputs'])
	stim_set.save(paths['inputs'])
	if debug:
		inputs.save(paths['inputs'])

print "- Elapsed Time: {0}".format(str(time.time() - input_set_time))
inputs_build_time = time.time() - input_set_time

# ######################################################################################################################
# Encode Input
# ======================================================================================================================
enc_layer = EncodingLayer(parameter_set.encoding_pars, signal=inputs.full_set_signal, online=online)
enc_layer.connect(parameter_set.encoding_pars, net)
enc_layer.extract_connectivity(net, sub_set=True, progress=True)

# ######################################################################################################################
# Set-up Analysis
# ======================================================================================================================
net.connect_devices()
net.connect_decoders(parameter_set.decoding_pars)

# Attach decoders to input encoding populations
if not empty(enc_layer.encoders) and hasattr(parameter_set.encoding_pars, "input_decoder") and \
				parameter_set.encoding_pars.input_decoder is not None:
	enc_layer.connect_decoders(parameter_set.encoding_pars.input_decoder)

# ######################################################################################################################
# Connect Network
# ======================================================================================================================
net.connect_populations(parameter_set.connection_pars)

########################################################################################################################
stimulus_set = stim_set
input_signal_set = inputs
set_name = "full"
record = True
store_activity = True
########################################################################################################################
print("\n\n***** Preparing to simulate {0} set *****".format(set_name))

# determine timing compensations required
enc_layer.determine_total_delay()
encoder_delay = enc_layer.total_delay
decoder_delays = []
decoder_resolutions = []
decoders = []
for n_pop in list(itertools.chain(*[net.merged_populations, net.populations, enc_layer.encoders])):
	if n_pop.decoding_layer is not None:
		decoders.append(n_pop.decoding_layer.extractors)
		n_pop.decoding_layer.determine_total_delay()
		decoder_delays.append(n_pop.decoding_layer.total_delays)
		decoder_resolutions.append(n_pop.decoding_layer.extractor_resolution)
decoder_delay = max(list(itertools.chain(*decoder_delays)))
decoder_resolution = min(list(itertools.chain(*decoder_resolutions)))
simulation_resolution = nest.GetKernelStatus()['resolution']
time_correction_factor = encoder_delay + decoder_resolution
if decoder_resolution != encoder_delay:
	print("To avoid errors in the delay compensation, it is advisable to set the output resolution to be the same " \
	      "as the encoder delays") # because the state resolution won't be enough to capture the time compensation..

# extract important parameters:
sampling_times = parameter_set.decoding_pars.sampling_times
if hasattr(parameter_set.encoding_pars.generator, "jitter"):
	jitter = parameter_set.encoding_pars.generator.jitter
else:
	jitter = None

# determine set to use and its properties
labels, set_labels, set_size, input_signal, stimulus_seq, signal_iterator = retrieve_data_set(set_name,
                                                                            stimulus_set, input_signal_set)

# set state sampling parameters - TODO!! why t_step and t_samp?
if sampling_times is None and not input_signal_set.online:
	t_samp = np.sort(list(iterate_obj_list(input_signal.offset_times)))  # extract stimulus offset times
elif sampling_times is None and input_signal_set.online:
	t_samp = [round(nest.GetKernelStatus()['time'])]  # offset times will be specified online, in the main iteration
elif sampling_times is not None and input_signal_set.online:
	t_samp = [round(nest.GetKernelStatus()['time'])]
	sub_sampling_times = sampling_times
else:
	t_samp = sampling_times

t0 = nest.GetKernelStatus()['time'] + decoder_resolution
epochs = {k: [] for k in labels}
if not record:
	print("\n!!! No population activity will be stored !!!")
if store_activity:
	if isinstance(store_activity, int):
		store = False
	else:
		store = True
	print("\n\n!!! Responses will be stored !!!")
else:
	store = store_activity
start_time = time.time()

####################################################################################################################
if sampling_times is None:  # one sample for each stimulus (acquired at the last time point of each stimulus)
	print(("\n\nSimulating {0} steps".format(str(set_size))))

	# ################################ Main Loop ###################################
	for idx, state_sample_time in enumerate(t_samp):

		# internal time
		internal_time = nest.GetKernelStatus()['time']
		stim_start = time.time()

		# determine simulation time for current stimulus
		local_signal, stimulus_duration, stimulus_onset, t_samp, state_sample_time, simulation_time = \
			retrieve_stimulus_timing(input_signal_set, idx, set_size, signal_iterator, t_samp, state_sample_time,
			                         input_signal)

		if idx < set_size:
			# store and display stimulus information
			print("\n\nSimulating step {0} / {1} - stimulus {2} [{3} ms]".format(str(idx + 1), str(set_size), str(
				set_labels[idx]), str(simulation_time)))
			epochs[set_labels[idx]].append((stimulus_onset, state_sample_time))
			state_sample_time += encoder_delay  # correct sampling time

			# update spike template data
			if all(['spike_pattern' in n for n in list(iterate_obj_list(enc_layer.generator_names))]):
				update_spike_template(enc_layer, idx, input_signal_set, stimulus_set, local_signal, t_samp,
				                      input_signal, jitter, stimulus_onset)
			# update signal
			elif input_signal_set is not None and local_signal is not None and input_signal_set.online:
				update_input_signals(enc_layer, idx, stimulus_seq, local_signal, decoder_resolution)

			# simulate and reset (if applicable)
			if internal_time == 0.:
				net.simulate(simulation_time + encoder_delay + decoder_delay)
				reset_decoders(net, enc_layer)
				net.simulate(simulation_resolution) #decoder_resolution)
			else:
				net.simulate(simulation_time - simulation_resolution)
				reset_decoders(net, enc_layer)
				net.simulate(simulation_resolution)


			#net.simulate(simulation_time)


			# extract and store activity
			# net.extract_population_activity(t_start=internal_time, t_stop=state_sample_time)
			# net.extract_network_activity()
			# enc_layer.extract_encoder_activity(t_start=stimulus_onset + encoder_delay, t_stop=state_sample_time)
			# if not empty(net.merged_populations):
			# 	net.merge_population_activity(start=stimulus_onset + encoder_delay, stop=state_sample_time,
			# 	                              save=True)

			# sample population activity
			if isinstance(store_activity, int) and set_size - idx == store_activity:
				store = True
				current_time = nest.GetKernelStatus()['time']
				epochs.update({'analysis_start': current_time})
			# if record:
			# 	extract_state_vectors(net, enc_layer, state_sample_time, store)
			# if not store:
			# 	flush(net, enc_layer, decoders=False)
			# if not record:
			# 	flush(net, enc_layer, decoders=True)

			time_keep(start_time, idx, set_size, stim_start)

	t0 = 200.
	net.extract_population_activity(t_start=t0, t_stop=nest.GetKernelStatus()['time']-simulation_resolution)
	net.extract_network_activity()
	enc_layer.extract_encoder_activity(t_start=t0, t_stop=nest.GetKernelStatus()['time']-simulation_resolution)
	if not empty(net.merged_populations):
		net.merge_population_activity(start=t0, stop=nest.GetKernelStatus()['time']-simulation_resolution,
		                              save=True)

	for ctr, n_pop in enumerate(list(itertools.chain(*[net.merged_populations,
		                                                   net.populations]))): #, enc_layer.encoders]))):
		if n_pop.decoding_layer is not None:
			n_pop.decoding_layer.extract_activity(start=t0, stop=nest.GetKernelStatus()['time'] -
			                                                     2*simulation_resolution,
			                                      save=True)
			if parameter_set.analysis_pars.store_activity:  # and debug:
				if isinstance(parameter_set.analysis_pars.store_activity, int) and \
								parameter_set.analysis_pars.store_activity <= parameter_set.stim_pars.test_set_length:
					n_pop.decoding_layer.sampled_times = n_pop.decoding_layer.sampled_times[-parameter_set.analysis_pars.store_activity:]
				else:
					n_pop.decoding_layer.sampled_times = n_pop.decoding_layer.sampled_times[-parameter_set.stim_pars.test_set_length:]
				n_pop.decoding_layer.evaluate_decoding(n_neurons=20, display=display, save=paths['figures'] + paths['label'])

			# for idx_state, n_state in enumerate(n_pop.decoding_layer.state_variables):
			# 	n_pop.decoding_layer.state_matrix[idx_state] = n_pop.decoding_layer.activity[idx_state].as_array()
			#
			# 	analyse_state_matrix(n_pop.decoding_layer.state_matrix[idx_state], set_labels)
	# compile_results(net, enc_layer, t0, time_correction_factor, record, store)

	####################################################################################################################