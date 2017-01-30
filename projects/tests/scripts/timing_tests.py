import itertools

__author__ = 'duarte'
from modules.parameters import ParameterSet, ParameterSpace, extract_nestvalid_dict
from modules.input_architect import EncodingLayer, StimulusSet, InputSignalSet
from modules.net_architect import Network
from modules.io import set_storage_locations
from modules.signals import iterate_obj_list, empty
from modules.visualization import set_global_rcParams, InputPlots, extract_encoder_connectivity, TopologyPlots
from modules.analysis import reset_decoders
from modules.auxiliary import iterate_input_sequence
import cPickle as pickle
from scipy.sparse import coo_matrix
import matplotlib.pyplot as pl
import numpy as np
import time
import nest

# ######################################################################################################################
# Experiment options
# ======================================================================================================================
plot = True
display = True
save = False
debug = True
online = True

# ######################################################################################################################
# Extract parameters from file and build global ParameterSet
# ======================================================================================================================
params_file = '../parameters/dc_input.py'
# params_file = '../parameters/spike_pattern_input.py'

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
net.merge_subpopulations([net.populations[0], net.populations[1]])

# ######################################################################################################################
# Randomize initial variable values
# ======================================================================================================================
for idx, n in enumerate(list(iterate_obj_list(net.populations))):
	if hasattr(parameter_set.net_pars, "randomize_neuron_pars"):
		randomize = parameter_set.net_pars.randomize_neuron_pars[idx]
		for k, v in randomize.items():
			n.randomize_initial_states(k, randomization_function=v[0], **v[1])

# ######################################################################################################################
# Connect Network
# ======================================================================================================================
# net.connect_populations(parameter_set.connection_pars)

# ######################################################################################################################
# Build and connect input
# ======================================================================================================================
# Create StimulusSet
stim_set_time = time.time()
stim = StimulusSet(parameter_set, unique_set=True)
stim.create_set(parameter_set.stim_pars.full_set_length)
stim.discard_from_set(parameter_set.stim_pars.transient_set_length)
stim.divide_set(parameter_set.stim_pars.transient_set_length, parameter_set.stim_pars.train_set_length,
                parameter_set.stim_pars.test_set_length)
print "- Elapsed Time: {0}".format(str(time.time()-stim_set_time))

# Create InputSignalSet
input_set_time = time.time()
inputs = InputSignalSet(parameter_set, stim, online=online)

inputs.generate_full_set(stim)
if stim.transient_set_labels:
	inputs.generate_transient_set(stim)

inputs.generate_unique_set(stim)
inputs.generate_train_set(stim)
inputs.generate_test_set(stim)
print "- Elapsed Time: {0}".format(str(time.time() - input_set_time))

# Plot example signal
if plot and debug and not online:
	fig_inp = pl.figure()
	ax1 = fig_inp.add_subplot(211)
	ax2 = fig_inp.add_subplot(212)
	fig_inp.suptitle('Input Stimulus / Signal')
	inp_plot = InputPlots(stim_obj=stim, input_obj=inputs.train_set_signal, noise_obj=inputs.train_set_noise)
	inp_plot.plot_stimulus_matrix(set='train', ax=ax1, save=False, display=False)
	inp_plot.plot_input_signal(ax=ax2, save=paths['figures']+paths['label'], display=display)
	inp_plot.plot_input_signal(save=paths['figures']+paths['label'], display=display)
	inp_plot.plot_signal_and_noise(save=paths['figures']+paths['label'], display=display)
parameter_set.kernel_pars.sim_time = inputs.train_stimulation_time + inputs.test_stimulation_time

if save:
	stim.save(paths['inputs'])
	if debug:
		inputs.save(paths['inputs'])

# ######################################################################################################################
# Encode Input
# ======================================================================================================================
enc_layer = EncodingLayer(parameter_set.encoding_pars, signal=inputs.full_set_signal, online=online)
enc_layer.connect(parameter_set.encoding_pars, net)

# if plot and debug:
# 	extract_encoder_connectivity(enc_layer, net, display, save=paths['figures']+paths['label'])

# ######################################################################################################################
# Set-up Analysis
# ======================================================================================================================
net.connect_devices()
net.connect_decoders(parameter_set.decoding_pars)

# Attach decoders to input encoding populations
if not empty(enc_layer.encoders) and hasattr(parameter_set.encoding_pars, "input_decoder"):
	enc_layer.connect_decoders(parameter_set.encoding_pars.input_decoder)


enc_layer.extract_connectivity(net)

# print enc_layer.generator_names


########################################################################################################################
stimulus_set = stim
input_signal_set = inputs
set_name = "full"
record = True
store_activity = True
########################################################################################################################

# determine timing compensations required
enc_layer.determine_total_delay()
encoder_delay = enc_layer.total_delay
decoder_delays = []
decoder_resolutions = []
for n_pop in list(itertools.chain(*[net.merged_populations, net.populations])):
	if n_pop.decoding_layer is not None:
		n_pop.decoding_layer.determine_total_delay()
		decoder_delays.append(n_pop.decoding_layer.total_delays)
		decoder_resolutions.append(n_pop.decoding_layer.extractor_resolution)
decoder_delay = max(list(itertools.chain(*decoder_delays)))
decoder_resolution = min(list(itertools.chain(*decoder_resolutions)))
time_correction_factor = encoder_delay + decoder_resolution

# extract important parameters:
sampling_times = parameter_set.decoding_pars.sampling_times
if hasattr(parameter_set.encoding_pars.generator, "jitter"):
	jitter = parameter_set.encoding_pars.generator.jitter
else:
	jitter = None

# determine set to use and its properties
if set_name is None:
	set_name = "full"
all_labels = getattr(stimulus_set, "{0}_set_labels".format(set_name))
if isinstance(all_labels[0], list):
	labels = np.unique(list(itertools.chain(*all_labels)))
	set_labels = list(itertools.chain(*all_labels))
else:
	labels = np.unique(all_labels)
	set_labels = all_labels
set_size = len(set_labels)
input_signal = getattr(input_signal_set, "{0}_set_signal".format(set_name))
stimulus_seq = getattr(stimulus_set, "{0}_set".format(set_name))
if input_signal_set.online:
	signal_iterator = getattr(input_signal_set, "{0}_set_signal_iterator".format(set_name))
else:
	signal_iterator = None

# set state sampling parameters - TODO!! why t_step and t_samp?
if sampling_times is None and not input_signal_set.online:
	t_samp = np.sort(list(iterate_obj_list(input_signal.offset_times))) # extract stimulus offset times from
# input_signal
elif sampling_times is None and input_signal_set.online:
	t_samp = [0.] # offset times will be specified online, in the main iteration

## rethink this!!
elif sampling_times is not None and input_signal_set.online:
	t_samp = sampling_times
	t_step = [0]
else:
	t_samp = sampling_times


if not record:
	print("\n!!! No population activity data will be gathered !!!")
if store_activity:
	print("\n\n!!! All responses will be stored !!!")
	epochs = {k: [] for k in labels}
	t0 = nest.GetKernelStatus()['time'] + decoder_resolution

print("\nSimulating {0} steps".format(str(set_size)))

# ################################ Main Loop ###################################
for idx, state_sample_time in enumerate(t_samp):

	internal_time = nest.GetKernelStatus()['time']

	# determine simulation time for current stimulus - TODO FUNCTION
	if input_signal_set.online and idx < set_size:
		local_signal = signal_iterator.next()
		stimulus_duration = list(itertools.chain(*local_signal.durations))[0]
		stimulus_onset = t_samp[-1] # prior to adding new step
		local_signal.time_offset(stimulus_onset)
		stimulus_offset = list(itertools.chain(*local_signal.offset_times))[0]
		t_samp.append(stimulus_offset)
		state_sample_time = t_samp[-1] # new time
		simulation_time = stimulus_duration # stimulus duration..
		if local_signal.intervals[-1]:
			simulation_time += local_signal.intervals[-1]
	else:
		local_signal = None
		simulation_time = state_sample_time
		if idx < len(t_samp) - 1:
			if input_signal.intervals[idx]:
				simulation_time += input_signal.intervals[idx]

	# correct for delays and sample acquisition times
	# if internal_time == 0.:
	# 	simulation_time += time_correction_factor

	assert (simulation_time > 0.), "Simulation time cannot be <= 0"
	# print t_samp, internal_time, simulation_time, state_sample_time


	if idx < set_size:
		if store_activity:
			epochs[set_labels[idx]].append((stimulus_onset, state_sample_time))

		print("\nSimulating step {0} / stimulus {1} [{2} ms]".format(str(idx + 1), str(set_labels[idx]),
		                                                             str(simulation_time)))

		# update spike template data - TODO FUNCTION
		if all(['spike_pattern' in n for n in list(iterate_obj_list(enc_layer.generator_names))]):
			assert (len(input_signal_set.spike_patterns) == stimulus_set.dims), "Incorrect number of spike " \
			                                                                    "patterns"
			if input_signal_set.online and local_signal is not None:
				stimulus_id = [nx for nx in range(stimulus_set.dims) if t_samp[-1] in local_signal.offset_times[
					nx]]
			else:
				stimulus_id = [nx for nx in range(stimulus_set.dims) if
				               t_samp[idx] in input_signal.offset_times[nx]]
			sk_pattern = input_signal_set.spike_patterns[stimulus_id[0]].copy()

			if jitter is not None:
				if jitter[1]:  # compensate for boundary effects
					sk_pattern.jitter(jitter[0])
					resize_window = sk_pattern.time_parameters()
					spks = sk_pattern.time_slice(resize_window[0] + jitter[0], resize_window[1] - jitter[
						0])
					spks.time_offset(-jitter[0])
				else:
					spks = sk_pattern.jitter(jitter[0])
			else:
				spks = sk_pattern

			spks = spks.time_offset(stimulus_onset, True)
			# spks.raster_plot(True)
			enc_layer.update_state(spks)


		# update signal - TODO FUNCTION
		elif input_signal_set is not None and local_signal is not None and input_signal_set.online:
			stim_input = coo_matrix(stimulus_seq.todense()[:, idx])
			local_signal.input_signal = local_signal.generate_single_step(stim_input)

			# print min(local_signal.input_signal.time_axis()), max(local_signal.input_signal.time_axis())
			# print min(local_signal.time_data), max(local_signal.time_data)
			#
			# local_signal.time_offset(local_signal.dt)
			# print min(local_signal.input_signal.time_axis()), max(local_signal.input_signal.time_axis())
			# print min(local_signal.time_data), max(local_signal.time_data)
			# if internal_time == 0.:
			# # 	## !!! TEST - device update times cannot be 0.
			# 	dt = nest.GetKernelStatus()['resolution']
			# if internal_time == 0.:
			local_signal.time_offset(decoder_resolution)
			enc_layer.update_state(local_signal.input_signal)

		# simulate and extract recordings from devices
		if internal_time == 0.:
			net.simulate(simulation_time + decoder_resolution)
			reset_decoders(net, enc_layer)
			net.simulate(encoder_delay)
		else:
			net.simulate(simulation_time - encoder_delay)
			reset_decoders(net, enc_layer)
			net.simulate(decoder_resolution)

		net.extract_population_activity(t_start=stimulus_onset + decoder_resolution, t_stop=state_sample_time)
		net.extract_network_activity()
		enc_layer.extract_encoder_activity(t_start=stimulus_onset + decoder_resolution, t_stop=state_sample_time)
		if not empty(net.merged_populations):
			net.merge_population_activity(start=stimulus_onset + decoder_resolution, stop=state_sample_time, save=True)


		if record: #- TODO FUNCTION
			# Extract and store state vectors
			for ctr, n_pop in enumerate(list(itertools.chain(*[net.merged_populations,
			                                                   net.populations, enc_layer.encoders]))):
				if n_pop.decoding_layer is not None:

					# n_pop.decoding_layer.extract_activity(start=stimulus_onset + decoder_resolution, stop=state_sample_time)
					# n_pop.decoding_layer.evaluate_decoding(n_neurons=5, display=display, save=False)


					n_pop.decoding_layer.extract_state_vector(time_point=round(state_sample_time, 2), save=True)  ### !!!
					# n_pop.decoding_layer.reset_states()
			# clear devices
			if not store_activity:
				net.flush_records(decoders=True)
				enc_layer.flush_records(decoders=True)
if record:
	# compile state matrices:
	for n_pop in list(itertools.chain(*[net.merged_populations, net.populations, enc_layer.encoders])):
		if n_pop.decoding_layer is not None:
			n_pop.decoding_layer.compile_state_matrix()

#
# # TODO FUNCTION
if store_activity:
	# store full activity
	net.extract_population_activity(t_start=t0,
	                                t_stop=nest.GetKernelStatus()['time'] - time_correction_factor)
	net.extract_network_activity()
	enc_layer.extract_encoder_activity(t_start=t0,
	                                   t_stop=nest.GetKernelStatus()['time'] - time_correction_factor)
	if not empty(net.merged_populations):
		net.merge_population_activity(start=t0,
		                              stop=nest.GetKernelStatus()['time'] - time_correction_factor,
		                              save=True)
	for ctr, n_pop in enumerate(list(itertools.chain(*[net.merged_populations,
	                                                   net.populations, enc_layer.encoders]))):
		if n_pop.decoding_layer is not None:
			n_pop.decoding_layer.extract_activity(start=t0,
			                                      stop=nest.GetKernelStatus()['time'] - time_correction_factor,
			                                      save=True)

######################################################################################################
for n_pop in list(itertools.chain(*[net.merged_populations, net.populations])):
	if n_pop.decoding_layer is not None:
		n_pop.decoding_layer.evaluate_decoding(n_neurons=2, display=display, save=paths['figures']+paths['label'])


'''
# ######################################################################################################################
# Simulate (Full Set)
# ======================================================================================================================
iterate_input_sequence(net, enc_layer, parameter_set, stim, inputs, set_name='full', record=True,
                          store_activity=True)
# evaluate the decoding methods (only if store_activity was set to True, otherwise no activity remains stored for
# analysis)
# if debug:
for ctr, n_pop in enumerate(list(itertools.chain(*[net.merged_populations,
					                net.populations]))):#, enc_layer.encoders]))):
	if n_pop.decoding_layer is not None:
		n_pop.decoding_layer.evaluate_decoding(display=display, save=paths['figures']+paths['label'])


if stim.transient_set_labels:
	if not online:
		print "\nTransient time = {0} ms".format(str(parameter_set.kernel_pars.transient_t))


	iterate_input_sequence(net, inputs.transient_set_signal, enc_layer, sampling_times=None,
	                       sampling_lag=2., stim_set=stim, input_set=inputs, set_name='transient',
	                       store_responses=False, record=False,
	                       jitter=parameter_set.encoding_pars.generator.jitter)
	parameter_set.kernel_pars.transient_t = nest.GetKernelStatus()['time']
	net.extract_population_activity(t_start=0., t_stop=parameter_set.kernel_pars.transient_t -
	                                                   parameter_set.kernel_pars.resolution)
	net.extract_network_activity()

	# sanity check
	activity = []
	for spikes in net.spiking_activity:
		activity.append(spikes.mean_rate())
	if not np.mean(activity) > 0:
		raise ValueError("No activity recorded in main network! Stopping simulation..")

	if parameter_set.kernel_pars.transient_t > 1000.:
		analysis_interval = [1000, parameter_set.kernel_pars.transient_t]
		results['population_activity'] = population_state(net, parameter_set=parameter_set,
		                                                  nPairs=500, time_bin=1.,
		                                                  start=analysis_interval[0],
		                                                  stop=analysis_interval[1] -
		                                                       parameter_set.kernel_pars.resolution,
		                                                  plot=plot, display=display,
		                                                  save=paths['figures']+paths['label'])
		enc_layer.extract_encoder_activity()
		# results.update(evaluate_encoding(enc_layer, parameter_set, analysis_interval,
		#                                  inputs.transient_set_signal, plot=plot, display=display,
		#                                  save=paths['figures']+paths['label']))

	net.flush_records()
	enc_layer.flush_records()

# ######################################################################################################################
# Simulate (Unique Sequence)
# ======================================================================================================================
if not online:
	print "\nUnique Sequence time = {0} ms".format(str(inputs.unique_stimulation_time))

iterate_input_sequence(net, inputs.unique_set_signal, enc_layer,
                       sampling_times=parameter_set.decoding_pars.global_sampling_times,
                       sampling_lag=2., stim_set=stim, input_set=inputs,
                       set_name='unique', store_responses=False,
                       jitter=parameter_set.encoding_pars.generator.jitter)
results['rank'] = get_state_rank(net)
n_stim = len(stim.elements)
print "State Rank: {0} / {1}".format(str(results['rank']), str(n_stim))
for n_pop in list(itertools.chain(*[net.populations, net.merged_populations])):
	if not empty(n_pop.state_matrix):
		n_pop.flush_states()
for n_enc in enc_layer.encoders:
	if not empty(n_enc.state_matrix):
		n_enc.flush_states()

# ######################################################################################################################
# Simulate (Train period)
# ======================================================================================================================
if not online:
	print "\nTrain time = {0} ms".format(str(inputs.train_stimulation_time))
iterate_input_sequence(net, inputs.train_set_signal, enc_layer,
                       sampling_times=parameter_set.decoding_pars.global_sampling_times,
                       sampling_lag=2., stim_set=stim, input_set=inputs,
                       set_name='train', store_responses=False,
                       jitter=parameter_set.encoding_pars.generator.jitter)

# ######################################################################################################################
# Train Readouts
# ======================================================================================================================
train_all_readouts(parameter_set, net, stim, inputs.train_set_signal, encoding_layer=enc_layer, flush=True, debug=debug,
                   plot=plot, display=display, save=paths)

# ######################################################################################################################
# Simulate (Test period)
# ======================================================================================================================
if not online:
	print "\nTest time = {0} ms".format(str(inputs.test_stimulation_time))
iterate_input_sequence(net, inputs.test_set_signal, enc_layer,
                       sampling_times=parameter_set.decoding_pars.global_sampling_times,
                       sampling_lag=2., stim_set=stim, input_set=inputs, set_name='test',
                       store_responses=False, #paths['activity'],
                       jitter=parameter_set.encoding_pars.generator.jitter)

# ######################################################################################################################
# Test Readouts
# ======================================================================================================================
test_all_readouts(parameter_set, net, stim, inputs.test_set_signal, encoding_layer=enc_layer, flush=False, debug=debug,
                  plot=plot, display=display, save=paths)

results['Performance'] = {}
results['Performance'].update(analyse_performance_results(net, enc_layer, plot=plot, display=display, save=paths[
	                                                                                                  'figures']+paths[
	'label']))

# ######################################################################################################################
# Save data
# ======================================================================================================================
if save:
	with open(paths['results'] + 'Results_' + parameter_set.label, 'w') as f:
		pickle.dump(results, f)
	parameter_set.save(paths['parameters'] + 'Parameters_' + parameter_set.label)

'''