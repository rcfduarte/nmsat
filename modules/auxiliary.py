"""
========================================================================================================================
Auxiliary Module
========================================================================================================================
Utility functions that are frequently used in specific experiments

Functions:
------------

Classes:
------------

========================================================================================================================
Copyright (C) 2017  Renato Duarte, Barna Zajzon

Neural Mircocircuit Simulation and Analysis Toolkit is free software;
you can redistribute it and/or modify it under the terms of the
GNU General Public License as published by the Free Software Foundation;
either version 2 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

The GNU General Public License does not permit this software to be
redistributed in proprietary programs.

"""
from scipy.sparse import coo_matrix
import signals
import net_architect
import input_architect
import parameters
import visualization
import analysis
import numpy as np
import itertools
import time
import nest


########################################################################################################################
# Auxiliary Functions
########################################################################################################################
def iterate_input_sequence(net, enc_layer, parameter_set, stimulus_set, input_signal_set, set_name, record=True,
                           store_activity=False):
	"""
	Run simulation sequentially, presenting one input stimulus at a time (if input is a discrete stimulus sequence),
	gathering the population responses in the DecodingLayers

	:param net: Network object
	:param enc_layer: EncodingLayer
	:param sampling_times: parameter specifying how to sample the population state (either at the end of each
	stimulus presentation (None), at fixed times (list or array of times relative to stimulus onset) or at a fixed
	sampling rate (float in [0, 1])
	:param stimulus_set: full StimulusSet object
	:param input_signal_set: full InputSignalSet object
	:param set_name: string with the name of the current set ('transient', 'unique', 'train', 'test')
	:param record: [bool] - acquire and store state matrices (according to sampling_times and state characteristics)
	:param store_activity: [bool] - record population activity for the entire simulation time (memory!)
	"""
	assert (isinstance(net, net_architect.Network)), "Please provide a Network object"
	assert (isinstance(enc_layer, input_architect.EncodingLayer)), "Please provide an EncodingLayer object"
	assert (isinstance(input_signal_set, input_architect.InputSignalSet)), "input_set must be InputSignalSet"
	assert (isinstance(parameter_set, parameters.ParameterSet)), "incorrect ParameterSet"
	assert (isinstance(stimulus_set, input_architect.StimulusSet)), "incorrect ParameterSet"

	print("\n\n***** Preparing to simulate {0} set *****".format(set_name))

	# determine timing compensations required
	enc_layer.determine_total_delay()
	encoder_delay = enc_layer.total_delay
	decoder_delays = []
	decoder_resolutions = []
	for n_pop in list(itertools.chain(*[net.merged_populations, net.populations, enc_layer.encoders])):
		if n_pop.decoding_layer is not None:
			n_pop.decoding_layer.determine_total_delay()
			decoder_delays.append(n_pop.decoding_layer.total_delays)
			decoder_resolutions.append(n_pop.decoding_layer.extractor_resolution)
	if not signals.empty(decoder_delays):
		decoder_delay = max(list(itertools.chain(*decoder_delays)))
	else:
		decoder_delay = 0.
	if not signals.empty(decoder_resolutions):
		decoder_resolution = min(list(itertools.chain(*decoder_resolutions)))
	else:
		decoder_resolution = 0.
	time_correction_factor = encoder_delay + decoder_resolution
	if decoder_resolution != encoder_delay:
		# because the state resolution won't be enough to capture the time compensation..
		print("To avoid errors in the delay compensation, it is advisable to set the output resolution "
		      "to be the same as the encoder delays")

	# extract important parameters:
	sampling_times = parameter_set.decoding_pars.sampling_times
	if hasattr(parameter_set.encoding_pars.generator, "jitter"):
		jitter = parameter_set.encoding_pars.generator.jitter
	else:
		jitter = None

	# determine set to use and its properties
	labels, set_labels, set_size, input_signal, stimulus_seq, signal_iterator = retrieve_data_set(set_name,
	                                                                            stimulus_set, input_signal_set)

	# set state sampling parameters
	if sampling_times is None and not input_signal_set.online:
		t_samp = np.sort(list(signals.iterate_obj_list(input_signal.offset_times)))  # extract stimulus offset times
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
		if isinstance(store_activity, int) and not isinstance(store_activity, bool):
			store = False
		else:
			store = True
		print("\n\n!!! Responses will be stored !!!")
	else:
		store = store_activity

	start_time = time.time()
	timing = dict(step_time=[], total_time=[])
	####################################################################################################################
	if sampling_times is None:  # one sample for each stimulus (acquired at the last time point of each stimulus)
		print("\n\nSimulating {0} steps".format(str(set_size)))

		# ################################ Main Loop ###################################
		for idx, state_sample_time in enumerate(t_samp):

			# internal time
			internal_time = nest.GetKernelStatus()['time']
			stim_start = time.time()

			# determine simulation time for current stimulus
			local_signal, stimulus_duration, stimulus_onset, t_samp,  state_sample_time, simulation_time = \
				retrieve_stimulus_timing(input_signal_set, idx, set_size, signal_iterator, t_samp, state_sample_time, input_signal)

			if idx < set_size:
				# store and display stimulus information
				print("\n\nSimulating step {0} / {1} - stimulus {2} [{3} ms]".format(str(idx + 1), str(set_size), str(
					set_labels[idx]), str(simulation_time)))
				epochs[set_labels[idx]].append((stimulus_onset, state_sample_time))
				state_sample_time += encoder_delay  # correct sampling time

				# update spike template data
				if all(['spike_pattern' in n for n in list(signals.iterate_obj_list(enc_layer.generator_names))]):
					update_spike_template(enc_layer, idx, input_signal_set, stimulus_set, local_signal, t_samp,
					                      input_signal, jitter, stimulus_onset)
				# update signal
				elif input_signal_set is not None and local_signal is not None and input_signal_set.online:
					update_input_signals(enc_layer, idx, stimulus_seq, local_signal, decoder_resolution)

				# simulate and reset (if applicable)
				if internal_time == 0.:
					net.simulate(simulation_time + encoder_delay + decoder_delay)
					analysis.reset_decoders(net, enc_layer)
					net.simulate(decoder_resolution)
				else:
					net.simulate(simulation_time - decoder_resolution)
					analysis.reset_decoders(net, enc_layer)
					net.simulate(decoder_resolution)

				# extract and store activity
				net.extract_population_activity()#t_start=stimulus_onset + encoder_delay, t_stop=state_sample_time)
				net.extract_network_activity()
				enc_layer.extract_encoder_activity(t_start=stimulus_onset + encoder_delay, t_stop=state_sample_time)
				if not signals.empty(net.merged_populations):
					net.merge_population_activity(start=stimulus_onset + encoder_delay, stop=state_sample_time,
					                              save=True)
				# sample population activity
				if isinstance(store_activity, int) and not isinstance(store_activity, bool) and set_size-(idx+1) == \
						store_activity:
					store = True
					t0 = nest.GetKernelStatus()['time']
					epochs.update({'analysis_start': t0})

					print "\n\n\n ANALYSIS START TIME: {0}".format(t0)
				if record:
					extract_state_vectors(net, enc_layer, state_sample_time)
				if not store:
					flush(net, enc_layer)

				timing['step_time'].append(time_keep(start_time, idx, set_size, stim_start))

		timing['total_time'] = (time.time() - start_time) / 60.
		compile_results(net, enc_layer, t0, time_correction_factor, record, store)

	####################################################################################################################
	# TODO - other state sampling methods
	else:
		raise NotImplementedError("Specify sampling times as None (last step sample), list or array of times, or float")

	return epochs, timing


def set_sampling_parameters(sampling_times, input_signal_set, input_signal):
	if sampling_times is None and not input_signal_set.online:
		t_samp = np.sort(list(signals.iterate_obj_list(input_signal.offset_times)))  # extract stimulus offset times
		sub_sampling_times = None
	elif sampling_times is None and input_signal_set.online:
		t_samp = [round(nest.GetKernelStatus()['time'])]  # offset times will be specified online, in the main
	# iteration
		sub_sampling_times = None
	elif sampling_times is not None and input_signal_set.online:
		t_samp = [round(nest.GetKernelStatus()['time'])]
		sub_sampling_times = sampling_times
	else:
		t_samp = sampling_times
		sub_sampling_times = None
	return t_samp, sub_sampling_times


def set_decoder_times(enc_layer, parameter_set, sampling_interval=None, correct_origin=False):
	"""
	Specify the origin and sampling interval, depending on the current setup
	:sampling_interval: provide a specific interval
	:correct_origin: force origin relative to this value
	:return:
	"""
	enc_layer.determine_total_delay()
	encoder_delay = enc_layer.total_delay
	stim_duration = parameter_set.input_pars.signal.durations
	stim_isi = parameter_set.input_pars.signal.i_stim_i
	if isinstance(correct_origin, float) or isinstance(correct_origin, int):
		add_origin = correct_origin
	else:
		add_origin = 0.
	if sampling_interval is not None:
		duration = sampling_interval
	else:
		sampling_interval = parameter_set.decoding_pars.sampling_times

		# TODO - other variants
		if (len(stim_duration) != 1 and np.mean(stim_duration) != stim_duration[0]) or all(stim_isi):
			raise NotImplementedError("Stimulus durations should be fixed and constant and inter-stimulus intervals == 0.")
		else:
			duration = stim_duration[0]

		if sampling_interval is not None:
			# divide stimulus times
			if isinstance(sampling_interval, list) or isinstance(sampling_interval, np.ndarray):
				assert(all(np.diff(sampling_interval)), "Sampling interval must be constant")
				duration = np.unique(np.diff(sampling_interval))[0]

	for extractor_idx, extractor_pars in enumerate(parameter_set.decoding_pars.state_extractor.state_specs):
		state_variable = parameter_set.decoding_pars.state_extractor.state_variable[extractor_idx]
		if state_variable != 'spikes':
			origin = add_origin + encoder_delay
			extractor_pars.update({'origin': add_origin + encoder_delay, 'interval': duration})
			print("Extractor {0}: \n- origin = {1} ms\n- interval = {2}".format(state_variable, str(origin),
			                                                                      str(duration)))
		else:
			origin = add_origin + encoder_delay + 0.1
			extractor_pars.update({'origin': add_origin + encoder_delay + 0.1, 'interval': duration})
			print("Extractor {0}: \n- origin = {1} ms\n- interval = {2}".format(state_variable, str(origin),
			                                                                      str(duration)))
	if not signals.empty(enc_layer.encoders) and hasattr(parameter_set.encoding_pars, "input_decoder") and \
					parameter_set.encoding_pars.input_decoder is not None:
		for extractor_idx, extractor_pars in enumerate(
				parameter_set.encoding_pars.input_decoder.state_extractor.state_specs):
			state_variable = parameter_set.encoding_pars.input_decoder.state_extractor.state_variable[extractor_idx]
			if state_variable != 'spikes':
				origin = add_origin + 0.0
				extractor_pars.update({'origin': add_origin + 0.0, 'interval': duration})
				print("Encoder Extractor {0}: \n- origin = {1} ms\n- interval = {2}".format(state_variable,
				                                                                             str(origin),
				                                                                      str(duration)))
			else:
				origin = add_origin + 0.1
				extractor_pars.update({'origin': add_origin + 0.1, 'interval': duration})
				print("Encoder Extractor {0}: \n- origin = {1} ms\n- interval = {2}".format(state_variable,
				                                                                              str(origin),
				                                                                              str(duration)))


def retrieve_data_set(set_name, stimulus_set, input_signal_set):
	"""
	Extract the properties of the dataset to be used
	:param set_name:
	:param stimulus_set:
	:param input_signal_set:
	:return:
	"""
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
	return labels, set_labels, set_size, input_signal, stimulus_seq, signal_iterator


def retrieve_stimulus_timing(input_signal_set, idx, set_size, signal_iterator, t_samp, state_sample_time, input_signal):
	"""
	Extract all relevant timing information from the current signal
	:param input_signal_set:
	:param idx:
	:param set_size:
	:param signal_iterator:
	:param t_samp:
	:param state_sample_time:
	:param input_signal:
	:return:
	"""
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
		if idx == 0:
			stimulus_duration = None
			stimulus_onset = 0.1
		else:
			stimulus_duration = None
			stimulus_onset = t_samp[idx-1]
		if idx < len(t_samp) - 1:
			if input_signal.intervals[idx]:
				simulation_time += input_signal.intervals[idx]

	return local_signal, stimulus_duration, stimulus_onset, t_samp, state_sample_time, simulation_time


def update_spike_template(enc_layer, idx, input_signal_set, stimulus_set, local_signal, t_samp, input_signal, jitter,
                          stimulus_onset):
	"""
	Read the current stimulus identity, extract the corresponding spike pattern, jitter if necessary, offset to the 
	stimulus onset time and update the spike generators
	:param enc_layer:
	:param idx:
	:param input_signal_set:
	:param stimulus_set:
	:param local_signal:
	:param t_samp:
	:param input_signal:
	:param jitter:
	:param stimulus_onset:
	:return:
	"""
	assert (len(input_signal_set.spike_patterns) == stimulus_set.dims), \
		"Incorrect number of spike patterns"

	if input_signal_set.online and local_signal is not None:
		stimulus_id = [nx for nx in range(stimulus_set.dims) if t_samp[-1] in local_signal.offset_times[nx]]
	else:
		stimulus_id = [nx for nx in range(stimulus_set.dims) if t_samp[idx] in input_signal.offset_times[nx]]
	spike_pattern = input_signal_set.spike_patterns[stimulus_id[0]].copy()

	if jitter is not None:
		if jitter[1]:  # compensate for boundary effects
			spike_pattern.jitter(jitter[0])
			resize_window = spike_pattern.time_parameters()
			spikes = spike_pattern.time_slice(resize_window[0] + jitter[0], resize_window[1] - jitter[0])
			spikes.time_offset(-jitter[0])
		else:
			spikes = spike_pattern.jitter(jitter[0])
	else:
		spikes = spike_pattern

	spikes = spikes.time_offset(stimulus_onset, True)
	enc_layer.update_state(spikes)


def update_input_signals(enc_layer, idx, stimulus_seq, local_signal, dt, noise=False, noise_parameters=None):
	"""
	Read the current signal and update the generators
	:param enc_layer:
	:param idx:
	:param stimulus_seq:
	:param local_signal:
	:param dt:
	:return:
	"""
	stim_input = coo_matrix(stimulus_seq.todense()[:, idx])
	local_signal.input_signal = local_signal.generate_single_step(stim_input)
	local_signal.time_offset(dt)
	if noise:
		assert(noise_parameters is not None), "Noise parameters must be provided!"
		local_signal.input_signal = add_noise(local_signal, noise_parameters)
	enc_layer.update_state(local_signal.input_signal)


def add_noise(local_signal, noise_parameters):
	"""
	Add a new noise realization to each step
	:param local_signal: 
	:param noise_parameters: 
	:return: 
	"""
	local_noise = input_architect.InputNoise(noise_parameters, start_time=local_signal.input_signal.t_start,
	                                         stop_time=local_signal.input_signal.t_stop+10)
	local_noise.generate()
	signal_array = local_signal.input_signal.as_array()
	noise_array = local_noise.noise_signal.as_array()[:, :signal_array.shape[1]]
	new_signal_array = signal_array + noise_array
	return signals.convert_array(new_signal_array, id_list=local_signal.input_signal.id_list(),
	                             dt=local_signal.dt, start=local_signal.input_signal.t_start, stop=local_signal.input_signal.t_stop)


def extract_state_vectors(net, enc_layer, sample_time):
	"""

	:param net:
	:param enc_layer:
	:param sample_time:
	:param store_activity:
	:return:
	"""
	# Extract and store state vectors
	for ctr, n_pop in enumerate(list(itertools.chain(*[net.merged_populations,
	                                                   net.populations, enc_layer.encoders]))):
		if n_pop.decoding_layer is not None:
			n_pop.decoding_layer.extract_state_vector(time_point=round(sample_time, 2), save=True) # TODO- choose
		# round precision


def flush(net, enc_layer, decoders=True):
	"""

	:param net:
	:param enc_layer:
	:param decoders:
	:return:
	"""
	# clear devices
	net.flush_records(decoders=decoders)
	enc_layer.flush_records(decoders=decoders)


def compile_results(net, enc_layer, t0, time_correction_factor, record, store_activity, store_decoders=False):
	"""

	:param net:
	:param enc_layer:
	:param t0:
	:param time_correction_factor:
	:param record:
	:param store_activity:
	:return:
	"""
	if record:
		# compile state matrices:
		for n_pop in list(itertools.chain(*[net.merged_populations, net.populations, enc_layer.encoders])):
			if n_pop.decoding_layer is not None:
				n_pop.decoding_layer.compile_state_matrix()
	if store_activity:
		# store full activity
		net.extract_population_activity(t_start=t0,
		                                t_stop=nest.GetKernelStatus()['time'] - time_correction_factor)
		net.extract_network_activity()
		# enc_layer.extract_encoder_activity(t_start=t0,
		#                                    t_stop=nest.GetKernelStatus()['time'] - time_correction_factor)
		if not signals.empty(net.merged_populations):
			net.merge_population_activity(start=t0,
			                              stop=nest.GetKernelStatus()['time'] - time_correction_factor,
			                              save=True)
		if store_decoders:
			for ctr, n_pop in enumerate(list(itertools.chain(*[net.merged_populations,
			                                                   net.populations]))): #, enc_layer.encoders]))):
				if n_pop.decoding_layer is not None:
					n_pop.decoding_layer.extract_activity(start=t0,
					                                      stop=nest.GetKernelStatus()['time'] - time_correction_factor,
					                                      save=True)


def time_keep(start_time, idx, set_size, t1):
	"""
	Measure current elapsed time and remaining time
	:return:
	"""
	t2 = time.time()
	total_time_elapsed = t2 - start_time
	cycle_count = idx + 1
	avg_cycle_time = total_time_elapsed / cycle_count
	cycles_remaining = set_size - cycle_count
	time_remaining = avg_cycle_time * cycles_remaining
	print("\nTime information: ")
	print("- Current step time: %.2f mins." % ((t2 - t1) / 60.))
	print("- Total elapsed time: %.2f mins." % (total_time_elapsed / 60.))
	print("- Estimated time remaining: %.2f mins." % (time_remaining / 60.))

	return ((t2 - t1) / 60.)


def process_input_sequence(parameter_set, net, enc_layer, stimulus_set, input_signal_set, set_name, record=True):
	"""
	Faster implementation - requires sampling parameters to be pre-set (set_decoder_times)
	:return:
	"""
	print("\n\n***** Preparing to simulate {0} set *****".format(set_name))

	# determine timing compensations required
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
	simulation_resolution = nest.GetKernelStatus()['resolution']

	# extract important parameters:
	sampling_times = parameter_set.decoding_pars.sampling_times
	if hasattr(parameter_set.encoding_pars.generator, "jitter"):
		jitter = parameter_set.encoding_pars.generator.jitter
	else:
		jitter = None
	if hasattr(parameter_set.input_pars, "noise") and parameter_set.input_pars.noise.N:
		signal_noise = True
		signal_noise_parameters = parameter_set.input_pars.noise
	else:
		signal_noise = False
		signal_noise_parameters = None

	# determine set to use and its properties
	labels, set_labels, set_size, input_signal, stimulus_seq, signal_iterator = retrieve_data_set(set_name,
	                                                                            stimulus_set, input_signal_set)
	# set state sampling parameters
	t_samp, sub_sampling_times = set_sampling_parameters(sampling_times, input_signal_set, input_signal)

	t0 = nest.GetKernelStatus()['time']
	epochs = {k: [] for k in labels}
	if not record:
		print("\n!!! No population activity will be stored !!!")
	start_time = time.time()
	timing = dict(step_time=[], total_time=[])

	####################################################################################################################
	if sampling_times is None:  # one sample for each stimulus (acquired at the last time point of each stimulus)
		print("\n\nSimulating {0} steps".format(str(set_size)))

		# ################################ Main Loop ###################################
		stim_idx = 0
		simulation_time = 0.0
		stimulus_onset = 0.0
		while stim_idx < set_size:
			state_sample_time = t_samp[stim_idx]

			stim_start = time.time()
			if stim_idx == 0:
				# generate first input
				local_signal, stimulus_duration, stimulus_onset, t_samp, state_sample_time, simulation_time = \
					retrieve_stimulus_timing(input_signal_set, stim_idx, set_size, signal_iterator, t_samp,
					                         state_sample_time, input_signal)
				epochs[set_labels[stim_idx]].append((stimulus_onset, state_sample_time))
				state_sample_time += encoder_delay  # correct sampling time

				print("\n\nSimulating step {0} / {1} - stimulus {2} [{3} ms]".format(str(stim_idx + 1),
					  str(set_size), str(set_labels[stim_idx]), str(simulation_time)))

				# update inputs / encoders
				if all(['spike_pattern' in n for n in list(signals.iterate_obj_list(enc_layer.generator_names))]):
					update_spike_template(enc_layer, stim_idx, input_signal_set, stimulus_set, local_signal, t_samp,
					                      input_signal, jitter, stimulus_onset)
				elif input_signal_set is not None and local_signal is not None and input_signal_set.online:
					update_input_signals(enc_layer, stim_idx, stimulus_seq, local_signal, simulation_resolution,
					                     signal_noise, signal_noise_parameters)

				# simulate main step:
				net.simulate(simulation_time)

				# generate next input
				stim_idx += 1
				local_signal, stimulus_duration, stimulus_onset, t_samp, state_sample_time, simulation_time = \
					retrieve_stimulus_timing(input_signal_set, stim_idx, set_size, signal_iterator, t_samp,
					                         state_sample_time, input_signal)

				# update inputs / encoders
				if all(['spike_pattern' in n for n in list(signals.iterate_obj_list(enc_layer.generator_names))]):
					update_spike_template(enc_layer, stim_idx, input_signal_set, stimulus_set, local_signal, t_samp,
					                      input_signal, jitter, stimulus_onset)
				elif input_signal_set is not None and local_signal is not None and input_signal_set.online:
					update_input_signals(enc_layer, stim_idx, stimulus_seq, local_signal, simulation_resolution,
					                     signal_noise, signal_noise_parameters)

				# simulate delays
				net.simulate(encoder_delay + decoder_delay)

				# reset
				analysis.reset_decoders(net, enc_layer)

				# add sample time to decoders...
				for n_pop in list(itertools.chain(*[net.merged_populations, net.populations, enc_layer.encoders])):
					if n_pop.decoding_layer is not None and sampling_times is None:
						n_pop.decoding_layer.sampled_times.append(state_sample_time)
					elif n_pop.decoding_layer is not None and (isinstance(sampling_times, list) or isinstance(
							sampling_times, np.ndarray)):
						sample_times = set(stimulus_onset + sampling_times)
						samp_times = set(n_pop.decoding_layer.sampled_times)
						n_pop.decoding_layer.sampled_times = list(np.sort(list(samp_times.union(sample_times))))

				# flush unnecessary information
				if not record:
					flush(net, enc_layer, decoders=True)
				else:
					flush(net, enc_layer, decoders=False)

			else:
				print(
					"\n\nSimulating step {0} / {1} - stimulus {2} [{3} ms]".format(str(stim_idx + 1), str(set_size),
					                                                               str(set_labels[stim_idx]),
					                                                               str(simulation_time)))
				epochs[set_labels[stim_idx]].append((stimulus_onset, state_sample_time))

				# simulate main step
				net.simulate(simulation_time - encoder_delay - decoder_delay)

				# generate next input
				stim_idx += 1
				local_signal, stimulus_duration, stimulus_onset, t_samp, state_sample_time, simulation_time = \
					retrieve_stimulus_timing(input_signal_set, stim_idx, set_size, signal_iterator, t_samp,
					                         state_sample_time, input_signal)

				# update inputs / encoders
				if stim_idx < set_size:
					if all(['spike_pattern' in n for n in list(signals.iterate_obj_list(enc_layer.generator_names))]):
						update_spike_template(enc_layer, stim_idx, input_signal_set, stimulus_set, local_signal, t_samp,
						                      input_signal, jitter, stimulus_onset)
					elif input_signal_set is not None and local_signal is not None and input_signal_set.online:
						update_input_signals(enc_layer, stim_idx, stimulus_seq, local_signal, simulation_resolution,
						                     signal_noise, signal_noise_parameters)

				# simulate delays
				net.simulate(encoder_delay + decoder_delay)

				# reset
				analysis.reset_decoders(net, enc_layer)

				# add sample time to decoders...
				for n_pop in list(itertools.chain(*[net.merged_populations, net.populations, enc_layer.encoders])):
					if n_pop.decoding_layer is not None and sampling_times is None:
						n_pop.decoding_layer.sampled_times.append(state_sample_time)
					elif n_pop.decoding_layer is not None and (isinstance(sampling_times, list) or isinstance(
							sampling_times, np.ndarray)):
						sample_times = set(stimulus_onset + sampling_times)
						samp_times = set(n_pop.decoding_layer.sampled_times)
						n_pop.decoding_layer.sampled_times = list(np.sort(list(samp_times.union(sample_times))))

				# flush unnecessary information
				if not record:
					flush(net, enc_layer, decoders=True)
				else:
					flush(net, enc_layer, decoders=False)

				if stim_idx == set_size:
					net.simulate(decoder_delay)

			timing['step_time'].append(time_keep(start_time, stim_idx, set_size, stim_start))

		timing['total_time'] = (time.time() - start_time) / 60.

		# gather states
		gather_states(net, enc_layer, t0, set_labels)

	####################################################################################################################
	# TODO alternative sampling methods
	# elif (sampling_times is not None) and (isinstance(sub_sampling_times, list) or
	# 	                                       isinstance(sub_sampling_times, np.ndarray)):  # multiple sampling
	# 	# times per stimulus (build multiple state matrices)
	# 	print("\nSimulating {0} steps".format(str(set_size)))

	return epochs, timing


# TODO this gives an error if nothing has been recorded: ex. running process_input_sequence
# TODO only for transient set, with record=False.. should handle this case gracefully
def gather_states(net, enc_layer, t0, set_labels, flush_devices=True):
	"""
	Set all the state matrices from recorded activity
	:param net:
	:param enc_layer:
	:return:
	"""
	# TODO - this is where different sampling methods are implemented or below in process_states
	for ctr, n_pop in enumerate(list(itertools.chain(*[net.merged_populations,
	                                                   net.populations, enc_layer.encoders]))):
		if n_pop.decoding_layer is not None:
			n_pop.decoding_layer.extract_activity(start=t0,
			                                      stop=nest.GetKernelStatus()['time'], save=True)
			for idx_state, n_state in enumerate(n_pop.decoding_layer.state_variables):
				n_pop.decoding_layer.state_matrix[idx_state] = n_pop.decoding_layer.activity[idx_state].as_array()
	if flush_devices:
		flush(net, enc_layer, decoders=True)


def process_states(net, enc_layer, target_matrix, stim_set, data_sets=None, accepted_idx=None,
				   evaluation_method=None, plot=False, display=True, save=False, save_paths=None):
	"""
	Post-processing step to set the correct timings of state samples, divide and re-organize dataset, ...
	:param net:
	:param enc_layer:
	:param target_matrix:
	:param stim_set:
	:param data_sets:
	:param accepted_idx:
	:param plot:
	:param display:
	:param save:
	:param save_paths:
	:return:
	"""
	results = dict(rank={}, performance={}, dimensionality={})

	if data_sets is None:
		data_sets = ["transient", "unique", "train", "test"]

	start_idx = 0
	for set_name in data_sets:
		if hasattr(stim_set, "{0}_set_labels".format(set_name)):
			labels = getattr(stim_set, "{0}_set_labels".format(set_name))
			if isinstance(labels[0], list):
				labels = list(itertools.chain(*getattr(stim_set, "{0}_set_labels".format(set_name))))
			set_start = start_idx
			set_end = len(labels) + set_start
			start_idx += len(labels)

			if accepted_idx is not None:
				accepted_ids = []
				for idx in accepted_idx:
					if set_end > idx >= set_start:
						accepted_ids.append(idx - set_start)
			else:
				accepted_ids = None

			for ctr, n_pop in enumerate(list(itertools.chain(*[net.merged_populations,
															   net.populations, enc_layer.encoders]))):
				if n_pop.decoding_layer is not None:
					dec_layer = n_pop.decoding_layer

					results['rank'].update({n_pop.name: {}})
					results['performance'].update({n_pop.name: {}})
					results['dimensionality'].update({n_pop.name: {}})

					# parse state variables
					for idx_var, var in enumerate(dec_layer.state_variables):
						state_matrix = dec_layer.state_matrix[idx_var][:, set_start:set_end]
						readouts = dec_layer.readouts[idx_var]
						target = target_matrix[:, set_start:set_end]
						if accepted_ids is not None:
							assert (len(accepted_ids) == target.shape[1]), "Incorrect {0} set labels or accepted " \
							                                               "ids".format(set_name)

						results['performance'][n_pop.name].update({var + str(idx_var): {}})
						results['dimensionality'][n_pop.name].update({var + str(idx_var): {}})

						print "\nPopulation {0}, variable {1}, set {2}: {3}".format(n_pop.name, var, set_name,
																					str(state_matrix.shape))
						if set_name == 'unique':
							results['rank'][n_pop.name].update({var + str(idx_var):
																	analysis.get_state_rank(state_matrix)})
						elif set_name == 'train':
							for readout in readouts:
								if readout.name[-1].isdigit(): # memory
									readout.set_index()
									print readout.name, readout.index

								readout.train(state_matrix, np.array(target), index=readout.index,
								              accepted=accepted_ids, display=display)

								readout.measure_stability(display=display)
								if plot and save:
									readout.plot_weights(display=display, save=save_paths['figures'] +
																			   save_paths['label'])
								elif plot:
									readout.plot_weights(display=display, save=False)

						elif set_name == 'test':
							for readout in readouts:
								print readout.name, readout.index
								output, tgt = readout.test(state_matrix, np.array(target), index=readout.index,
														   accepted=accepted_ids, display=display)

								results['performance'][n_pop.name][var + str(idx_var)].update(
									{readout.name: readout.measure_performance(tgt, output, evaluation_method,
																			   display=display)})
								results['performance'][n_pop.name][var + str(idx_var)][readout.name].update(
									{'norm_wOut': readout.norm_wout})
							results['dimensionality'][n_pop.name].update(
								{var + str(idx_var): analysis.compute_dimensionality(state_matrix)})
						if plot and set_name != 'transient':
							if save:
								analysis.analyse_state_matrix(state_matrix, labels, label=n_pop.name + var + set_name,
								                              plot=plot, display=display,
															  save=save_paths['figures']+save_paths['label'])
							else:
								analysis.analyse_state_matrix(state_matrix, labels, label=n_pop.name + var + set_name,
								                              plot=plot, display=display, save=False)
						if save and set_name != 'transient':
							np.save(save_paths['activity'] + save_paths['label'] +
							        '_population{0}_state{1}_{2}.npy'.format(n_pop.name, var, set_name), state_matrix)
	return results


