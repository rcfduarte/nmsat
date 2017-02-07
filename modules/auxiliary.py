"""
=====================================================================================
Auxiliary Module
=====================================================================================
Functions that are frequently used...

Functions:
------------

Classes:
------------

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

	print "\n\n***** Preparing to simulate {0} set *****".format(set_name)

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
	decoder_delay = max(list(itertools.chain(*decoder_delays)))
	decoder_resolution = min(list(itertools.chain(*decoder_resolutions)))
	time_correction_factor = encoder_delay + decoder_resolution
	if decoder_resolution != encoder_delay:
		print "To avoid errors in the delay compensation, it is advisable to set the output resolution to be the same " \
		      "as the encoder delays" # because the state resolution won't be enough to capture the time compensation..

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
		print("\n\n!!! All responses will be stored !!!")

	####################################################################################################################
	if sampling_times is None:  # one sample for each stimulus (acquired at the last time point of each stimulus)
		print("\nSimulating {0} steps".format(str(set_size)))

		# ################################ Main Loop ###################################
		for idx, state_sample_time in enumerate(t_samp):

			# internal time
			internal_time = nest.GetKernelStatus()['time']

			# determine simulation time for current stimulus
			local_signal, stimulus_duration, stimulus_onset, t_samp,  state_sample_time, simulation_time = \
				retrieve_stimulus_timing(input_signal_set, idx, set_size, signal_iterator, t_samp, state_sample_time, input_signal)

			if idx < set_size:
				# store and display stimulus information
				print("\nSimulating step {0} / {1} - stimulus {2} [{3} ms]".format(str(idx + 1), str(set_size), str(
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
				# net.extract_population_activity(t_start=stimulus_onset + encoder_delay, t_stop=state_sample_time)
				net.extract_population_activity(t_start=internal_time, t_stop=state_sample_time)
				net.extract_network_activity()
				enc_layer.extract_encoder_activity(t_start=stimulus_onset + encoder_delay, t_stop=state_sample_time)
				if not signals.empty(net.merged_populations):
					net.merge_population_activity(start=stimulus_onset + encoder_delay, stop=state_sample_time,
					                              save=True)
				# sample population activity
				if record:
					extract_state_vectors(net, enc_layer, state_sample_time, store_activity)

		compile_results(net, enc_layer, t0, time_correction_factor, record, store_activity)

	####################################################################################################################
	elif (sampling_times is not None) and (isinstance(sub_sampling_times, list) or isinstance(sub_sampling_times, np.ndarray)):  # multiple
		# sampling times per stimulus (build multiple state matrices)
		print("\nSimulating {0} steps".format(str(set_size)))

		# initialize state matrices
		for ctr, n_pop in enumerate(list(itertools.chain(*[net.merged_populations, net.populations,
		                                                   enc_layer.encoders]))):
			if n_pop.decoding_layer is not None:
				n_pop.decoding_layer.state_matrix = [[[] for _ in range(len(sub_sampling_times))] for _ in range(len(
					n_pop.decoding_layer.extractors))]
				n_pop.decoding_layer.state_sample_times = list(sampling_times)
			# if not signals.empty(n_pop.state_extractors) and len(n_pop.state_extractors) == 1:
			# 	n_pop.state_matrix = [[[] for _ in range(len(sub_sampling_times))]]
			# elif not signals.empty(n_pop.state_extractors) and len(n_pop.state_extractors) > 1:
			# 	n_pop.state_matrix = [[[] for _ in range(len(sub_sampling_times))] for _ in range(len(n_pop.state_extractors))]
			# if not signals.empty(n_pop.state_extractors):
			# 	## TODO currently decoding_layer doesn't have state_sample_times
			# 	n_pop.decoding_layer.state_sample_times = list(sampling_times)

		# ################################ Main Loop ###################################
		for idx, t in enumerate(t_samp):

			t_int = nest.GetKernelStatus()['time']

			if input_signal_set.online and idx < set_size:
				local_signal = signal_iterator.next()
				local_signal.time_offset(t_int)
				intervals.append(local_signal.intervals)
				t_samp.append(list(itertools.chain(*local_signal.offset_times))[0])
				t = t_samp[-1]
				if intervals[-1]:
					t += intervals[-1]
				t_sim = t - t_int
				if store_activity:
					epochs[set_labels[idx]].append((t_int, t_samp[-1]))
					# print(idx, set_labels[idx], epochs[set_labels[idx]][-1])
			else:
				local_signal = None
				if idx < len(t_samp) - 1:
					if intervals[idx]:
						t += intervals[idx]
					if store_activity:
						epochs[set_labels[idx]].append((t_int, t_samp[idx]))
						# print(idx, set_labels[idx], epochs[set_labels[idx]][-1])
				t_sim = t - t_int

			if t_sim > 0.:  # len(t_samp) <= set_size + 1 and
				print("\nSimulating step {0} / stimulus {1} [{2} ms]".format(str(idx + 1), str(set_labels[idx]),
				                                                             str(t_sim)))

				# update spike template data
				if all(['spike_pattern' in n for n in list(signals.iterate_obj_list(enc_layer.generator_names))]):
					assert (len(input_signal_set.spike_patterns) == stimulus_set.dims), "Incorrect number of spike " \
					                                                                 "patterns"
					if input_signal_set.online and local_signal is not None:
						stimulus_id = [nx for nx in range(stimulus_set.dims) if t_samp[-1] in local_signal.offset_times[
							nx]]
					else:
						stimulus_id = [nx for nx in range(stimulus_set.dims) if
						               t_samp[idx] in input_signal.offset_times[nx]]
					if t_int == 0.:
						spks = input_signal_set.spike_patterns[stimulus_id[0]].time_offset(t_int +
						            nest.GetKernelStatus()['resolution'], True)
					else:
						spks = input_signal_set.spike_patterns[stimulus_id[0]].time_offset(t_int, True)

					if jitter is not None:
						spks.jitter(jitter)
					enc_layer.update_state(spks)

				# update signal
				elif input_signal_set is not None and local_signal is not None and input_signal_set.online:
					stim_input = coo_matrix(stimulus_seq.todense()[:, idx])
					local_signal.input_signal = local_signal.generate_single_step(stim_input)

					if t_int == 0.:
						dt = nest.GetKernelStatus()['resolution']
						local_signal.input_signal.time_offset(dt)
					enc_layer.update_state(local_signal.input_signal)

				net.simulate(t_sim)
				net.extract_population_activity(t_start=t_int, t_stop=t_int+t)
				net.extract_network_activity()
				enc_layer.extract_encoder_activity(t_start=t_int, t_stop=t_int+t)
				if not signals.empty(net.merged_populations):
					net.merge_population_activity(start=t_int, stop=t_int+t, save=True)

				if record:
					#######################################################################################
					# Extract state
					# =====================================================================================
					# Extract merged responses
					for ctr, n_pop in enumerate(list(itertools.chain(*[net.merged_populations,
					                                                   net.populations,
					                                                   enc_layer.encoders]))):
						# if n_pop in net.merged_populations and idx == 0 and not n_pop.name[-1].isdigit():
						# 	n_pop.name += str(ctr)
						if n_pop.decoding_layer is not None:
							print("Collecting state matrices from Population {0}".format(str(n_pop.name)))
							for sample_idx, n_sample_time in enumerate(sub_sampling_times):
								assert (n_sample_time >= 10.), "Minimum sampling time must be >= 10 ms"
								visualization.progress_bar(float(sample_idx + 1) / float(len(sub_sampling_times)))
								sample_time = t_int + n_sample_time
								state_vectors = n_pop.decoding_layer.extract_state_vector(time_point=sample_time,
								                                                          save=False)
								for state_id, state_vec in enumerate(state_vectors):
									n_pop.decoding_layer.state_matrix[state_id][sample_idx].append(state_vec[0])

					if not store_activity:
						net.flush_records(decoders=True)
						enc_layer.flush_records(decoders=True)

		if store_activity:
			# store full activity
			net.extract_population_activity()
			net.extract_network_activity()
			enc_layer.extract_encoder_activity()
			if not signals.empty(net.merged_populations):
				net.merge_population_activity(start=t0,
				            stop=nest.GetKernelStatus()['time']-nest.GetKernelStatus()['resolution'], save=True)
			for ctr, n_pop in enumerate(list(itertools.chain(*[net.merged_populations,
			                                                   net.populations, enc_layer.encoders]))):
				if n_pop.decoding_layer is not None:
					n_pop.decoding_layer.extract_activity(start=t0,
				            stop=nest.GetKernelStatus()['time']-nest.GetKernelStatus()['resolution'], save=True)
		if record:
			# compile state matrices:
			for n_pop in list(itertools.chain(*[net.merged_populations, net.populations, enc_layer.encoders])):
				if n_pop.decoding_layer is not None:
					n_pop.decoding_layer.compile_state_matrix(sampling_times=sampling_times)

	####################################################################################################################
	elif sampling_times is not None and isinstance(t_samp, float) and not average:  # sub-sampled state (and input)
		# multiple sampling times per stimulus (build multiple state matrices)
		if not input_signal_set.online:
			t_samp = np.sort(list(signals.iterate_obj_list(input_signal.offset_times)))

		sample_every_n = int(round(t_samp ** (-1)))  # * step_size  # take one sample of activity every n steps

		if input_signal_set.online:
			print("\nSimulating {0} steps".format(str(set_size)))
		else:
			print("\nSimulating {0} ms in {1} steps".format(str(
				input_signal.input_signal.t_stop - input_signal.input_signal.t_start), str(len(t_samp))))

		# initialize state matrices
		for n_pop in list(itertools.chain(*[net.merged_populations,
		                                    net.populations, enc_layer.encoders])):
			if not signals.empty(n_pop.state_extractors):
				n_pop.state_matrix = [[] for _ in range(len(n_pop.state_extractors))]

		# ################################ Main Loop ###################################
		for idx, t in enumerate(t_samp):

			t_int = nest.GetKernelStatus()['time']

			if input_signal_set.online and idx < set_size:
				local_signal = signal_iterator.next()
				local_signal.time_offset(t_int)
				intervals.append(local_signal.intervals)
				t_samp.append(list(itertools.chain(*local_signal.offset_times))[0])
				t = t_samp[-1]
				if intervals[-1]:
					t += intervals[-1]
				t_sim = t - t_int
				if store_activity:
					epochs[set_labels[idx]].append((t_int, t_samp[-1]))
					print(idx, set_labels[idx], epochs[set_labels[idx]][-1])
			else:
				local_signal = None
				if idx < len(t_samp) - 1:
					if intervals[idx]:
						t += intervals[idx]
					if store_activity:
						epochs[set_labels[idx]].append((t_int, t_samp[idx]))
						print(idx, set_labels[idx], epochs[set_labels[idx]][-1])
				t_sim = t - t_int
			if t_sim > 0.:
				print("\nSimulating step {0} [{1} ms]".format(str(idx + 1), str(t_sim)))
				# Spike templates (need to be updated online)
				if np.mean(['spike_pattern' in n for n in list(signals.iterate_obj_list(enc_layer.generator_names))]) == 1. \
						and \
								input_signal_set is not None:
					assert (len(input_signal_set.spike_patterns) == stimulus_set.dims), "Incorrect number of spike patterns"
					if input_signal_set.online and local_signal is not None:
						stimulus_id = [nx for nx in range(stimulus_set.dims) if t_samp[-1] in local_signal.offset_times[
							nx]]
					else:
						stimulus_id = [nx for nx in range(stimulus_set.dims) if
						               t_samp[idx] in input_signal.offset_times[nx]]
					if t_int == 0.:
						spks = input_signal_set.spike_patterns[stimulus_id[0]].time_offset(t_int + nest.GetKernelStatus()[
							'resolution'], True)
					else:
						spks = input_signal_set.spike_patterns[stimulus_id[0]].time_offset(t_int, True)
					if jitter is not None:
						spks.jitter(jitter)
					enc_layer.update_state(spks)

				elif input_signal_set is not None and local_signal is not None and input_signal_set.online:
					stim_input = coo_matrix(stimulus_seq.todense()[:, idx])
					local_signal.input_signal = local_signal.generate_single_step(stim_input)

					if t_int == 0.:
						dt = nest.GetKernelStatus()['resolution']
						local_signal.input_signal.time_offset(dt)
					enc_layer.update_state(local_signal.input_signal)

				net.simulate(t_sim)
				net.extract_population_activity()
				net.extract_network_activity()
				enc_layer.extract_encoder_activity()

				if record:
					# Extract state
					for ctr, n_pop in enumerate(list(itertools.chain(*[net.merged_populations,
					                                                   net.populations, enc_layer.encoders]))):
						if n_pop in net.merged_populations and idx == 0 and not n_pop.name[-1].isdigit():
							n_pop.name += str(ctr)
						if not signals.empty(n_pop.state_extractors):
							print("Collecting response samples from Population {0} [rate = {1}]".format(str(
								n_pop.name), str(t_samp)))
							responses = n_pop.extract_response_matrix(start=t_int, stop=t, save=False)
							for response_idx, n_response in enumerate(responses):
								if store_activity:
									np.save(store_activity + n_pop.name + '-StateVar{0}-Response{1}.npy'.format(str(
										response_idx), str(idx)), n_response)
								if idx == 0:
									n_pop.state_matrix[response_idx] = n_response.as_array()[:, ::sample_every_n]
								else:
									n_pop.state_matrix[response_idx] = np.append(n_pop.state_matrix[response_idx],
									                                             n_response.as_array()[:,
									                                             ::sample_every_n], axis=1)
					# clear devices
					if not store_activity:
						net.flush_records(decoders=True)
						enc_layer.flush_records(decoders=True)

	####################################################################################################################
	elif sampling_times is not None and isinstance(t_samp, float) and average:  # sub-sampled state (and input)
		# multiple sampling times per stimulus (build multiple state matrices), state vector = average over stimulus
		# presentation time
		if not input_signal_set.online:
			t_samp = np.sort(list(signals.iterate_obj_list(input_signal.offset_times)))

		sample_every_n = int(round(t_samp ** (-1)))  # * step_size  # take one sample of activity every n steps

		if input_signal_set.online:
			print("\nSimulating {0} steps".format(str(set_size)))
		else:
			print("\nSimulating {0} ms in {1} steps".format(str(
				input_signal.input_signal.t_stop - input_signal.input_signal.t_start), str(len(t_samp))))

		# initialize state matrices
		for n_pop in list(itertools.chain(*[net.merged_populations,
		                                    net.populations, enc_layer.encoders])):
			if not signals.empty(n_pop.state_extractors):
				n_pop.state_matrix = [[] for _ in range(len(n_pop.state_extractors))]
		inter_stim_int = 0

		# ################################ Main Loop ###################################
		for idx, t in enumerate(t_samp):

			t_int = nest.GetKernelStatus()['time']

			if input_signal_set.online and idx < set_size:
				local_signal = signal_iterator.next()
				local_signal.time_offset(t_int)
				intervals.append(local_signal.intervals)
				t_samp.append(list(itertools.chain(*local_signal.offset_times))[0])
				t = t_samp[-1]
				if intervals[-1]:
					t += intervals[-1]
					inter_stim_int = intervals[-1]
				t_sim = t - t_int
				if store_activity:
					epochs[set_labels[idx]].append((t_int, t_samp[-1]))
					print(idx, set_labels[idx], epochs[set_labels[idx]][-1])
			else:
				local_signal = None
				if idx < len(t_samp) - 1:
					if intervals[idx]:
						t += intervals[idx]
						inter_stim_int = intervals[idx]
					if store_activity:
						epochs[set_labels[idx]].append((t_int, t_samp[idx]))
						print(idx, set_labels[idx], epochs[set_labels[idx]][-1])
				t_sim = t - t_int
			if t_sim > 0.:
				print("\nSimulating step {0} [{1} ms]".format(str(idx + 1), str(t_sim)))
				# Spike templates (need to be updated online)
				if np.mean(['spike_pattern' in n for n in list(signals.iterate_obj_list(enc_layer.generator_names))]) == 1. \
						and \
								input_signal_set is not None:
					assert (len(input_signal_set.spike_patterns) == stimulus_set.dims), "Incorrect number of spike patterns"
					if input_signal_set.online and local_signal is not None:
						stimulus_id = [nx for nx in range(stimulus_set.dims) if t_samp[-1] in local_signal.offset_times[
							nx]]
					else:
						stimulus_id = [nx for nx in range(stimulus_set.dims) if
						               t_samp[idx] in input_signal.offset_times[nx]]
					if t_int == 0.:
						spks = input_signal_set.spike_patterns[stimulus_id[0]].time_offset(t_int + nest.GetKernelStatus()[
							'resolution'], True)
					else:
						spks = input_signal_set.spike_patterns[stimulus_id[0]].time_offset(t_int, True)
					if jitter is not None:
						spks.jitter(jitter)
					enc_layer.update_state(spks)

				elif input_signal_set is not None and local_signal is not None and input_signal_set.online:
					stim_input = coo_matrix(stimulus_seq.todense()[:, idx])
					local_signal.input_signal = local_signal.generate_single_step(stim_input)

					if t_int == 0.:
						dt = nest.GetKernelStatus()['resolution']
						local_signal.input_signal.time_offset(dt)
					enc_layer.update_state(local_signal.input_signal)

				net.simulate(t_sim)
				net.extract_population_activity()
				net.extract_network_activity()
				enc_layer.extract_encoder_activity()

				if record:
					# Extract state
					for ctr, n_pop in enumerate(list(itertools.chain(*[net.merged_populations,
					                                                   net.populations, enc_layer.encoders]))):
						if n_pop in net.merged_populations and idx == 0 and not n_pop.name[-1].isdigit():
							n_pop.name += str(ctr)
						if not signals.empty(n_pop.state_extractors):
							print("Collecting response samples from Population {0} [rate = {1}]".format(str(
								n_pop.name), str(sub_sampling_times)))
							responses = n_pop.extract_response_matrix(start=t_int, stop=t - inter_stim_int, save=False)
							for response_idx, n_response in enumerate(responses):
								if store_activity:
									np.save(
										store_activity + n_pop.name + '-StateVar{0}-Response{1}.npy'.format(str(
											response_idx), str(idx)), n_response)
								if idx == 0:
									subsampled_states = n_response.as_array()[:, ::sample_every_n]
									n_pop.state_matrix[response_idx] = np.array([np.mean(subsampled_states, 1)]).T
								else:
									subsampled_states = n_response.as_array()[:, ::sample_every_n]
									state_vec = np.array([np.mean(subsampled_states, 1)]).T
									n_pop.state_matrix[response_idx] = np.append(n_pop.state_matrix[response_idx],
									                                             state_vec,
									                                             axis=1)
					# clear devices
					if not store_activity:
						net.flush_records(decoders=True)
						enc_layer.flush_records(decoders=True)
	else:
		raise NotImplementedError("Specify sampling times as None (last step sample), list or array of times, or float")

	if store_activity:
		return epochs


def retrieve_data_set(set_name, stimulus_set, input_signal_set):
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
		stimulus_duration = None
		stimulus_onset = None
		if idx < len(t_samp) - 1:
			if input_signal.intervals[idx]:
				simulation_time += input_signal.intervals[idx]

	return local_signal, stimulus_duration, stimulus_onset, t_samp, state_sample_time, simulation_time


def update_spike_template(enc_layer, idx, input_signal_set, stimulus_set, local_signal, t_samp, input_signal, jitter,
                          stimulus_onset):
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
	enc_layer.update_state(spks)


def update_input_signals(enc_layer, idx, stimulus_seq, local_signal, decoder_resolution):
	stim_input = coo_matrix(stimulus_seq.todense()[:, idx])
	local_signal.input_signal = local_signal.generate_single_step(stim_input)
	local_signal.time_offset(decoder_resolution)
	enc_layer.update_state(local_signal.input_signal)


def extract_state_vectors(net, enc_layer, sample_time, store_activity):
	# Extract and store state vectors
	for ctr, n_pop in enumerate(list(itertools.chain(*[net.merged_populations,
	                                                   net.populations, enc_layer.encoders]))):
		if n_pop.decoding_layer is not None:
			n_pop.decoding_layer.extract_state_vector(time_point=round(sample_time, 2), save=True)
	# clear devices
	if not store_activity:
		net.flush_records(decoders=True)
		enc_layer.flush_records(decoders=True)


def compile_results(net, enc_layer, t0, time_correction_factor, record, store_activity):
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
		for ctr, n_pop in enumerate(list(itertools.chain(*[net.merged_populations,
		                                                   net.populations]))): #, enc_layer.encoders]))):
			if n_pop.decoding_layer is not None:
				n_pop.decoding_layer.extract_activity(start=t0,
				                                      stop=nest.GetKernelStatus()['time'] - time_correction_factor,
				                                      save=True)


# def process_states(net, enc_layer, parameters, stim, inputs, set_name, target=None):
#
# 	target_matrix = stim.full_set.todense()
# 	for ctr, n_pop in enumerate(list(itertools.chain(*[net.merged_populations,
# 	                                                   net.populations]))):  # , enc_layer.encoders]))):
# 		if n_pop.decoding_layer is not None:
# 			dec_layer = n_pop.decoding_layer
# 			if store_activity and debug:
# 				dec_layer.evaluate_decoding(n_neurons=10, display=display, save=paths['figures'] + paths['label'])
#
# 			results['rank'].update({n_pop.name: {}})
# 			results['performance'].update({n_pop.name: {}})
#
# 			# parse state variables
# 			for idx_var, var in enumerate(dec_layer.state_variables):
# 				results['performance'][n_pop.name].update({var: {}})
# 				time_steps = 0
# 				end_step = 0
# 				state_matrix = dec_layer.state_matrix[idx_var]
# 				readouts = dec_layer.readouts[idx_var]
#
# 				labels = getattr(stim, "{0}_set_labels".format(stim_set))
# 				if not empty(labels) and not empty(state_matrix):
# 					print "Population {0}, variable {1}, set {2}: {3}".format(n_pop.name, var, stim_set,
# 						                                                          str(state.shape))
# 					target = target_matrix[:, time_steps:end_step]
# 						time_steps += len(labels)
# 						if stim_set == 'unique':
# 							results['rank'][n_pop.name].update({var + str(idx_var): get_state_rank(state)})
# 						elif stim_set == 'train':
# 							for readout in readouts:
# 								readout_train(readout, state, target=np.array(target), index=None, accepted=None,
# 								              display=display, plot=plot, save=paths['figures'] + paths['label'])
# 						elif stim_set == 'test':
# 							for readout in readouts:
# 								results['performance'][n_pop.name][var].update(
# 									{readout.name: readout_test(readout, state,
# 									                            target=np.array(target), index=None, accepted=None,
# 									                            display=display)})
# 						if plot:
# 							analyse_state_matrix(state_matrix, stim.full_set_labels, label=n_pop.name + var + stim_set,
# 							                     plot=plot, display=display, save=paths['figures'] + paths['label'])