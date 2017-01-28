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
import numpy as np
import itertools
import nest


########################################################################################################################
# Auxiliary Functions
########################################################################################################################
def iterate_input_sequence(net, enc_layer, parameter_set, stimulus_set, input_signal_set,
                           set_name=None, record=True, store_activity=False):
	"""
	Run simulation sequentiallu, presenting one input stimulus at a time (if input is a discrete stimulus sequence),
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
	:param jitter: if input is spike pattern, add jitter
	:param average: [bool] - if True, state vector is average of subsampled activity vectors
	"""
	assert (isinstance(net, net_architect.Network)), "Please provide a Network object"
	assert (isinstance(enc_layer, input_architect.EncodingLayer)), "Please provide an EncodingLayer object"
	assert (isinstance(input_signal_set, input_architect.InputSignalSet)), "input_set must be InputSignalSet"
	assert (isinstance(parameter_set, parameters.ParameterSet)), "incorrect ParameterSet"
	assert (isinstance(stimulus_set, input_architect.StimulusSet)), "incorrect ParameterSet"

	# extract important parameters:
	sampling_times = parameter_set.decoding_pars.sampling_times
	if hasattr(parameter_set.encoding_pars.generator, "jitter"):
		jitter = parameter_set.encoding_pars.generator.jitter
	else:
		jitter = None

	# determine set to use and its properties
	if set_name is None:
		set_name = 'full'
	current_set = set_name
	all_labels = getattr(stimulus_set, '{0}_set_labels'.format(current_set))
	if isinstance(all_labels[0], list):
		labels = np.unique(list(itertools.chain(*all_labels)))
		set_labels = list(itertools.chain(*all_labels))
	else:
		labels = np.unique(all_labels)
		set_labels = all_labels
	set_size = len(set_labels)
	input_signal = getattr(input_signal_set, "{0}_set_signal".format(current_set))
	stimulus_seq = getattr(stimulus_set, '{0}_set'.format(current_set))
	if input_signal_set.online:
		signal_iterator = getattr(input_signal_set, '{0}_set_signal_iterator'.format(current_set))
		intervals = [0]
	else:
		signal_iterator = None
		intervals = input_signal.intervals

	# set state sampling parameters - TODO!! why t_step and t_samp?
	if sampling_times is None and not input_signal_set.online:
		t_samp = np.sort(list(signals.iterate_obj_list(input_signal.offset_times)))
	elif sampling_times is None and input_signal_set.online:
		t_samp = [0]
	elif sampling_times is not None and input_signal_set.online:
		t_samp = sampling_times
		t_step = [0]
	else:
		t_samp = sampling_times

	# if not input_signal_set.online:
	# 	intervals = input_signal.intervals
	# 	set_size = len(list(signals.iterate_obj_list(input_signal.amplitudes)))
	# 	set_names = ['train', 'test', 'transient', 'unique', 'full']
	# 	set_sizes = {k: 0 for k in set_names}
	# 	for k, v in stimulus_set.__dict__.items():
	# 		for set_name in set_names:
	# 			if k == '{0}_set_labels'.format(set_name):
	# 				set_sizes['{0}'.format(set_name)] = len(v)
	# 	current_set = [key for key, value in set_sizes.items() if set_size == value][0]
	# else:
	# 	assert (set_name is not None), "set_name needs to be provided in online mode.."
	# 	current_set = set_name
	# 	set_size = len(getattr(stimulus_set, '{0}_set_labels'.format(current_set)))
	# 	signal_iterator = getattr(input_signal_set, '{0}_set_signal_iterator'.format(current_set))
	# 	stimulus_seq = getattr(stimulus_set, '{0}_set'.format(current_set))
	# 	intervals = [0]  # ?
	if not record:
		print("\n!!! No population activity will be stored !!!")
	if store_activity:
		print("\n\n!!! All responses will be stored !!!")
		epochs = {k: [] for k in labels}
		t0 = nest.GetKernelStatus()['time']
	####################################################################################################################
	if sampling_times is None:  # one sample for each stimulus (acquired at the last time point of each stimulus)
		print("\nSimulating {0} steps".format(str(set_size)))

		# ################################ Main Loop ###################################
		for idx, t in enumerate(t_samp):

			# internal time
			t_int = nest.GetKernelStatus()['time']

			# determine simulation time for current stimulus
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

			if len(t_samp) <= set_size + 1 and t_sim > 0.:
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
						## spike times cannot be zero
						spks = input_signal_set.spike_patterns[stimulus_id[0]].time_offset(t_int +
						            nest.GetKernelStatus()['resolution'], True)
					else:
						spks = input_signal_set.spike_patterns[stimulus_id[0]].time_offset(t_int, True)

					if jitter is not None:
						print jitter
						spks.jitter(jitter) # TODO ...
					enc_layer.update_state(spks)

				# update signal
				elif input_signal_set is not None and local_signal is not None and input_signal_set.online:
					stim_input = coo_matrix(stimulus_seq.todense()[:, idx])
					local_signal.input_signal = local_signal.generate_single_step(stim_input)

					if t_int == 0.:
						## !!! TEST - device update times cannot be 0.
						dt = nest.GetKernelStatus()['resolution']
						local_signal.input_signal.time_offset(dt)
					enc_layer.update_state(local_signal.input_signal)

				# simulate and extract recordings from devices
				net.simulate(t_sim)# + nest.GetKernelStatus()['resolution'])
				net.extract_population_activity(t_start=t_int, t_stop=t_int+t)
				net.extract_network_activity()
				enc_layer.extract_encoder_activity(t_start=t_int, t_stop=t_int+t)
				if not signals.empty(net.merged_populations):
					net.merge_population_activity(start=t_int, stop=t_int+t, save=True)

				if record:
					# Extract and store state vectors
					for ctr, n_pop in enumerate(list(itertools.chain(*[net.merged_populations,
					                                                   net.populations, enc_layer.encoders]))):
						if n_pop.decoding_layer is not None:
							n_pop.decoding_layer.extract_state_vector(time_point=t, save=True, reset=False) ### !!!
							n_pop.decoding_layer.reset_states()
					# clear devices
					if not store_activity:
						net.flush_records(decoders=True)
						enc_layer.flush_records(decoders=True)

		if store_activity:
			# store full activity
			net.extract_population_activity(t_start=t0, t_stop=nest.GetKernelStatus()['time']-nest.GetKernelStatus()['resolution'])
			net.extract_network_activity()
			enc_layer.extract_encoder_activity(t_start=t0, t_stop=nest.GetKernelStatus()['time']-nest.GetKernelStatus()['resolution'])
			if not signals.empty(net.merged_populations):
				net.merge_population_activity(start=t0,
				            stop=nest.GetKernelStatus()['time']-nest.GetKernelStatus()['resolution'], save=True)
			for ctr, n_pop in enumerate(list(itertools.chain(*[net.merged_populations,
			                                                   net.populations, enc_layer.encoders]))):
				if n_pop.decoding_layer is not None:
					n_pop.decoding_layer.extract_activity(save=True)
		if record:
			# compile state matrices:
			for n_pop in list(itertools.chain(*[net.merged_populations, net.populations, enc_layer.encoders])):
				if n_pop.decoding_layer is not None:
					n_pop.decoding_layer.compile_state_matrix()

	####################################################################################################################
	elif (sampling_times is not None) and (isinstance(t_samp, list) or isinstance(t_samp, np.ndarray)):  # multiple
		# sampling times per stimulus (build multiple state matrices)
		print("\nSimulating {0} steps".format(str(set_size)))

		# initialize state matrices
		for ctr, n_pop in enumerate(list(itertools.chain(*[net.merged_populations, net.populations,
		                                                   enc_layer.encoders]))):
			if n_pop.decoding_layer is not None:
				n_pop.decoding_layer.state_matrix = [[[] for _ in range(len(t_samp))] for _ in range(len(
					n_pop.decoding_layer.extractors))]
				n_pop.decoding_layer.state_sample_times = list(sampling_times)
			# if not signals.empty(n_pop.state_extractors) and len(n_pop.state_extractors) == 1:
			# 	n_pop.state_matrix = [[[] for _ in range(len(t_samp))]]
			# elif not signals.empty(n_pop.state_extractors) and len(n_pop.state_extractors) > 1:
			# 	n_pop.state_matrix = [[[] for _ in range(len(t_samp))] for _ in range(len(n_pop.state_extractors))]
			# if not signals.empty(n_pop.state_extractors):
			# 	## TODO currently decoding_layer doesn't have state_sample_times
			# 	n_pop.decoding_layer.state_sample_times = list(sampling_times)

		# ################################ Main Loop ###################################
		for idx, t in enumerate(t_step):

			t_int = nest.GetKernelStatus()['time']

			if input_signal_set.online and idx < set_size:
				local_signal = signal_iterator.next()
				local_signal.time_offset(t_int)
				intervals.append(local_signal.intervals)
				t_step.append(list(itertools.chain(*local_signal.offset_times))[0])
				t = t_step[-1]
				if intervals[-1]:
					t += intervals[-1]
				t_sim = t - t_int
				if store_activity:
					epochs[set_labels[idx]].append((t_int, t_samp[-1]))
					# print(idx, set_labels[idx], epochs[set_labels[idx]][-1])
			else:
				local_signal = None
				if idx < len(t_step) - 1:
					if intervals[idx]:
						t += intervals[idx]
					if store_activity:
						epochs[set_labels[idx]].append((t_int, t_samp[idx]))
						# print(idx, set_labels[idx], epochs[set_labels[idx]][-1])
				t_sim = t - t_int

			if t_sim > 0.:  # len(t_step) <= set_size + 1 and
				print("\nSimulating step {0} / stimulus {1} [{2} ms]".format(str(idx + 1), str(set_labels[idx]),
				                                                             str(t_sim)))

				# update spike template data
				if all(['spike_pattern' in n for n in list(signals.iterate_obj_list(enc_layer.generator_names))]):
					assert (len(input_signal_set.spike_patterns) == stimulus_set.dims), "Incorrect number of spike " \
					                                                                 "patterns"
					if input_signal_set.online and local_signal is not None:
						stimulus_id = [nx for nx in range(stimulus_set.dims) if t_step[-1] in local_signal.offset_times[
							nx]]
					else:
						stimulus_id = [nx for nx in range(stimulus_set.dims) if
						               t_step[idx] in input_signal.offset_times[nx]]
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
							for sample_idx, n_sample_time in enumerate(t_samp):
								assert (n_sample_time >= 10.), "Minimum sampling time must be >= 10 ms"
								visualization.progress_bar(float(sample_idx + 1) / float(len(t_samp)))
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
			t_step = np.sort(list(signals.iterate_obj_list(input_signal.offset_times)))

		sample_every_n = int(round(t_samp ** (-1)))  # * step_size  # take one sample of activity every n steps

		if input_signal_set.online:
			print("\nSimulating {0} steps".format(str(set_size)))
		else:
			print("\nSimulating {0} ms in {1} steps".format(str(
				input_signal.input_signal.t_stop - input_signal.input_signal.t_start), str(len(t_step))))

		# initialize state matrices
		for n_pop in list(itertools.chain(*[net.merged_populations,
		                                    net.populations, enc_layer.encoders])):
			if not signals.empty(n_pop.state_extractors):
				n_pop.state_matrix = [[] for _ in range(len(n_pop.state_extractors))]

		# ################################ Main Loop ###################################
		for idx, t in enumerate(t_step):

			t_int = nest.GetKernelStatus()['time']

			if input_signal_set.online and idx < set_size:
				local_signal = signal_iterator.next()
				local_signal.time_offset(t_int)
				intervals.append(local_signal.intervals)
				t_step.append(list(itertools.chain(*local_signal.offset_times))[0])
				t = t_step[-1]
				if intervals[-1]:
					t += intervals[-1]
				t_sim = t - t_int
				if store_activity:
					epochs[set_labels[idx]].append((t_int, t_samp[-1]))
					print(idx, set_labels[idx], epochs[set_labels[idx]][-1])
			else:
				local_signal = None
				if idx < len(t_step) - 1:
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
						stimulus_id = [nx for nx in range(stimulus_set.dims) if t_step[-1] in local_signal.offset_times[
							nx]]
					else:
						stimulus_id = [nx for nx in range(stimulus_set.dims) if
						               t_step[idx] in input_signal.offset_times[nx]]
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
			t_step = np.sort(list(signals.iterate_obj_list(input_signal.offset_times)))

		sample_every_n = int(round(t_samp ** (-1)))  # * step_size  # take one sample of activity every n steps

		if input_signal_set.online:
			print("\nSimulating {0} steps".format(str(set_size)))
		else:
			print("\nSimulating {0} ms in {1} steps".format(str(
				input_signal.input_signal.t_stop - input_signal.input_signal.t_start), str(len(t_step))))

		# initialize state matrices
		for n_pop in list(itertools.chain(*[net.merged_populations,
		                                    net.populations, enc_layer.encoders])):
			if not signals.empty(n_pop.state_extractors):
				n_pop.state_matrix = [[] for _ in range(len(n_pop.state_extractors))]
		inter_stim_int = 0

		# ################################ Main Loop ###################################
		for idx, t in enumerate(t_step):

			t_int = nest.GetKernelStatus()['time']

			if input_signal_set.online and idx < set_size:
				local_signal = signal_iterator.next()
				local_signal.time_offset(t_int)
				intervals.append(local_signal.intervals)
				t_step.append(list(itertools.chain(*local_signal.offset_times))[0])
				t = t_step[-1]
				if intervals[-1]:
					t += intervals[-1]
					inter_stim_int = intervals[-1]
				t_sim = t - t_int
				if store_activity:
					epochs[set_labels[idx]].append((t_int, t_samp[-1]))
					print(idx, set_labels[idx], epochs[set_labels[idx]][-1])
			else:
				local_signal = None
				if idx < len(t_step) - 1:
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
						stimulus_id = [nx for nx in range(stimulus_set.dims) if t_step[-1] in local_signal.offset_times[
							nx]]
					else:
						stimulus_id = [nx for nx in range(stimulus_set.dims) if
						               t_step[idx] in input_signal.offset_times[nx]]
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
