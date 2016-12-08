import sys
sys.path.append("../../")
import cPickle as pickle
import numpy as np
import scipy.optimize as opt
import scipy.integrate as integ
import scipy.stats as st
from scipy.sparse import coo_matrix
import os
import nest
from modules.parameters import *
from modules.net_architect import *
from modules.input_architect import *
from modules.analysis import *
from modules.visualization import progress_bar
import pylab as pl

########################################################################################################################
# Auxiliary Functions
########################################################################################################################
def iterate_input_sequence(network_obj, input_signal, enc_layer, sampling_times=None, stim_set=None,
                           input_set=None, set_name=None, record=True,
                           store_responses=False, jitter=None, average=False):
	"""
	Run simulation iteratively, presenting one input stimulus at a time..
	:param network_obj: Network object
	:param input_signal: InputSignal object
	:param enc_layer: EncodingLayer
	:param sampling_times:
	:param stim_set: full StimulusSet object
	:param input_set: full InputSignalSet object
	:param set_name: string with the name of the current set ('transient', 'unique', 'train', 'test')
	:param record: [bool] - acquire state matrix (according to sampling_times)
	:param store_responses: [bool] - record entire population activity (memory!)
	:param jitter: if input is spike pattern, add jitter
	:param average: [bool] - if True, state vector is average of subsampled activity vectors
	"""
	if not (isinstance(network_obj, Network)):
		raise TypeError("Please provide a Network object")
	if not isinstance(enc_layer, EncodingLayer):
		raise TypeError("Please provide an EncodingLayer object")
	if input_set is not None and not isinstance(input_set, InputSignalSet):
		raise TypeError("input_set must be InputSignalSet")

	sampling_lag = 2.
	if sampling_times is None and not input_set.online:
		t_samp = np.sort(list(iterate_obj_list(input_signal.offset_times)))
	elif sampling_times is None and input_set.online:
		t_samp = [0]
	elif sampling_times is not None and input_set.online:
		t_samp = sampling_times
		t_step = [0]
	else:
		t_samp = sampling_times

	if not input_set.online:
		intervals = input_signal.intervals
		set_size = len(list(iterate_obj_list(input_signal.amplitudes)))
		set_names = ['train', 'test', 'transient', 'unique', 'full']
		set_sizes = {k: 0 for k in set_names}
		for k, v in stim_set.__dict__.items():
			for set_name in set_names:
				if k == '{0}_set_labels'.format(set_name):
					set_sizes['{0}'.format(set_name)] = len(v)
		current_set = [key for key, value in set_sizes.items() if set_size == value][0]
	else:
		assert(set_name is not None), "set_name needs to be provided in online mode.."
		current_set 	= set_name
		set_size 		= len(getattr(stim_set, '{0}_set_labels'.format(current_set)))
		signal_iterator = getattr(input_set, '{0}_set_signal_iterator'.format(current_set))
		stimulus_seq 	= getattr(stim_set, '{0}_set'.format(current_set))
		intervals 		= [0] #?

	if store_responses:
		print "\n\n!!! All responses will be stored !!!"
		labels = np.unique(getattr(stim_set, '{0}_set_labels'.format(current_set))				   )
		# set_labels = getattr(stim_set, '{0}_set_labels'.format(current_set))

		tmp_set_labels = getattr(stim_set, '{0}_set_labels'.format(current_set))
		set_labels = [label if not isinstance(label, list) else label[0] for label in tmp_set_labels]
		print "set_labels: " + str(set_labels)
		epochs = {k: [] for k in labels}

	####################################################################################################################
	if sampling_times is None:  # one sample for each stimulus Question can we simulate x seconds within one sample?
		if input_set.online:
			print "\nSimulating {0} steps".format(str(set_size))
		else:
			print "\nSimulating {0} ms in {1} steps".format(str(
				input_signal.input_signal.t_stop-input_signal.input_signal.t_start), str(set_size))

		# ################################ Main Loop ###################################
		for idx, t in enumerate(t_samp):
			t_int = nest.GetKernelStatus()['time']

			if input_set.online and idx < set_size:
				local_signal = signal_iterator.next()
				local_signal.time_offset(t_int)
				intervals.append(local_signal.intervals)
				t_samp.append(list(itertools.chain(*local_signal.offset_times))[0])
				t = t_samp[-1]
				if intervals[-1]:
					t += intervals[-1]
				t_sim = t - t_int
				if store_responses:
					print "###shyte: " + str(idx) + ", type: " + str(type(idx))
					print "###shyte 2: " + str(set_labels[idx])
					epochs[set_labels[idx]].append((t_int, t_samp[-1]))
					print idx, set_labels[idx], epochs[set_labels[idx]][-1]
			else:
				local_signal = None
				if idx < len(t_samp) - 1:
					if intervals[idx]:
						t += intervals[idx]
					if store_responses:
						epochs[set_labels[idx]].append((t_int, t_samp[idx]))
						print idx, set_labels[idx], epochs[set_labels[idx]][-1]
				t_sim = t - t_int

			if store_responses and input_set.online and idx < set_size:
				epochs[set_labels[idx]].append((t_int, t_samp[-1]))
				print idx, set_labels[idx], epochs[set_labels[idx]][-1]
				print (t_int, t_samp[-1])
			elif store_responses and input_set.online:
				# TODO this is a bug, idx can be == len(set_labels), causing an IndexError
				# Question: what to do here?
				epochs[set_labels[idx]].append((t_int, t))
				print (t_int, t)
			elif store_responses:
					epochs[set_labels[idx]].append((t_int, t-intervals[idx]))
					print epochs[set_labels[idx]][-1]
			if len(t_samp) <= set_size + 1 and t_sim > 0.:
				print "\nSimulating step {0} [{1} ms]".format(str(idx + 1), str(t_sim))

				# Spike templates (need to be updated online)
				if np.mean(['spike_pattern' in n for n in list(iterate_obj_list(enc_layer.generator_names))]) == 1. and \
						input_set is not None:
					assert(len(input_set.spike_patterns) == stim_set.dims), "Incorrect number of spike patterns"
					if input_set.online and local_signal is not None:
						stimulus_id = [nx for nx in range(stim_set.dims) if t_samp[-1] in local_signal.offset_times[nx]]
					else:
						stimulus_id = [nx for nx in range(stim_set.dims) if t_samp[idx] in input_signal.offset_times[nx]]
					if t_int == 0.:
						spks = input_set.spike_patterns[stimulus_id[0]].time_offset(
							t_int + nest.GetKernelStatus()['resolution'], True)
					else:
						spks = input_set.spike_patterns[stimulus_id[0]].time_offset(t_int, True)
					if jitter is not None:
						spks.jitter(jitter)

					enc_layer.update_state(spks)

				elif input_set is not None and local_signal is not None and input_set.online:
					stim_input = coo_matrix(stimulus_seq.todense()[:, idx])
					local_signal.input_signal = local_signal.generate_single_step(stim_input)

					if t_int == 0.:
						dt = nest.GetKernelStatus()['resolution']
						local_signal.input_signal.time_offset(dt)
					enc_layer.update_state(local_signal.input_signal)

				network_obj.simulate(t_sim)
				network_obj.extract_population_activity()
				network_obj.extract_network_activity()

				if record:
					#######################################################################################
					# Extract state
					# =====================================================================================
					# Extract merged responses
					if not empty(network_obj.merged_populations):
						for ctr, n_pop in enumerate(network_obj.merged_populations):
							n_pop.extract_state_vector(time_point=t, lag=sampling_lag, save=True)
							if idx == 0 and not n_pop.name[-1].isdigit():
								n_pop.name += str(ctr)
					# Extract from populations
					if not empty(network_obj.state_extractors):
						for n_pop in network_obj.populations:
							n_pop.extract_state_vector(time_point=t, lag=sampling_lag, save=True)

					if not store_responses:
						network_obj.flush_records(decoders=True)
					else:
						epochs.update({})

					enc_layer.extract_encoder_activity()

					if not empty(enc_layer.encoders) and not empty(enc_layer.state_extractors):
						for ctr, n_enc in enumerate(enc_layer.encoders):
							n_enc.extract_state_vector(time_point=t, lag=sampling_lag, save=True)

						if not store_responses:
							enc_layer.flush_records(decoders=True)
		if record:
			# compile matrices:
			if not empty(network_obj.merged_populations):
				for ctr, n_pop in enumerate(network_obj.merged_populations):
					if not empty(n_pop.state_matrix):
						n_pop.compile_state_matrix()
			if not empty(network_obj.state_extractors):
				for n_pop in network_obj.populations:
					if not empty(n_pop.state_matrix):
						n_pop.compile_state_matrix()
			if not empty(enc_layer.encoders) and not empty(enc_layer.state_extractors):
				for ctr, n_enc in enumerate(enc_layer.encoders):
					n_enc.compile_state_matrix()

	####################################################################################################################
	elif (sampling_times is not None) and (isinstance(t_samp, list) or isinstance(t_samp, np.ndarray)): # multiple
		# sampling times per stimulus (build multiple state matrices)
		if not input_set.online:
			t_step = np.sort(list(iterate_obj_list(input_signal.offset_times)))
		if input_set.online:
			print "\nSimulating {0} steps".format(str(set_size))
		else:
			print "\nSimulating {0} ms in {1} steps".format(str(
				input_signal.input_signal.t_stop-input_signal.input_signal.t_start), str(set_size))

		# initialize state matrices
		if not empty(network_obj.merged_populations):
			for ctr, n_pop in enumerate(network_obj.merged_populations):
				if not empty(n_pop.state_extractors) and len(n_pop.state_extractors) == 1:
					n_pop.state_matrix = [[[] for _ in range(len(t_samp))]]
				elif not empty(n_pop.state_extractors) and len(n_pop.state_extractors) > 1:
					n_pop.state_matrix = [[[] for _ in range(len(t_samp))] for _ in range(len(n_pop.state_extractors))]
				if not empty(n_pop.state_extractors):
					n_pop.state_sample_times = list(sampling_times)
		if not empty(network_obj.state_extractors):
			for n_pop in network_obj.populations:
				if not empty(n_pop.state_extractors) and len(n_pop.state_extractors) == 1:
					n_pop.state_matrix = [[[] for _ in range(len(t_samp))]]
				elif not empty(n_pop.state_extractors) and len(n_pop.state_extractors) > 1:
					n_pop.state_matrix = [[[] for _ in range(len(t_samp))] for _ in range(len(n_pop.state_extractors))]
				if not empty(n_pop.state_extractors):
					n_pop.state_sample_times = list(sampling_times)
		if not empty(enc_layer.encoders) and not empty(enc_layer.state_extractors):
			for ctr, n_enc in enumerate(enc_layer.encoders):
				if not empty(n_enc.state_extractors) and len(n_enc.state_extractors) == 1:
					n_enc.state_matrix = [[[] for _ in range(len(t_samp))]]
				elif not empty(n_enc.state_extractors) and len(n_enc.state_extractors) > 1:
					n_enc.state_matrix = [[[] for _ in range(len(t_samp))] for _ in range(len(n_enc.state_extractors))]
				if not empty(n_enc.state_extractors):
					n_enc.state_sample_times = list(sampling_times)

		# ################################ Main Loop ###################################
		for idx, t in enumerate(t_step):

			t_int = nest.GetKernelStatus()['time']

			if input_set.online and idx < set_size:
				local_signal = signal_iterator.next()
				local_signal.time_offset(t_int)
				intervals.append(local_signal.intervals)
				t_step.append(list(itertools.chain(*local_signal.offset_times))[0])
				t = t_step[-1]
				if intervals[-1]:
					t += intervals[-1]
				t_sim = t - t_int
			else:
				local_signal = None
				if idx < len(t_step) - 1:
					if intervals[idx]:
						t += intervals[idx]
				t_sim = t - t_int

			if t_sim > 0.: # len(t_step) <= set_size + 1 and
				print "\nSimulating step {0} [{1} ms]".format(str(idx + 1), str(t_sim))

				# Spike templates (need to be updated online)
				if np.mean(['spike_pattern' in n for n in list(iterate_obj_list(enc_layer.generator_names))]) == 1. \
						and \
								input_set is not None:
					assert (len(input_set.spike_patterns) == stim_set.dims), "Incorrect number of spike patterns"
					if input_set.online and local_signal is not None:
						stimulus_id = [nx for nx in range(stim_set.dims) if t_step[-1] in local_signal.offset_times[
							nx]]
					else:
						stimulus_id = [nx for nx in range(stim_set.dims) if
						               t_step[idx] in input_signal.offset_times[nx]]
					if t_int == 0.:
						spks = input_set.spike_patterns[stimulus_id[0]].time_offset(t_int + nest.GetKernelStatus()[
							'resolution'], True)
					else:
						spks = input_set.spike_patterns[stimulus_id[0]].time_offset(t_int, True)
					if jitter is not None:
						spks.jitter(jitter)
					enc_layer.update_state(spks)

				elif input_set is not None and local_signal is not None and input_set.online:
					stim_input = coo_matrix(stimulus_seq.todense()[:, idx])
					local_signal.input_signal = local_signal.generate_single_step(stim_input)

					if t_int == 0.:
						dt = nest.GetKernelStatus()['resolution']
						local_signal.input_signal.time_offset(dt)
					enc_layer.update_state(local_signal.input_signal)

				network_obj.simulate(t_sim)
				network_obj.extract_population_activity()
				network_obj.extract_network_activity()

				if record:
					#######################################################################################
					# Extract state
					# =====================================================================================
					# Extract merged responses
					if not empty(network_obj.merged_populations):
						for ctr, n_pop in enumerate(network_obj.merged_populations):
							if idx == 0 and not n_pop.name[-1].isdigit():
								n_pop.name += str(ctr)
							if not empty(n_pop.state_extractors):
								print "Collecting state matrices from Population {0}".format(str(n_pop.name))
								for sample_idx, n_sample_time in enumerate(t_samp):
									assert(n_sample_time >= sampling_lag), "Minimum sampling time must be >= sampling lag"
									progress_bar(float(sample_idx+1)/float(len(t_samp)))
									sample_time = t_int + n_sample_time
									state_vectors = n_pop.extract_state_vector(time_point=sample_time, lag=sampling_lag,
									                                                       save=False)
									for state_id, state_vec in enumerate(state_vectors):
										n_pop.state_matrix[state_id][sample_idx].append(state_vec[0])
					# Extract from populations
					if not empty(network_obj.state_extractors):
						for n_pop in network_obj.populations:
							if not empty(n_pop.state_extractors):
								print "Collecting state matrices from Population {0}".format(str(n_pop.name))
								for sample_idx, n_sample_time in enumerate(t_samp):
									assert (n_sample_time >= sampling_lag), "Minimum sampling time must be >= sampling lag"
									progress_bar(float(sample_idx+1) / float(len(t_samp)))
									sample_time = t_int + n_sample_time
									state_vectors = n_pop.extract_state_vector(time_point=sample_time, lag=sampling_lag,
									                                           save=False)
									for state_id, state_vec in enumerate(state_vectors):
										n_pop.state_matrix[state_id][sample_idx].append(state_vec[0])
					if not store_responses:
						network_obj.flush_records(decoders=True)
					enc_layer.extract_encoder_activity()

					# Extract from Encoders
					if not empty(enc_layer.encoders) and not empty(enc_layer.state_extractors):
						for ctr, n_enc in enumerate(enc_layer.encoders):
							if not empty(n_enc.state_extractors):
								print "Collecting state matrices from Encoder {0}".format(str(n_enc.name))
								for sample_idx, n_sample_time in enumerate(t_samp):
									assert (n_sample_time >= sampling_lag), "Minimum sampling time must be >= sampling lag"
									progress_bar(float(sample_idx+1) / float(len(t_samp)))
									sample_time = t_int + n_sample_time
									state_vectors = n_enc.extract_state_vector(time_point=sample_time, lag=sampling_lag,
									                                           save=False)
									for state_id, state_vec in enumerate(state_vectors):
										n_enc.state_matrix[state_id][sample_idx].append(state_vec[0])
						if not store_responses:
							enc_layer.flush_records(decoders=True)
		if record:
			# compile matrices:
			if not empty(network_obj.merged_populations):
				for ctr, n_pop in enumerate(network_obj.merged_populations):
					if not empty(n_pop.state_matrix):
						n_pop.compile_state_matrix(sampling_times=sampling_times)
			if not empty(network_obj.state_extractors):
				for n_pop in network_obj.populations:
					if not empty(n_pop.state_matrix):
						n_pop.compile_state_matrix(sampling_times=sampling_times)
			if not empty(enc_layer.encoders) and not empty(enc_layer.state_extractors):
				for ctr, n_enc in enumerate(enc_layer.encoders):
					n_enc.compile_state_matrix(sampling_times=sampling_times)
	####################################################################################################################
	elif sampling_times is not None and isinstance(t_samp, float) and not average:  # sub-sampled state (and input)
		# multiple sampling times per stimulus (build multiple state matrices)
		if not input_set.online:
			t_step = np.sort(list(iterate_obj_list(input_signal.offset_times)))

		sample_every_n = int(round(t_samp ** (-1))) #* step_size  # take one sample of activity every n steps

		if store_responses:
			assert(isinstance(store_responses, str)), "Please provide a path to store the responses in"
			print "Warning: All response matrices will be stored to file!"

		if input_set.online:
			print "\nSimulating {0} steps".format(str(set_size))
		else:
			print "\nSimulating {0} ms in {1} steps".format(str(
				input_signal.input_signal.t_stop - input_signal.input_signal.t_start), str(len(t_step)))

		# initialize state matrices
		if not empty(network_obj.merged_populations):
			for n_pop in network_obj.merged_populations:
				if not empty(n_pop.state_extractors):
					n_pop.state_matrix = [[] for _ in range(len(n_pop.state_extractors))]
		if not empty(network_obj.state_extractors):
			for n_pop in network_obj.populations:
				if not empty(n_pop.state_extractors):
					n_pop.state_matrix = [[] for _ in range(len(n_pop.state_extractors))]
		if not empty(enc_layer.encoders) and not empty(enc_layer.state_extractors):
			for n_pop in enc_layer.encoders:
				if not empty(n_pop.state_extractors):
					n_pop.state_matrix = [[] for _ in range(len(n_pop.state_extractors))]

		for idx, t in enumerate(t_step):

			t_int = nest.GetKernelStatus()['time']

			if input_set.online and idx < set_size:
				local_signal = signal_iterator.next()
				local_signal.time_offset(t_int)
				intervals.append(local_signal.intervals)
				t_step.append(list(itertools.chain(*local_signal.offset_times))[0])
				t = t_step[-1]
				if intervals[-1]:
					t += intervals[-1]
				t_sim = t - t_int
			else:
				local_signal = None
				if idx < len(t_step) - 1:
					if intervals[idx]:
						t += intervals[idx]
				t_sim = t - t_int
			if t_sim > 0.:
				print "\nSimulating step {0} [{1} ms]".format(str(idx + 1), str(t_sim))
				# Spike templates (need to be updated online)
				if np.mean(['spike_pattern' in n for n in list(iterate_obj_list(enc_layer.generator_names))]) == 1. \
						and \
								input_set is not None:
					assert (len(input_set.spike_patterns) == stim_set.dims), "Incorrect number of spike patterns"
					if input_set.online and local_signal is not None:
						stimulus_id = [nx for nx in range(stim_set.dims) if t_step[-1] in local_signal.offset_times[
							nx]]
					else:
						stimulus_id = [nx for nx in range(stim_set.dims) if
						               t_step[idx] in input_signal.offset_times[nx]]
					if t_int == 0.:
						spks = input_set.spike_patterns[stimulus_id[0]].time_offset(t_int + nest.GetKernelStatus()[
							'resolution'], True)
					else:
						spks = input_set.spike_patterns[stimulus_id[0]].time_offset(t_int, True)
					if jitter is not None:
						spks.jitter(jitter)
					enc_layer.update_state(spks)

				elif input_set is not None and local_signal is not None and input_set.online:
					stim_input = coo_matrix(stimulus_seq.todense()[:, idx])
					local_signal.input_signal = local_signal.generate_single_step(stim_input)

					if t_int == 0.:
						dt = nest.GetKernelStatus()['resolution']
						local_signal.input_signal.time_offset(dt)
					enc_layer.update_state(local_signal.input_signal)

				network_obj.simulate(t_sim)
				network_obj.extract_population_activity()
				network_obj.extract_network_activity()

				if record:
					#######################################################################################
					# Extract state
					# =====================================================================================
					# Extract merged responses
					if not empty(network_obj.merged_populations):
						for ctr, n_pop in enumerate(network_obj.merged_populations):
							if idx == 0 and not n_pop.name[-1].isdigit():
								n_pop.name += str(ctr)
							if not empty(n_pop.state_extractors):
								print "Collecting response samples from Population {0} [rate = {1}]".format(str(
									n_pop.name), str(t_samp))
								responses = n_pop.extract_response_matrix(start=t_int, stop=t, save=False)
								for response_idx, n_response in enumerate(responses):
									if store_responses:
										np.save(store_responses + n_pop.name + '-StateVar{0}-Response{1}.npy'.format(str(
											response_idx), str(idx)), n_response)
									if idx == 0:
										n_pop.state_matrix[response_idx] = n_response.as_array()[:, ::sample_every_n]
										print n_pop.state_matrix[response_idx].shape
									else:
										n_pop.state_matrix[response_idx] = np.append(n_pop.state_matrix[response_idx],
										                                   n_response.as_array()[:, ::sample_every_n], axis=1)
										print n_pop.state_matrix[response_idx].shape

					# Extract from populations
					if not empty(network_obj.state_extractors):
						for n_pop in network_obj.populations:
							if not empty(n_pop.state_extractors):
								print "Collecting response samples from Population {0} [rate = {1}]".format(str(
									n_pop.name), str(t_samp))
								responses = n_pop.extract_response_matrix(start=t_int, stop=t, save=False)
								for response_idx, n_response in enumerate(responses):
									if store_responses:
										np.save(store_responses + n_pop.name + '-StateVar{0}-Response{1}.npy'.format(str(
											response_idx), str(idx)), n_response)
									if idx == 0:
										n_pop.state_matrix[response_idx] = n_response.as_array()[:, ::sample_every_n]
										#print n_pop.state_matrix[response_idx].shape
									else:
										n_pop.state_matrix[response_idx] = np.append(n_pop.state_matrix[response_idx], n_response.as_array(
										)[:, ::sample_every_n], axis=1)
										#print n_pop.state_matrix[response_idx].shape
					if not store_responses:
						network_obj.flush_records(decoders=True)
					enc_layer.extract_encoder_activity()

					# Extract from Encoders
					if not empty(enc_layer.encoders) and not empty(enc_layer.state_extractors):
						for ctr, n_enc in enumerate(enc_layer.encoders):
							if not empty(n_enc.state_extractors):
								print "Collecting response samples from Encoder {0} [rate = {1}]".format(str(
									n_enc.name), str(t_samp))
								responses = n_enc.extract_response_matrix(start=t_int, stop=t, save=False)
								for response_idx, n_response in enumerate(responses):
									if store_responses:
										np.save(store_responses + n_enc.name + '-StateVar{0}-Response{1}.npy'.format(str(
											response_idx), str(idx)), n_response)
									if idx == 0:
										n_enc.state_matrix[response_idx] = n_response.as_array()[:, ::sample_every_n]
									else:
										n_enc.state_matrix[response_idx] = np.append(n_enc.state_matrix[response_idx], n_response.as_array(
										)[:, ::sample_every_n], axis=1)
						if not store_responses:
							enc_layer.flush_records(decoders=True)
	####################################################################################################################
	elif sampling_times is not None and isinstance(t_samp, float) and average:  # sub-sampled state (and input)
		# multiple sampling times per stimulus (build multiple state matrices), state vector = average over stimulus
		# presentation time
		if not input_set.online:
			t_step = np.sort(list(iterate_obj_list(input_signal.offset_times)))

		sample_every_n = int(round(t_samp ** (-1)))  # * step_size  # take one sample of activity every n steps

		if store_responses:
			assert (isinstance(store_responses, str)), "Please provide a path to store the responses in"
			print "Warning: All response matrices will be stored to file!"

		if input_set.online:
			print "\nSimulating {0} steps".format(str(set_size))
		else:
			print "\nSimulating {0} ms in {1} steps".format(str(
				input_signal.input_signal.t_stop - input_signal.input_signal.t_start), str(len(t_step)))

		# initialize state matrices
		if not empty(network_obj.merged_populations):
			for n_pop in network_obj.merged_populations:
				if not empty(n_pop.state_extractors):
					n_pop.state_matrix = [[] for _ in range(len(n_pop.state_extractors))]
		if not empty(network_obj.state_extractors):
			for n_pop in network_obj.populations:
				if not empty(n_pop.state_extractors):
					n_pop.state_matrix = [[] for _ in range(len(n_pop.state_extractors))]
		if not empty(enc_layer.encoders) and not empty(enc_layer.state_extractors):
			for n_pop in enc_layer.encoders:
				if not empty(n_pop.state_extractors):
					n_pop.state_matrix = [[] for _ in range(len(n_pop.state_extractors))]

		inter_stim_int = 0

		for idx, t in enumerate(t_step):

			t_int = nest.GetKernelStatus()['time']

			if input_set.online and idx < set_size:
				local_signal = signal_iterator.next()
				local_signal.time_offset(t_int)
				intervals.append(local_signal.intervals)
				t_step.append(list(itertools.chain(*local_signal.offset_times))[0])
				t = t_step[-1]
				if intervals[-1]:
					t += intervals[-1]
					inter_stim_int = intervals[-1]
				t_sim = t - t_int
			else:
				local_signal = None
				if idx < len(t_step) - 1:
					if intervals[idx]:
						t += intervals[idx]
						inter_stim_int = intervals[idx]
				t_sim = t - t_int
			if t_sim > 0.:
				print "\nSimulating step {0} [{1} ms]".format(str(idx + 1), str(t_sim))
				# Spike templates (need to be updated online)
				if np.mean(['spike_pattern' in n for n in list(iterate_obj_list(enc_layer.generator_names))]) == 1. \
						and \
								input_set is not None:
					assert (len(input_set.spike_patterns) == stim_set.dims), "Incorrect number of spike patterns"
					if input_set.online and local_signal is not None:
						stimulus_id = [nx for nx in range(stim_set.dims) if t_step[-1] in local_signal.offset_times[
							nx]]
					else:
						stimulus_id = [nx for nx in range(stim_set.dims) if
						               t_step[idx] in input_signal.offset_times[nx]]
					if t_int == 0.:
						spks = input_set.spike_patterns[stimulus_id[0]].time_offset(t_int + nest.GetKernelStatus()[
							'resolution'], True)
					else:
						spks = input_set.spike_patterns[stimulus_id[0]].time_offset(t_int, True)
					if jitter is not None:
						spks.jitter(jitter)
					enc_layer.update_state(spks)

				elif input_set is not None and local_signal is not None and input_set.online:
					stim_input = coo_matrix(stimulus_seq.todense()[:, idx])
					local_signal.input_signal = local_signal.generate_single_step(stim_input)

					if t_int == 0.:
						dt = nest.GetKernelStatus()['resolution']
						local_signal.input_signal.time_offset(dt)
					enc_layer.update_state(local_signal.input_signal)

				network_obj.simulate(t_sim)
				network_obj.extract_population_activity()
				network_obj.extract_network_activity()

				if record:
					#######################################################################################
					# Extract state
					# =====================================================================================
					# Extract merged responses
					if not empty(network_obj.merged_populations):
						for ctr, n_pop in enumerate(network_obj.merged_populations):
							if idx == 0 and not n_pop.name[-1].isdigit():
								n_pop.name += str(ctr)
							if not empty(n_pop.state_extractors):
								print "Collecting response samples from Population {0} [rate = {1}]".format(str(
									n_pop.name), str(t_samp))
								responses = n_pop.extract_response_matrix(start=t_int, stop=t-inter_stim_int, save=False)
								for response_idx, n_response in enumerate(responses):
									if store_responses:
										np.save(
											store_responses + n_pop.name + '-StateVar{0}-Response{1}.npy'.format(str(
												response_idx), str(idx)), n_response)
									if idx == 0:
										subsampled_states = n_response.as_array()[:, ::sample_every_n]
										n_pop.state_matrix[response_idx] = np.array([np.mean(subsampled_states, 1)]).T
										#print n_pop.state_matrix[response_idx].shape
									else:
										subsampled_states = n_response.as_array()[:, ::sample_every_n]
										state_vec = np.array([np.mean(subsampled_states, 1)]).T
										n_pop.state_matrix[response_idx] = np.append(n_pop.state_matrix[response_idx],
										                                             state_vec,
										                                             axis=1)
										#print n_pop.state_matrix[response_idx].shape

					# Extract from populations
					if not empty(network_obj.state_extractors):
						for n_pop in network_obj.populations:
							if not empty(n_pop.state_extractors):
								print "Collecting response samples from Population {0} [rate = {1}]".format(str(
									n_pop.name), str(t_samp))
								responses = n_pop.extract_response_matrix(start=t_int, stop=t-inter_stim_int, save=False)
								for response_idx, n_response in enumerate(responses):
									if store_responses:
										np.save(
											store_responses + n_pop.name + '-StateVar{0}-Response{1}.npy'.format(str(
												response_idx), str(idx)), n_response)
									if idx == 0:
										subsampled_states = n_response.as_array()[:, ::sample_every_n]
										n_pop.state_matrix[response_idx] = np.array([np.mean(subsampled_states, 1)]).T
										#print n_pop.state_matrix[response_idx].shape
									else:
										subsampled_states = n_response.as_array()[:, ::sample_every_n]
										state_vec = np.array([np.mean(subsampled_states, 1)]).T
										n_pop.state_matrix[response_idx] = np.append(n_pop.state_matrix[response_idx],
										                                             state_vec, axis=1)
										#print n_pop.state_matrix[response_idx].shape
					if not store_responses:
						network_obj.flush_records(decoders=True)
					enc_layer.extract_encoder_activity()

					# Extract from Encoders
					if not empty(enc_layer.encoders) and not empty(enc_layer.state_extractors):
						for ctr, n_enc in enumerate(enc_layer.encoders):
							if not empty(n_enc.state_extractors):
								print "Collecting response samples from Encoder {0} [rate = {1}]".format(str(
									n_enc.name), str(t_samp))
								responses = n_enc.extract_response_matrix(start=t_int, stop=t-inter_stim_int,
								                                          save=False)
								for response_idx, n_response in enumerate(responses):
									if store_responses:
										np.save(
											store_responses + n_enc.name + '-StateVar{0}-Response{1}.npy'.format(str(
												response_idx), str(idx)), n_response)
									if idx == 0:
										subsampled_states = n_response.as_array()[:, ::sample_every_n]
										n_enc.state_matrix[response_idx] = np.array([np.mean(subsampled_states, 1)]).T
									else:
										subsampled_states = n_response.as_array()[:, ::sample_every_n]
										state_vec = np.array([np.mean(subsampled_states, 1)]).T
										n_enc.state_matrix[response_idx] = np.append(n_enc.state_matrix[response_idx],
										                                             state_vec, axis=1)
						if not store_responses:
							enc_layer.flush_records(decoders=True)
	else:
		raise NotImplementedError("Specify sampling times as None (last step sample), list or array of times, or float")

	if store_responses:
		return epochs