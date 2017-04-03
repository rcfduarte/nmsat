__author__ = 'duarte'
import matplotlib.pyplot as pl
from modules.visualization import get_cmap
from modules.auxiliary import gather_states, retrieve_data_set, retrieve_stimulus_timing, set_sampling_parameters, \
	update_input_signals, update_spike_template, flush, time_keep
import modules.signals as signals
import modules.analysis as analysis
import nest
import time
import numpy as np
from scipy.stats import sem
import itertools


def harvest_results(pars, analysis_dict, results_path, plot=True, display=True, save=False):
	processed = dict()
	lab = dict() # to check

	fig1 = pl.figure()
	fig1.suptitle(analysis_dict['fig_title'])
	axes = []
	for ax_n, ax_title in enumerate(analysis_dict['ax_titles']):
		ax = fig1.add_subplot(len(analysis_dict['ax_titles']), 1, ax_n + 1)

		colors = get_cmap(len(analysis_dict['key_sets'][ax_n]), 'Accent')
		for idx_k, keys in enumerate(analysis_dict['key_sets'][ax_n]):
			print "\nHarvesting {0}".format(keys)
			labels, result = pars.harvest(results_path, key_set=keys)
			if plot:
				ax.plot(pars.parameter_axes['xticks'], np.mean(result.astype(float), 1), '-', c=colors(idx_k),
				        label=analysis_dict['labels'][ax_n][idx_k])
				ax.errorbar(pars.parameter_axes['xticks'], np.mean(result.astype(float), 1), marker='o', mfc=colors(idx_k),
				            mec=colors(idx_k), ms=2, linestyle='none', ecolor=colors(idx_k), yerr=sem(result.astype(
					float), 1))
			processed.update({keys: result})
			lab.update({keys: labels})
		if plot:
			ax.set_xlabel(r'$' + pars.parameter_axes['xlabel'] + '$')
			ax.set_xlim([min(pars.parameter_axes['xticks']), max(pars.parameter_axes['xticks'])])
			ax.set_title(ax_title)
			ax.legend()
		axes.append(ax)
	if save and plot:
		fig1.savefig(save + '_Results_{0}'.format(analysis_dict['fig_title']))
	if display and plot:
		pl.show(block=False)

	return processed, lab, axes, fig1


def process_input_sequence(parameter_set, net, enc_layer, stimulus_set, input_signal_set, set_name, record=True,
                           save_data=False):
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

		if save_data:
			# store initial states and connectivity
			pass

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
					update_input_signals(enc_layer, stim_idx, stimulus_seq, local_signal, simulation_resolution)

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
					update_input_signals(enc_layer, stim_idx, stimulus_seq, local_signal, simulation_resolution)

				# simulate delays
				net.simulate(encoder_delay + decoder_delay)

				# reset
				analysis.reset_decoders(net, enc_layer)

				# add sample time to decoders...
				for n_pop in list(itertools.chain(*[net.merged_populations, net.populations, enc_layer.encoders])):
					if n_pop.decoding_layer is not None:
						n_pop.decoding_layer.sampled_times.append(state_sample_time)

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
						update_input_signals(enc_layer, stim_idx, stimulus_seq, local_signal, simulation_resolution)

				# simulate delays
				net.simulate(encoder_delay + decoder_delay)

				# reset
				analysis.reset_decoders(net, enc_layer)

				# add sample time to decoders...
				for n_pop in list(itertools.chain(*[net.merged_populations, net.populations, enc_layer.encoders])):
					if n_pop.decoding_layer is not None:
						n_pop.decoding_layer.sampled_times.append(state_sample_time)

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

	return epochs, timing
