__author__ = 'duarte'
import matplotlib.pyplot as pl
from modules.visualization import get_cmap
from modules.auxiliary import gather_states, retrieve_data_set, retrieve_stimulus_timing, set_sampling_parameters, \
	update_input_signals, flush, time_keep
from modules.input_architect import generate_template
from modules.parameters import copy_dict
import modules.signals as signals
import modules.analysis as analysis
import nest
import time
import numpy as np
from scipy.stats import sem
import itertools
import pickle


def harvest_results(pars, analysis_dict, results_path, plot=True, display=True, save=False):
	processed = dict()
	lab = dict() # to check
	if plot:
		fig1 = pl.figure()
		fig1.suptitle(analysis_dict['fig_title'])
	axes = []
	for ax_n, ax_title in enumerate(analysis_dict['ax_titles']):
		if plot:
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

	return processed, lab, axes


def process_input_sequence(parameter_set, net, enc_layer, stimulus_set, input_signal_set, set_name, record=True,
                           save_data=False, storage_paths=None, extra_step=None):
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
	if save_data:
		# store stimulus sequence
		full_seq = {'full_sequence': set_labels}
		with open(storage_paths['inputs']+storage_paths['label']+'_Input_sequence.pkl', 'w') as fp:
			pickle.dump(full_seq, fp)

	####################################################################################################################
	# if sampling_times is None:  # one sample for each stimulus (acquired at the last time point of each stimulus)
	print("\n\nSimulating {0} steps".format(str(set_size)))

	# ################################ Main Loop ###################################
	stim_idx = 0
	simulation_time = 0.0
	stimulus_onset = 0.0

	if save_data:
		# store initial states
		initial_states = {}
		for neuron_id in net.merged_populations[0].gids:
			st = nest.GetStatus([neuron_id])[0]
			status = copy_dict(st, {'element_type': str(st['element_type']),
			                        'recordables': str(st['recordables']),
			                        'model': str(st['model'])})
			initial_states.update({neuron_id: status})

		with open(storage_paths['other'] + storage_paths['label'] + '_InitialConditions.pkl', 'w') as fp:
			pickle.dump(initial_states, fp)

		# store connectivity
		net.extract_synaptic_weights()
		net.extract_synaptic_delays()
		enc_layer.extract_connectivity(net, sub_set=False, progress=True)

		connectivities = {'weights': {}, 'delays': {}}
		connectivities['weights'].update(enc_layer.synaptic_weights)
		connectivities['weights'].update(net.synaptic_weights)
		connectivities['delays'].update(enc_layer.connection_delays)
		connectivities['delays'].update(net.synaptic_delays)

		with open(storage_paths['other'] + storage_paths['label'] + '_Connectivity.pkl', 'w') as fp:
			pickle.dump(connectivities, fp)

		# store templates
		spike_templates = {}
		for idx, nn in enumerate(input_signal_set.spike_patterns):
			spike_templates.update({'stim{0}'.format(str(idx)): {'senders': [], 'times': []}})
			all_times = []
			all_senders = []
			for neuron_id, spk_train in nn.spiketrains.items():
				times = np.round(spk_train.spike_times, 1)
				all_times.append(times)
				all_senders.append(np.ones_like(times) * neuron_id)

			spike_templates['stim{0}'.format(str(idx))]['times'] = list(itertools.chain(*all_times))
			spike_templates['stim{0}'.format(str(idx))]['senders'] = list(itertools.chain(*all_senders))

		with open(storage_paths['inputs'] + storage_paths['label'] + '_SpikeTemplates.pkl', 'w') as fp:
			pickle.dump(spike_templates, fp)

		# prepare to store network and encoder activity
		net_activity = {}
		stim_activity = {}

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
				inp_spikes = update_spike_template(enc_layer, stim_idx, input_signal_set, stimulus_set,
				                                  local_signal, t_samp,
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

			if save_data:
				E_spikes = nest.GetStatus(net.device_gids[0][0])[0]['events']
				I_spikes = nest.GetStatus(net.device_gids[1][0])[0]['events']
				all_senders = list(itertools.chain(*[E_spikes['senders'], I_spikes['senders']]))
				all_times = list(itertools.chain(*[E_spikes['times'], I_spikes['times']]))
				print min(all_times), max(all_times) # check that previous spikes were deleted
				net_activity.update({'senders{0}'.format(stim_idx): all_senders, 'times{0}'.format(stim_idx):
					all_times})

				inp_times = []
				inp_ids = []
				for nn in inp_spikes.id_list:
					spk_times = [round(n, 1) for n in inp_spikes[nn].spike_times]
					inp_times.append(spk_times)
					inp_ids.append(nn * np.ones_like(spk_times))
				stim_activity.update({'senders{0}'.format(stim_idx): list(itertools.chain(*inp_times)),
				                      'times{0}'.format(stim_idx): list(itertools.chain(*inp_ids))})

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
					print n_pop.decoding_layer.sampled_times

			if save_data:
				E_spikes = nest.GetStatus(net.device_gids[0][0])[0]['events']
				I_spikes = nest.GetStatus(net.device_gids[1][0])[0]['events']
				all_senders = list(itertools.chain(*[E_spikes['senders'], I_spikes['senders']]))
				all_times = list(itertools.chain(*[E_spikes['times'], I_spikes['times']]))
				print min(all_times), max(all_times)  # check that previous spikes were deleted
				net_activity.update({'senders{0}'.format(stim_idx): all_senders, 'times{0}'.format(stim_idx):
					all_times})

			# flush unnecessary information
			if not record:
				flush(net, enc_layer, decoders=True)
			else:
				flush(net, enc_layer, decoders=False)

			if stim_idx == set_size:
				net.simulate(decoder_delay)
				if extra_step is not None:
					net.simulate(extra_step)

		timing['step_time'].append(time_keep(start_time, stim_idx, set_size, stim_start))

	timing['total_time'] = (time.time() - start_time) / 60.

	# gather states
	gather_states(net, enc_layer, t0, set_labels) # , flush_devices=False)

	if save_data:
		with open(storage_paths['activity'] + storage_paths['label'] + '_NetworkActivity.pkl', 'w') as fp:
			pickle.dump(net_activity, fp)
		with open(storage_paths['activity'] + storage_paths['label'] + '_InputActivity.pkl', 'w') as fp:
			pickle.dump(stim_activity, fp)

	return epochs, timing


def process_states(net, enc_layer, target_matrix, stim_set, data_sets=None, accepted_idx=None, plot=False,
                   display=True, save=False, save_paths=None):
	"""

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
							results['rank'][n_pop.name].update({var + str(idx_var): analysis.get_state_rank(state_matrix)})
						elif set_name == 'train':
							for readout in readouts:
								readout.train(state_matrix, np.array(target), index=None, accepted=accepted_ids,
								              display=display)

								readout.measure_stability(display=display)
								if plot and save:
									readout.plot_weights(display=display, save=save_paths['figures'] + save_paths[
										'label'])
								elif plot:
									readout.plot_weights(display=display, save=False)

						elif set_name == 'test':
							for readout in readouts:
								output, target = readout.test(state_matrix, np.array(target), index=None,
									                            accepted=accepted_ids, display=display)

								results['performance'][n_pop.name][var + str(idx_var)].update(
									{readout.name: readout.measure_performance(target, output, display=display)})
								results['performance'][n_pop.name][var + str(idx_var)].update(
									{readout.name: readout.measure_performance(target, display=display)})
								results['performance'][n_pop.name][var + str(idx_var)][readout.name].update(
									{'norm_wOut': readout.norm_wout})
								results['dimensionality'][n_pop.name].update(
									{var + str(idx_var): analysis.compute_dimensionality(state_matrix)})
						if plot and set_name != 'transient':
							if save:
								analysis.analyse_state_matrix(state_matrix, labels, label=n_pop.name + var + set_name,
								                              plot=plot, display=display, save=save_paths[
									                          'figures']+save_paths['label'])
							else:
								analysis.analyse_state_matrix(state_matrix, labels, label=n_pop.name + var + set_name,
								                              plot=plot, display=display, save=False)
						if save:
							np.save(save_paths['activity'] + save_paths['label'] +
							        '_population{0}_state{1}_{2}.npy'.format(n_pop.name, var, set_name), state_matrix)
	return results


def update_spike_template(enc_layer, idx, input_signal_set, stimulus_set, local_signal, t_samp, input_signal, jitter,
                          stimulus_onset, add_noise=False):
	"""

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

	if isinstance(add_noise, float):
		noise_realization = generate_template(n_neurons=len(spks.id_list), rate=spks.mean_rate(), duration=add_noise,
		                                      resolution=0.01, rng=None, store=False)
		noise_realization = noise_realization.time_offset(spks.last_spike_time(), True)
		spks.merge(noise_realization)

	spks = spks.time_offset(stimulus_onset, True)
	enc_layer.update_state(spks)

	return spks