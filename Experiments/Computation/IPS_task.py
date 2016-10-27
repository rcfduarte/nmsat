from __init__ import *


def run(parameter_set, plot=False, display=False, save=False, debug=False, online=True, dataset_path=''):
	"""
	Run the RC sequence processing task (adapted for the language input
	:param parameter_set:
	:param plot:
	:param display:
	:param save:
	:param debug:
	:return:
	"""
	import scipy.io as sio
	if not isinstance(parameter_set, ParameterSet):
		if isinstance(parameter_set, basestring) or isinstance(parameter_set, dict):
			parameter_set = ParameterSet(parameter_set)
		else:
			raise TypeError("parameter_set must be ParameterSet, string with full path to parameter file or dictionary")

	###################################################################################
	# Setup extra variables and parameters
	# =================================================================================
	if plot:
		import Modules.visualization as vis
		vis.set_global_rcParams(parameter_set.kernel_pars['mpl_path'])
	paths = set_storage_locations(parameter_set, save)

	np.random.seed(parameter_set.kernel_pars['np_seed'])
	results = dict()
	##################################################################################
	# Set kernel and simulation parameters
	# ================================================================================
	print '\nRuning ParameterSet {0}'.format(parameter_set.label)
	nest.ResetKernel()
	nest.set_verbosity('M_WARNING')
	nest.SetKernelStatus(extract_nestvalid_dict(parameter_set.kernel_pars.as_dict(), type='kernel'))

	###################################################################################
	# Build network
	# =================================================================================
	net = Network(parameter_set.net_pars)
	net.merge_subpopulations([net.populations[0], net.populations[1]], name='EI')

	###################################################################################
	# Randomize initial variable values
	# =================================================================================
	for n in list(iterate_obj_list(net.populations)):
		n.randomize_initial_states('V_m', randomization_function=np.random.uniform, low=-70., high=-55.)
	# n.randomize_initial_states('V_th', randomization_function=np.random.normal, loc=-50., scale=1.)

	###################################################################################
	# Build and connect input
	# =================================================================================
	# LOAD StimulusSet
	stim_set_time = time.time()
	data = sio.loadmat(dataset_path)
	data_mat = data['X']
	word_sequence = data_mat[:, :75].T
	target_roles = data_mat[:, -7:].T

	# mark end-of-sentence
	eos_markers = []
	for n in range(target_roles.shape[1]):
		if not np.mean(target_roles[:, n]):
			eos_markers.append(n)

	# discard eos_markers in transient set (because full_set[0] == word_sequence[transient_set_length] and
	# full_set_target[0] == target_roles[transient_set_length]):
	eos_markers = np.array(eos_markers)
	eos_markers -= parameter_set.stim_pars.transient_set_length
	eos_markers = eos_markers[np.where(eos_markers > 0.)]

	# extract word labels (for the StimulusSet object - identity of each input stimulus)
	seq_labels = []
	for n in range(word_sequence.shape[1]):
		seq_labels.append(np.where(word_sequence[:, n])[0][0])

	# split data sets (train+test uses full_set)
	transient_set = word_sequence[:, :parameter_set.stim_pars.transient_set_length]
	train_set = word_sequence[:, parameter_set.stim_pars.transient_set_length:parameter_set.stim_pars
		                                                                          .transient_set_length + parameter_set.stim_pars.train_set_length]
	test_set = word_sequence[:, parameter_set.stim_pars.transient_set_length +
	                            parameter_set.stim_pars.train_set_length:parameter_set.stim_pars.transient_set_length +
	                                                                     parameter_set.stim_pars.train_set_length +
	                                                                     parameter_set.stim_pars.test_set_length]
	full_set = word_sequence[:,
	           parameter_set.stim_pars.transient_set_length:parameter_set.stim_pars.transient_set_length +
	                                                        parameter_set.stim_pars.train_set_length +
	                                                        parameter_set.stim_pars.test_set_length]
	full_set_labels = seq_labels[
	                  parameter_set.stim_pars.transient_set_length:parameter_set.stim_pars.transient_set_length +
	                                                               parameter_set.stim_pars.train_set_length +
	                                                               parameter_set.stim_pars.test_set_length]
	full_set_targets = target_roles[:,
	                   parameter_set.stim_pars.transient_set_length:parameter_set.stim_pars.transient_set_length +
	                                                                parameter_set.stim_pars.train_set_length +
	                                                                parameter_set.stim_pars.test_set_length]
	# Create StimulusSet
	stim = StimulusSet()
	stim.load_data(full_set, type='full_set')
	stim.load_data(full_set_labels, type='full_set_labels')
	stim.load_data(transient_set, type='transient_set')
	stim.load_data(seq_labels[:parameter_set.stim_pars.transient_set_length], type='transient_set_labels')
	stim.load_data(train_set, type='train_set')
	stim.load_data(seq_labels[parameter_set.stim_pars.transient_set_length:parameter_set.stim_pars
	               .transient_set_length + parameter_set.stim_pars.train_set_length], type='train_set_labels')
	stim.load_data(test_set, type='test_set')
	stim.load_data(seq_labels[parameter_set.stim_pars.transient_set_length +
	                          parameter_set.stim_pars.train_set_length:parameter_set.stim_pars.transient_set_length +
	                                                                   parameter_set.stim_pars.train_set_length +
	                                                                   parameter_set.stim_pars.test_set_length],
	               type='test_set_labels')
	print "- Elapsed Time: {0}".format(str(time.time() - stim_set_time))

	# Create InputSignalSet
	input_set_time = time.time()
	inputs = InputSignalSet(parameter_set, stim, online=online)
	if not empty(stim.transient_set_labels):
		inputs.generate_transient_set(stim)
		parameter_set.kernel_pars.transient_t = inputs.transient_stimulation_time
	# if not online:
	inputs.generate_full_set(stim)
	# inputs.generate_unique_set(stim)
	inputs.generate_train_set(stim)
	inputs.generate_test_set(stim)
	print "- Elapsed Time: {0}".format(str(time.time() - input_set_time))

	# Plot example signal
	if plot and debug and not online:
		fig_inp = pl.figure()
		ax1 = fig_inp.add_subplot(211)
		ax2 = fig_inp.add_subplot(212)
		fig_inp.suptitle('Input Stimulus / Signal')
		inp_plot = vis.InputPlots(stim_obj=stim, input_obj=inputs.test_set_signal, noise_obj=inputs.test_set_noise)
		inp_plot.plot_stimulus_matrix(set='test', ax=ax1, save=False, display=False)
		inp_plot.plot_input_signal(ax=ax2, save=paths['figures'] + paths['label'], display=display)
		inp_plot.plot_input_signal(save=paths['figures'] + paths['label'], display=display)
		inp_plot.plot_signal_and_noise(save=paths['figures'] + paths['label'], display=display)
	parameter_set.kernel_pars.sim_time = inputs.train_stimulation_time + inputs.test_stimulation_time

	if save:
		stim.save(paths['inputs'])
		if debug:
			inputs.save(paths['inputs'])

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
		vis.extract_encoder_connectivity(enc_layer, net, display, save=paths['figures']+paths['label'])

	#######################################################################################
	# Set-up Analysis
	# =====================================================================================
	net.connect_devices()
	net.connect_decoders(parameter_set.decoding_pars)

	######################################################################################
	# Connect Network
	# ====================================================================================
	net.connect_populations(parameter_set.connection_pars)

	if plot and debug:
		fig_W = pl.figure()
		topology = vis.TopologyPlots(parameter_set.connection_pars, net)
		topology.print_network(depth=3)
		ax1 = pl.subplot2grid((6, 6), (0, 0), rowspan=4, colspan=4)
		ax2 = pl.subplot2grid((6, 6), (5, 0), rowspan=1, colspan=4)
		ax3 = pl.subplot2grid((6, 6), (0, 5), rowspan=4, colspan=1)
		ax4 = pl.subplot2grid((6, 6), (5, 5), rowspan=1, colspan=1)
		topology.plot_connectivity(parameter_set.connection_pars.synapse_types,
	 	                           ax=[ax1, ax3, ax2, ax4], display=display, save=paths['figures']+paths['label'])

	######################################################################################
	# Simulate (Initial Transient)
	# ====================================================================================
	if not empty(stim.transient_set_labels):
		if not online:
			print "\nTransient time = {0} ms".format(str(parameter_set.kernel_pars.transient_t))

		iterate_input_sequence(net, inputs.transient_set_signal, enc_layer, sampling_times=None, stim_set=stim,
		                       input_set=inputs, set_name='transient', store_responses=False, record=False)
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
			                                                  save=paths['figures'] + paths['label'])
			enc_layer.extract_encoder_activity()
		# results.update(evaluate_encoding(enc_layer, parameter_set, analysis_interval,
		#                                  inputs.transient_set_signal, plot=plot, display=display,
		#                                  save=paths['figures']+paths['label']))

		net.flush_records()
		enc_layer.flush_records()

	#######################################################################################
	# Simulate (Train period)
	# =====================================================================================
	if not online:
		print "\nFull time = {0} ms".format(str(inputs.full_stimulation_time))
	iterate_input_sequence(net, inputs.full_set_signal, enc_layer,
	                       sampling_times=parameter_set.decoding_pars.global_sampling_times,
	                       stim_set=stim, input_set=inputs, set_name='full', store_responses=False,
	                       average=True)

	#######################################################################################
	# Process Train Data
	# =====================================================================================
	from sklearn import preprocessing

	set_labels = stim.full_set_labels
	shuffle_states = True
	standardize = True

	# state of merged populations
	if not empty(net.merged_populations):
		for ctr, n_pop in enumerate(net.merged_populations):
			if not empty(n_pop.state_matrix):
				state_dimensions = np.array(n_pop.state_matrix).shape
				population_readouts = n_pop.readouts
				chunker = lambda lst, sz: [lst[i:i + sz] for i in range(0, len(lst), sz)]
				n_pop.readouts = chunker(population_readouts, len(population_readouts) / state_dimensions[0])
				# copy readouts for each state matrix
				if n_pop.state_sample_times:
					n_copies = len(n_pop.state_sample_times)
					all_readouts = n_pop.copy_readout_set(n_copies)
					n_pop.readouts = all_readouts

				for idx_state, n_state in enumerate(n_pop.state_matrix):
					if not isinstance(n_state, list):
						print "\nTraining {0} readouts from Population {1}".format(str(n_pop.decoding_pars['readout'][
							                                                               'N']), str(n_pop.name))
						state_matrix = n_state.copy()
						full_set_targets = target_roles[:,
						                   parameter_set.stim_pars.transient_set_length:parameter_set.stim_pars
							                                                                .transient_set_length +
						                                                                parameter_set.stim_pars.train_set_length +
						                                                                parameter_set.stim_pars.test_set_length]
						if shuffle_states:
							# Shuffle states
							shuffled_idx = np.random.permutation(state_matrix.shape[1])

							eos_idx = np.where([shuffled_idx == n for n in eos_markers])[1]
							final_idx = np.where([shuffled_idx == n - 1 for n in eos_markers])[1]

							state_train = state_matrix[:, shuffled_idx[:parameter_set.stim_pars.train_set_length]]
							state_test = state_matrix[:, shuffled_idx[parameter_set.stim_pars.train_set_length:]]
							full_set_targets = full_set_targets[:, shuffled_idx]
							state_matrix = state_matrix[:, shuffled_idx]
						else:
							state_train = state_matrix[:, :parameter_set.stim_pars.train_set_length]
							state_test = state_matrix[:, parameter_set.stim_pars.train_set_length:]
							eos_idx = eos_markers[eos_markers <= state_matrix.shape[1]]
							final_idx = eos_markers[eos_markers <= state_matrix.shape[1]] - 1

						if standardize:
							# Standardize
							scaler = preprocessing.StandardScaler().fit(state_train.T)
							state_train = scaler.transform(state_train.T).T
							state_test = scaler.transform(state_test.T).T
							state_matrix = np.append(state_train, state_test, 1)

						overall_target_train = full_set_targets[:, :parameter_set.stim_pars.train_set_length]
						overall_state_train = state_train

						overall_test_pop = eos_idx[eos_idx >= parameter_set.stim_pars.train_set_length]
						overall_target_test = np.delete(full_set_targets.copy(), overall_test_pop, 1)
						overall_target_test = overall_target_test[:, parameter_set.stim_pars.train_set_length:]
						overall_state_test = np.delete(state_matrix.copy(), overall_test_pop, 1)
						overall_state_test = overall_state_test[:, parameter_set.stim_pars.train_set_length:]

						final_train_idx = final_idx[final_idx < parameter_set.stim_pars.train_set_length]
						final_target_train = full_set_targets[:, final_train_idx]
						final_state_train = state_matrix[:, final_train_idx]

						final_target_idx = final_idx[final_idx >= parameter_set.stim_pars.train_set_length]
						final_target_test = full_set_targets[:, final_target_idx]
						final_state_test = state_matrix[:, final_target_idx]

						label = n_pop.name + '-Test-StateVar{0}'.format(str(idx_state))
						if save:
							save_path = paths['figures'] + label
						else:
							save_path = False
						overall_label = n_pop.name + 'OVERALL-Test-StateVar{0}'.format(str(idx_state))
						final_label = n_pop.name + 'FINAL-Test-StateVar{0}'.format(str(idx_state))
						if save:
							np.save(paths['activity'] + overall_label, overall_state_test)
							np.save(paths['activity'] + final_label, final_state_test)
						if debug:
							l = [np.where(overall_target_test[:, n])[0][0] for n in range(overall_target_test.shape[1])]
							analyse_state_matrix(overall_state_test, l, label=overall_label, plot=plot, display=display,
							                     save=save_path)
							l = [np.where(final_target_test[:, n])[0][0] for n in range(final_target_test.shape[1])]
							analyse_state_matrix(final_state_test, l, label=final_label, plot=plot, display=display,
							                     save=save_path)

						population_readouts = n_pop.readouts
						for readout in population_readouts[idx_state]:
							readout.set_index()
							if readout.name[:-1] == 'overall':
								# overall performance
								discrete_readout_train(overall_state_train, overall_target_train, readout,
								                       readout.index)
								discrete_readout_test(overall_state_test, overall_target_test, readout, readout.index)
							elif readout.name[:-1] == 'final':
								# overall performance
								discrete_readout_train(final_state_train, final_target_train, readout, readout.index)
								discrete_readout_test(final_state_test, final_target_test, readout, readout.index)
							else:
								raise TypeError("Incorrect readout name...")
							if plot:
								if save_path:
									save_path2 = save_path + readout.name + readout.rule
								else:
									save_path2 = False
								readout.plot_weights(display=display, save=save_path2)
								readout.plot_confusion(display=display, save=save_path2)
								if readout.fit_obj:
									if readout.name[:-1] == 'overall':
										vis.plot_2d_regression_fit(readout.fit_obj, overall_state_train.T, np.argmax(
											overall_target_train, 0), readout, display=display, save=save_path2)
									elif readout.name[:1] == 'final':
										vis.plot_2d_regression_fit(readout.fit_obj, final_state_train.T, np.argmax(
											final_target_train, 0), readout, display=display, save=save_path2)

	# Extract from populations
	if not empty(net.state_extractors):
		for n_pop in net.populations:
			if not empty(n_pop.state_extractors):
				for ctr, n_pop in enumerate(net.populations):
					if not empty(n_pop.state_matrix):
						state_dimensions = np.array(n_pop.state_matrix).shape
						population_readouts = n_pop.readouts
						chunker = lambda lst, sz: [lst[i:i + sz] for i in range(0, len(lst), sz)]
						n_pop.readouts = chunker(population_readouts, len(population_readouts) / state_dimensions[0])
						# copy readouts for each state matrix
						if n_pop.state_sample_times:
							n_copies = len(n_pop.state_sample_times)
							all_readouts = n_pop.copy_readout_set(n_copies)
							n_pop.readouts = all_readouts

						for idx_state, n_state in enumerate(n_pop.state_matrix):
							if not isinstance(n_state, list):
								print "\nTraining {0} readouts from Population {1}".format(
									str(n_pop.decoding_pars['readout'][
										    'N']), str(n_pop.name))
								state_matrix = n_state.copy()
								full_set_targets = target_roles[:,
								                   parameter_set.stim_pars.transient_set_length:parameter_set.stim_pars
									                                                                .transient_set_length +
								                                                                parameter_set.stim_pars.train_set_length +
								                                                                parameter_set.stim_pars.test_set_length]
								if shuffle_states:
									# Shuffle states
									shuffled_idx = np.random.permutation(state_matrix.shape[1])

									eos_idx = np.where([shuffled_idx == n for n in eos_markers])[1]
									final_idx = np.where([shuffled_idx == n - 1 for n in eos_markers])[1]

									state_train = state_matrix[:,
									              shuffled_idx[:parameter_set.stim_pars.train_set_length]]
									state_test = state_matrix[:,
									             shuffled_idx[parameter_set.stim_pars.train_set_length:]]
									full_set_targets = full_set_targets[:, shuffled_idx]
									state_matrix = state_matrix[:, shuffled_idx]
								else:
									state_train = state_matrix[:, :parameter_set.stim_pars.train_set_length]
									state_test = state_matrix[:, parameter_set.stim_pars.train_set_length:]
									eos_idx = eos_markers[eos_markers <= state_matrix.shape[1]]
									final_idx = eos_markers[eos_markers <= state_matrix.shape[1]] - 1

								if standardize:
									# Standardize
									scaler = preprocessing.StandardScaler().fit(state_train.T)
									state_train = scaler.transform(state_train.T).T
									state_test = scaler.transform(state_test.T).T
									state_matrix = np.append(state_train, state_test, 1)

								overall_target_train = full_set_targets[:, :parameter_set.stim_pars.train_set_length]
								overall_state_train = state_train

								overall_test_pop = eos_idx[eos_idx >= parameter_set.stim_pars.train_set_length]
								overall_target_test = np.delete(full_set_targets.copy(), overall_test_pop, 1)
								overall_target_test = overall_target_test[:, parameter_set.stim_pars.train_set_length:]
								overall_state_test = np.delete(state_matrix.copy(), overall_test_pop, 1)
								overall_state_test = overall_state_test[:, parameter_set.stim_pars.train_set_length:]

								final_train_idx = final_idx[final_idx < parameter_set.stim_pars.train_set_length]
								final_target_train = full_set_targets[:, final_train_idx]
								final_state_train = state_matrix[:, final_train_idx]

								final_target_idx = final_idx[final_idx >= parameter_set.stim_pars.train_set_length]
								final_target_test = full_set_targets[:, final_target_idx]
								final_state_test = state_matrix[:, final_target_idx]

								label = n_pop.name + '-Test-StateVar{0}'.format(str(idx_state))
								if save:
									save_path = paths['figures'] + label
								else:
									save_path = False
								overall_label = n_pop.name + 'OVERALL-Test-StateVar{0}'.format(str(idx_state))
								final_label = n_pop.name + 'FINAL-Test-StateVar{0}'.format(str(idx_state))
								if save:
									np.save(paths['activity'] + overall_label, overall_state_test)
									np.save(paths['activity'] + final_label, final_state_test)
								if debug:
									l = [np.where(overall_target_test[:, n])[0][0] for n in
									     range(overall_target_test.shape[1])]
									analyse_state_matrix(overall_state_test, l, label=overall_label, plot=plot,
									                     display=display,
									                     save=save_path)
									l = [np.where(final_target_test[:, n])[0][0] for n in range(final_target_test.shape[
										                                                            1])]
									analyse_state_matrix(final_state_test, l, label=final_label, plot=plot,
									                     display=display,
									                     save=save_path)

								population_readouts = n_pop.readouts
								for readout in population_readouts[idx_state]:
									readout.set_index()
									if readout.name[:-1] == 'overall':
										# overall performance
										discrete_readout_train(overall_state_train, overall_target_train, readout,
										                       readout.index)
										discrete_readout_test(overall_state_test, overall_target_test, readout,
										                      readout.index)
									elif readout.name[:-1] == 'final':
										# overall performance
										discrete_readout_train(final_state_train, final_target_train, readout,
										                       readout.index)
										discrete_readout_test(final_state_test, final_target_test, readout,
										                      readout.index)
									else:
										raise TypeError("Incorrect readout name...")
									if plot:
										if save_path:
											save_path += readout.name + readout.rule
										readout.plot_weights(display=display, save=save_path)
										readout.plot_confusion(display=display, save=save_path)
										if readout.fit_obj:
											if readout.name[:-1] == 'overall':
												vis.plot_2d_regression_fit(readout.fit_obj, overall_state_train.T,
												                       np.argmax(
													                       overall_target_train, 0), readout,
												                       display=display, save=save_path)
											elif readout.name[:1] == 'final':
												vis.plot_2d_regression_fit(readout.fit_obj, final_state_train.T,
												                         np.argmax(
													final_target_train, 0), readout, display=display, save=save_path)

	results['performance'] = {}
	results['performance'].update(analyse_performance_results(net, enc_layer, plot=plot, display=display, save=paths[
		                                                                                                           'figures'] +
	                                                                                                           paths[
		                                                                                                           'label']))

	#######################################################################################
	# Save data
	# =====================================================================================
	if save:
		with open(paths['results'] + 'Results_' + parameter_set.label, 'w') as f:
			pickle.dump(results, f)
		parameter_set.save(paths['parameters'] + 'Parameters_' + parameter_set.label)
