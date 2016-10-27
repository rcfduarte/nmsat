from __init__ import *


def run(parameter_set, analysis_interval=None, plot=False, display=False, save=False, debug=False):
	"""
	Estimate the fading memory function, with noisy input
	:param parameter_set:
	:param analysis_interval:
	:param plot:
	:param display:
	:param save:
	:return:
	"""
	if not isinstance(parameter_set, ParameterSet):
		if isinstance(parameter_set, basestring) or isinstance(parameter_set, dict):
			parameter_set = ParameterSet(parameter_set)
		else:
			raise TypeError("parameter_set must be ParameterSet, string with full path to parameter file or dictionary")
	if analysis_interval is None:
		analysis_interval = [parameter_set.kernel_pars.transient_t,
		                     parameter_set.kernel_pars.sim_time + parameter_set.kernel_pars.transient_t]

	###################################################################################
	# Setup extra variables and parameters
	# =================================================================================
	if plot:
		import Modules.visualization as vis
		vis.set_global_rcParams(parameter_set.kernel_pars['mpl_path'])
	if save:
		path = parameter_set.kernel_pars.data_path + \
		       parameter_set.kernel_pars.data_prefix + '/'
		save_path = path + parameter_set.label
		if not os.path.exists(path):
			os.mkdir(path)
	else:
		save_path = False

	np.random.seed(parameter_set.kernel_pars['np_seed'])

	##################################################################################
	# Set kernel and simulation parameters
	# ================================================================================
	print '\nRuning ParameterSet {0}'.format(parameter_set.label)
	nest.ResetKernel()
	nest.SetKernelStatus(extract_nestvalid_dict(parameter_set.kernel_pars.as_dict(), type='kernel'))

	####################################################
	# Build network
	# ===================================================
	net = Network(parameter_set.net_pars)
	for n in list(iterate_obj_list(net.populations)):
		n.randomize_initial_states('V_m',
		                           randomization_function=np.random.uniform,
		                           low=-70., high=-55.)
		n.randomize_initial_states('V_th', randomization_function=np.random.normal,
		                           loc=-55., scale=2.)

	##########################################################
	# Build and connect input
	# =========================================================
	# Build noisy input sequence
	input_seq = [1]
	total_stimulation_time = parameter_set.kernel_pars.sim_time + parameter_set.kernel_pars.transient_t
	input_noise = InputNoise(parameter_set.input_pars.noise,
	                         stop_time=total_stimulation_time)
	input_noise.generate()
	input_noise.re_seed(parameter_set.kernel_pars.np_seed)

	if plot:
		inp_plot = vis.InputPlots(stim_obj=None, input_obj=None, noise_obj=input_noise)
		inp_plot.plot_noise_component(display=display, save=save_path)

	#######################################################################################
	# Encode Input
	# =====================================================================================
	enc_layer = EncodingLayer(parameter_set.encoding_pars, signal=input_noise)
	enc_layer.connect(parameter_set.encoding_pars, net)

	######################################################################################
	# Set-up Analysis
	# =====================================================================================
	net.connect_devices()
	net.connect_decoders(parameter_set.decoding_pars)

	######################################################################################
	# Connect Network
	# =====================================================================================
	net.connect_populations(parameter_set.connection_pars)

	######################################################################################
	# Simulate
	# =====================================================================================
	if parameter_set.kernel_pars.transient_t:
		print "\nTransient time = {0} ms".format(str(parameter_set.kernel_pars.transient_t))
		net.simulate(parameter_set.kernel_pars.transient_t)
		net.flush_records(decoders=True)
		enc_layer.flush_records()

	if parameter_set.input_pars.noise.resolution == 0.1:
		net.simulate(parameter_set.kernel_pars.sim_time)  # + 0.1)
	else:
		net.simulate(parameter_set.kernel_pars.sim_time + 0.1)
	######################################################################################
	# Extract and store data
	# ===================================================================================
	net.extract_population_activity()
	net.extract_network_activity()
	net.flush_records(decoders=False)

	enc_layer.extract_encoder_activity()
	enc_layer.flush_records()

	#######################################################
	# Analyse / plot response to train set
	# ======================================================
	results = dict()
	results['population_activity'] = population_state(net, parameter_set=parameter_set,
	                                                  nPairs=500, time_bin=1.,
	                                                  start=analysis_interval[0],
	                                                  stop=analysis_interval[0] + 1000.,
	                                                  plot=plot, display=display,
	                                                  save=save_path + 'Population')

	for idx, n_enc in enumerate(enc_layer.encoders):
		new_pars = ParameterSet(parameter_set.copy())
		new_pars.kernel_pars.data_prefix = 'Input Encoder {0}'.format(n_enc.name)
		results['input_activity_{0}'.format(str(idx))] = population_state(n_enc,
		                                                                  parameter_set=parameter_set,
		                                                                  nPairs=500, time_bin=1.,
		                                                                  start=analysis_interval[0],
		                                                                  stop=analysis_interval[0] + 1000.,
		                                                                  plot=plot, display=display,
		                                                                  save=save_path + 'Input')

	#######################################################################################
	# Extract response matrices
	# =====================================================================================
	# Extract merged responses
	if not empty(net.merged_populations):
		for ctr, n_pop in enumerate(net.merged_populations):
			n_pop.extract_response_matrix(start=analysis_interval[0], stop=analysis_interval[1], save=True)
			n_pop.name += str(ctr)
	# Extract from populations
	if not empty(net.state_extractors):
		for n_pop in net.populations:
			n_pop.extract_response_matrix(start=analysis_interval[0], stop=analysis_interval[1], save=True)
			if plot and debug:
				if len(n_pop.response_matrix) == 1:
					vis.plot_response(n_pop.response_matrix[0], n_pop.response_matrix[0].time_axis(), n_pop,
					                  display=display, save=save_path + n_pop.name)
				elif len(n_pop.response_matrix) > 1:
					for idx_nnn, nnn in enumerate(n_pop.response_matrix):
						vis.plot_response(nnn, nnn.time_axis(), n_pop, display=display, save=save_path + n_pop.name +
						                                                                     str(idx_nnn))

	#######################################################################################
	# Train Readouts
	# =====================================================================================
	# Set targets
	cut_off_time = parameter_set.kernel_pars.transient_t  # / parameter_set.input_pars.noise.resolution
	t_axis = np.arange(cut_off_time, total_stimulation_time, parameter_set.input_pars.noise.resolution)
	global_target = input_noise.noise_signal.time_slice(t_start=cut_off_time, t_stop=total_stimulation_time).as_array()

	# Set baseline random output (for comparison)
	input_noise_r2 = InputNoise(parameter_set.input_pars.noise,
	                            stop_time=total_stimulation_time)
	input_noise_r2.generate()
	input_noise.re_seed(parameter_set.kernel_pars.np_seed)

	baseline_out = input_noise_r2.noise_signal.time_slice(t_start=cut_off_time,
	                                                      t_stop=total_stimulation_time).as_array()

	print "\n******************************\nFading Memory Evaluation\n*******************************\nBaseline (" \
	      "random): "
	# Error
	MAE = np.mean(np.abs(baseline_out[0] - global_target[0]))
	SE = []
	for n in range(len(baseline_out[0])):
		SE.append((baseline_out[0, n] - global_target[0, n]) ** 2)
	MSE = np.mean(SE)
	NRMSE = np.sqrt(MSE) / (np.max(baseline_out) - np.min(baseline_out))
	print "\t- MAE = {0}".format(str(MAE))
	print "\t- MSE = {0}".format(str(MSE))
	print "\t -NRMSE = {0}".format(str(NRMSE))

	# memory
	COV = (np.cov(global_target, baseline_out) ** 2.)
	VARS = np.var(baseline_out) * np.var(global_target)
	FMF = COV / VARS
	baseline = FMF[0, 1]
	print "\t- M[0] = {0}".format(str(FMF[0, 1]))
	results['Baseline'] = {'MAE': MAE,
	                       'MSE': MSE,
	                       'NRMSE': NRMSE,
	                       'M[0]': FMF[0, 1]}

	#################################
	# Train Readouts
	#################################
	read_pops = []
	if not empty(net.merged_populations):
		for n_pop in net.merged_populations:
			results['{0}'.format(n_pop.name)] = {}
			if hasattr(n_pop, "decoding_pars"):
				print "Population {0}".format(n_pop.name)
				read_pops.append(n_pop)
				readout_labels = n_pop.decoding_pars['readout']['labels']
				pop_readouts = n_pop.readouts

				indices = -np.arange(len(readout_labels))
				for index, readout in enumerate(n_pop.readouts):
					if index < 10:
						internal_idx = int(readout.name[-1])
					elif 10 <= index < 100:
						internal_idx = int(readout.name[-2:])
					elif 100 <= index < 1000:
						internal_idx = int(readout.name[-3:])
					elif 1000 <= index < 10000:
						internal_idx = int(readout.name[-4:])
					else:
						internal_idx = int(readout.name[-5:])

					internal_idx += 1

					if len(n_pop.response_matrix) == 1:
						response_matrix = n_pop.response_matrix[0].as_array()
						if internal_idx == 1:
							output, results_1 = fmf_readout(response_matrix, global_target, readout, internal_idx,
							                                plot=plot,
							                                display=display, save=save_path + n_pop.name + str(1))
						else:
							output, results_1 = fmf_readout(response_matrix, global_target, readout, internal_idx,
							                                plot=False, display=False, save=False)

						results['{0}'.format(n_pop.name)].update(
							{'Readout_{1}'.format(n_pop.name, str(index)): results_1})

					else:
						for resp_idx, n_response in enumerate(n_pop.response_matrix):
							response_matrix = n_response.as_array()
							if internal_idx == 1:
								output, results_1 = fmf_readout(response_matrix, global_target, readout, internal_idx,
								                                plot=plot, display=display,
								                                save=save_path + n_pop.name + str(1))
							else:
								output, results_1 = fmf_readout(response_matrix, global_target, readout, internal_idx,
								                                plot=False, display=False, save=False)

							results['{0}'.format(n_pop.name)].update(
								{'Readout_{1}_R{2}'.format(n_pop.name, str(resp_idx),
								                           str(index)): results_1})
	if not empty(net.state_extractors):
		for n_pop in net.populations:
			results['{0}'.format(n_pop.name)] = {}
			if hasattr(n_pop, "decoding_pars"):
				print "\nPopulation {0}".format(n_pop.name)
				read_pops.append(n_pop)
				readout_labels = n_pop.decoding_pars['readout']['labels']
				pop_readouts = n_pop.readouts
				indices = -np.arange(len(readout_labels))

				if len(n_pop.response_matrix) == 1:
					for index, readout in enumerate(n_pop.readouts):
						if index < 10:
							internal_idx = int(readout.name[-1])
						elif 10 <= index < 100:
							internal_idx = int(readout.name[-2:])
						elif 100 <= index < 1000:
							internal_idx = int(readout.name[-3:])
						elif 1000 <= index < 10000:
							internal_idx = int(readout.name[-4:])
						else:
							internal_idx = int(readout.name[-5:])

						internal_idx += 1
						response_matrix = n_pop.response_matrix[0].as_array()

						if internal_idx == 1:
							output, results_1 = fmf_readout(response_matrix, global_target, readout, internal_idx,
							                                plot=True,
							                                display=display, save=save_path + n_pop.name + str(1))
						else:
							output, results_1 = fmf_readout(response_matrix, global_target, readout, internal_idx,
							                                plot=False, display=False, save=False)

						results['{0}'.format(n_pop.name)].update(
							{'Readout_{1}'.format(n_pop.name, str(index)): results_1})

				else:
					for resp_idx, n_response in enumerate(n_pop.response_matrix):
						readout_set = n_pop.readouts[resp_idx * len(indices):(resp_idx + 1) * len(indices)]
						for index, readout in enumerate(readout_set):
							if index < 10:
								internal_idx = int(readout.name[-1])
							elif 10 <= index < 100:
								internal_idx = int(readout.name[-2:])
							elif 100 <= index < 1000:
								internal_idx = int(readout.name[-3:])
							elif 1000 <= index < 10000:
								internal_idx = int(readout.name[-4:])
							else:
								internal_idx = int(readout.name[-5:])
							internal_idx += 1
							response_matrix = n_response.as_array()

							if internal_idx == 1:
								output, results_1 = fmf_readout(response_matrix, global_target, readout, internal_idx,
								                                plot=plot, display=display,
								                                save=save_path + n_pop.name + str(1))
							else:
								output, results_1 = fmf_readout(response_matrix, global_target, readout, internal_idx,
								                                plot=False, display=False, save=False)

							results['{0}'.format(n_pop.name)].update(
								{'Readout_{1}_R{2}'.format(n_pop.name, str(resp_idx),
								                           str(index)): results_1})

	for pop in read_pops:
		dx = np.min(np.diff(t_axis))
		if plot:
			globals()['fig_{0}'.format(pop.name)] = pl.figure()

		if len(pop.response_matrix) == 1:
			fmf = [results[pop.name][x]['fmf'] for idx, x in enumerate(np.sort(results[pop.name].keys()))]
			MC_trap = np.trapz(fmf, dx=1)
			MC_simp = integ.simps(fmf, dx=1)
			MC_trad = np.sum(fmf[1:])
			results[pop.name]['MC'] = {'MC_trap': MC_trap, 'MC_simp': MC_simp, 'MC_trad': MC_trad}

			if plot:
				ax_1 = globals()['fig_{0}'.format(pop.name)].add_subplot(111)
				vis.plot_fmf(t_axis, fmf, ax_1, label=pop.name, display=display, save=save_path + pop.name)
		else:
			ax_ctr = 0
			for resp_idx, n_response in enumerate(pop.response_matrix):
				ax_ctr += 1
				fmf = [results[pop.name][x]['fmf'] for idx, x in enumerate(np.sort(results[pop.name].keys())) if
				       resp_idx * len(indices) <= idx < (resp_idx + 1) * len(indices)]
				MC_trap = np.trapz(fmf, dx=1)
				MC_simp = integ.simps(fmf, dx=1)
				MC_trad = np.sum(fmf[1:])
				results[pop.name]['MC'] = {'MC_trap': MC_trap, 'MC_simp': MC_simp, 'MC_trad': MC_trad}

				if plot:
					globals()['ax1_{0}'.format(resp_idx)] = globals()['fig_{0}'.format(pop.name)].add_subplot(1,
					                                                                                          len(
						                                                                                          pop.response_matrix),
					                                                                                          ax_ctr)

					vis.plot_fmf(t_axis, fmf, globals()['ax1_{0}'.format(resp_idx)],
					         label=pop.name + 'State_{0}'.format(str(
						         resp_idx)), display=display, save=save_path + pop.name + str(resp_idx))
	if save:
		with open(path + 'Results_' + parameter_set.label, 'w') as f:
			pickle.dump(results, f)
		parameter_set.save(path + 'Parameters_' + parameter_set.label)

	return results