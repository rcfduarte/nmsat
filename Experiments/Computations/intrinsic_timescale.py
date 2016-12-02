from __init__ import *


def run(parameter_set, analysis_interval=None, population='E', plot=False, display=False, save=True):
	"""
	Analyse network dynamics when driven by Poisson input
	:param parameter_set: must be consistent with the computation, i.e. input must be poisson...
	:param analysis_interval: temporal interval to analyse (if None the entire simulation time will be used)
	:param plot: plot results - either show them or save to file
	:param display: show figures/reports
	:param save: save results (provide path to figures...)
	:return results_dictionary:
	"""
	if not isinstance(parameter_set, ParameterSet):
		if isinstance(parameter_set, basestring) or isinstance(parameter_set, dict):
			parameter_set = ParameterSet(parameter_set)
		else:
			raise TypeError("parameter_set must be ParameterSet, string with full path to parameter file or dictionary")

	if plot:
		import Modules.visualization as vis
		vis.set_global_rcParams(parameter_set.kernel_pars['mpl_path'])

	if analysis_interval is None:
		analysis_interval = [parameter_set.kernel_pars.transient_t, parameter_set.kernel_pars.sim_time]

	########################################################
	# Set kernel and simulation parameters
	#=======================================================
	spike_lists = []
	responses = []
	vms = []
	for n_trial in range(parameter_set.kernel_pars.n_trials):
		assert isinstance(parameter_set.kernel_pars['grng_seed'], list), "Provide rng seeds as a list of len == " \
		                                                                 "n_trials"
		assert len(parameter_set.kernel_pars['grng_seed']) == parameter_set.kernel_pars.n_trials, "Provide rng seeds " \
		                                                                                          "as  a list of len == " \
		                                                                                          "n_trials"

		np.random.seed(parameter_set.kernel_pars['grng_seed'][n_trial])

		print '\nTrial {0}'.format(str(n_trial))
		nest.ResetKernel()
		kernel_pars = copy_dict(extract_nestvalid_dict(parameter_set.kernel_pars.as_dict(), param_type='kernel'),
								{'grng_seed': parameter_set.kernel_pars['grng_seed'][n_trial]})
		nest.SetKernelStatus(kernel_pars)

		####################################################
		# Build network
		#===================================================
		net = Network(parameter_set.net_pars)

		for n in list(iterate_obj_list(net.populations)):
			n.randomize_initial_states('V_m',
			                           randomization_function=np.random.uniform,
			                           low=-70., high=-55.)
		# n.randomize_initial_states('V_th', randomization_function=np.random.normal, loc=-50., scale=1.)

		########################################################
		# Build and connect input
		#=======================================================
		enc_layer = EncodingLayer(parameter_set.encoding_pars)
		enc_layer.connect(parameter_set.encoding_pars, net)
		########################################################
		# Set-up Analysis
		#=======================================================
		net.connect_devices()
		decoders = DecodingLayer(parameter_set.decoding_pars, net_obj=net)
		#######################################################
		# Connect Network
		#======================================================
		net.connect_populations(parameter_set.connection_pars)
		#######################################################
		# Simulate
		#======================================================
		if parameter_set.kernel_pars.transient_t:
			net.simulate(parameter_set.kernel_pars.transient_t)
			net.flush_records()

		net.simulate(parameter_set.kernel_pars.sim_time + 1.)  # +1 to acquire last step...
		#######################################################
		# Extract and store data
		#======================================================
		net.extract_population_activity()
		net.extract_network_activity()
		net.flush_records()
		#######################################################
		# Analyse / plot data
		#======================================================
		if population:
			pop_names = list(iterate_obj_list(net.population_names))
			pop_objs = list(iterate_obj_list(net.populations))
			pop_idx = pop_names.index(population)
			p = pop_objs[pop_idx]
		else:
			pop_idx = 0
			p = net.merge_subpopulations(sub_populations=net.populations, name='Global')
			gids = []
			new_SpkList = SpikeList([], [], 0., parameter_set.kernel_pars.sim_time,
			                              np.sum(list(iterate_obj_list(net.n_neurons))))
			for n in list(iterate_obj_list(net.spiking_activity)):
				gids.append(n.id_list)
				for idd in n.id_list:
					new_SpkList.append(idd, n.spiketrains[idd])
			p.spiking_activity = new_SpkList

			for n in list(iterate_obj_list(net.analog_activity)):
				p.analog_activity.append(n)

			for n in list(iterate_obj_list(net.populations)):
				if not gids:
					gids.append(np.array(n.gids))

		if analysis_interval is not None:
			spike_lists.append(p.spiking_activity.time_slice(analysis_interval[0], analysis_interval[1]))
		else:
			spike_lists.append(p.spiking_activity)

		vars = parameter_set.decoding_pars.state_extractor['state_variable']

		for rec_idx, rec_var in enumerate(vars):
			if rec_var == 'V_m':
				t_axis, state = decoders.extractors[pop_idx].compile_state_matrix()
				ids = np.random.randint(0, state.shape[0], parameter_set.kernel_pars.neurons_per_trial)
				vms.append(state[ids, :])
			elif rec_var == 'spikes':
				t_axis, state = decoders.extractors[pop_idx].compile_state_matrix()
				ids = np.random.randint(0, state.shape[0], parameter_set.kernel_pars.neurons_per_trial)
				responses.append(state[ids, :])

	## Single neuron spike counts Autocorrelation fit
	tbin = parameter_set.kernel_pars.time_bin
	n_trial = parameter_set.kernel_pars.neurons_per_trial
	counts = get_total_counts(spike_lists, time_bin=tbin, n_per_trial=n_trial)
	acc = cross_trial_cc(counts)

	time_axis = np.arange(0.,  analysis_interval[1], tbin)
	initial_guess = 1., 1., 10.
	fit_params, _ = opt.leastsq(err_func, initial_guess, args=(time_axis, np.mean(acc, 0), acc_function))

	error = np.sum((np.mean(acc, 0) - acc_function(time_axis, *fit_params))**2)

	## Population Rate autocorrelation fit
	rates = np.array([np.mean(ll.firing_rate(tbin), 0) for ll in spike_lists])
	acc_rate = cross_trial_cc(rates)
	fit_rates, _ = opt.leastsq(err_func, initial_guess, args=(time_axis, np.mean(acc_rate, 0), acc_function))

	error_rates = np.sum((np.mean(acc_rate, 0) - acc_function(time_axis, *fit_rates))**2)

	if list(responses):
		## Full response autocorrelation fit
		response = np.concatenate(responses)
		acc_resp = cross_trial_cc(response)
		time_axis_resp = np.arange(0., analysis_interval[1], 1.)
		fit_resp, _ = opt.leastsq(err_func, initial_guess, args=(time_axis_resp, np.mean(acc_resp, 0), acc_function))

		error_resp = np.sum((np.mean(acc_resp, 0) - acc_function(time_axis_resp, *fit_resp))**2)

	if list(vms):
		vms = np.concatenate(vms)
		acc_vms = cross_trial_cc(vms)
		time_axis_vm = np.arange(0., analysis_interval[1], 1.)
		fit_vm, _ = opt.leastsq(err_func, initial_guess, args=(time_axis_vm, np.mean(acc_vms, 0), acc_function))

		error_vm = np.sum((np.mean(acc_vms, 0) - acc_function(time_axis_vm, *fit_vm)) ** 2)

	if save:
		path = parameter_set.kernel_pars.data_path + \
		       parameter_set.kernel_pars.data_prefix + '/'
		save_path = path + parameter_set.label
		if not os.path.exists(path):
			os.mkdir(path)
	else:
		save_path = False

	if plot:
		from Modules.visualization import plot_acc

		plot_acc(time_axis, acc, fit_params, acc_function, title=r'Single Neuron Counts ($y_{i}(t)$)',
		         ax=None, display=display, save=str(save_path)+'counts')
		plot_acc(time_axis, acc_rate, fit_rates, acc_function, title=r'Population Rates ($r(t)$)',
		         ax=None, display=display, save=str(save_path)+'rates')
		if list(responses):
			plot_acc(time_axis_resp, acc_resp, fit_resp, acc_function, title=r'Neuron State ($x_{i}(t)$)',
			         ax=None, display=display, save=str(save_path)+'responses')
		if list(vms):
			plot_acc(time_axis_vm, acc_vms, fit_vm, acc_function, title=r'Membrane Potential ($V_{i}(t)$)',
			         ax=None, display=display, save=str(save_path)+'vms')

	#######################################################
	# Save data
	#======================================================
	results = dict(single={}, rate={}, response={}, vms={})

	if list(acc):
		results['single']['counts'] = counts
		results['single']['accs'] = acc
		results['single']['time_axis'] = time_axis
		results['single']['initial_guess'] = initial_guess
		results['single']['fit_params'] = fit_params
		results['single']['MSE'] = error

	if list(acc_rate):
		results['rate']['rates'] = rates
		results['rate']['accs'] = acc_rate
		results['rate']['time_axis'] = time_axis
		results['rate']['initial_guess'] = initial_guess
		results['rate']['fit_params'] = fit_rates
		results['rate']['MSE'] = error_rates

	if list(responses):
		results['response']['resp'] = response
		results['response']['accs'] = acc_resp
		results['response']['time_axis'] = time_axis_resp
		results['response']['initial_guess'] = initial_guess
		results['response']['fit_params'] = fit_resp
		results['response']['MSE'] = error_resp

	if list(vms):
		results['vms']['resp'] = vms
		results['vms']['accs'] = acc_vms
		results['vms']['time_axis'] = time_axis_vm
		results['vms']['initial_guess'] = initial_guess
		results['vms']['fit_params'] = fit_vm
		results['vms']['MSE'] = error_vm

	if save:
		with open(path + 'Results_' + parameter_set.label, 'w') as f:
			pickle.dump(results, f)
		parameter_set.save(path+'Parameters_'+parameter_set.label)

	return results