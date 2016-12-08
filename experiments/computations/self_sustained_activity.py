from __init__ import *


def run(parameter_set, analysis_interval=None, plot=False, display=False, save=True, debug=False):
	"""
	Analyse network dynamics how long can the network sustain its activity
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
		import modules.visualization as vis

		vis.set_global_rcParams(parameter_set.kernel_pars['mpl_path'])

	np.random.seed(parameter_set.kernel_pars['np_seed'])

	analysis_interval = [parameter_set.kernel_pars.start_state_analysis,
	                     parameter_set.kernel_pars.transient_t]
	# to analyse the state of the circuit...
	########################################################
	# Set kernel and simulation parameters
	# =======================================================
	if save:
		path = parameter_set.kernel_pars.data_path + \
		       parameter_set.kernel_pars.data_prefix + '/'
		save_path = path + parameter_set.label
		if not os.path.exists(path):
			os.mkdir(path)
	else:
		save_path = False

	print '\nRuning ParameterSet {0}'.format(parameter_set.label)
	nest.ResetKernel()
	nest.SetKernelStatus(extract_nestvalid_dict(parameter_set.kernel_pars.as_dict(), param_type='kernel'))

	####################################################
	# Build network
	# ===================================================
	net = Network(parameter_set.net_pars)
	for n in list(iterate_obj_list(net.populations)):
		n.randomize_initial_states('V_m',
		                           randomization_function=np.random.uniform,
		                           low=-70., high=-55.)
	# n.randomize_initial_states('V_th', randomization_function=np.random.normal, loc=-50., scale=1.)

	##########################################################
	# Build and connect input
	# =========================================================
	t_axis = np.arange(0., parameter_set.kernel_pars.transient_t + parameter_set.kernel_pars.sim_time, 1.)
	decay = t_axis[parameter_set.kernel_pars.transient_t + parameter_set.kernel_pars.start_state_analysis:]
	decay_t = np.arange(0., float(len(decay)), 1.)
	initial_rate = parameter_set.kernel_pars.base_rate

	signal_array = np.ones_like(t_axis) * initial_rate
	signal_array[parameter_set.kernel_pars.transient_t +
	             parameter_set.kernel_pars.start_state_analysis:] = initial_rate * np.exp(-decay_t /
	                                                                                      parameter_set.kernel_pars.input_decay_tau)
	input_signal = InputSignal()
	input_signal.load_signal(signal_array, dt=1., onset=0.)

	############################################################
	# Encode Input
	# ===========================================================
	enc_layer = EncodingLayer(parameter_set.encoding_pars, signal=input_signal)
	enc_layer.connect(parameter_set.encoding_pars, net)

	if debug:
		# Parse activity data
		encoder_activity = [enc_layer.encoders[x].spiking_activity for x in
		                    range(parameter_set.encoding_pars.encoder.N)]
		encoder_size = [enc_layer.encoders[x].size for x in range(parameter_set.encoding_pars.encoder.N)]
		gids = []
		new_SpkList = SpikeList([], [], 0., parameter_set.kernel_pars.sim_time, np.sum(encoder_size))
		for ii, n in enumerate(encoder_activity):
			gids.append(n.id_list)
			if ii > 0:
				gids[1] += gids[ii - 1][-1] + 1
				id_list = n.id_list + (gids[ii - 1][-1] + 1)
				for idd in id_list:
					new_SpkList.append(idd, n.spiketrains[idd - (gids[ii - 1][-1] + 1)])
			else:
				for idd in n.id_list:
					new_SpkList.append(idd, n.spiketrains[idd])

		# Activity Plots
		rp = vis.SpikePlots(new_SpkList, start=100., stop=parameter_set.kernel_pars.transient_t + 100.)
		rp.print_activity_report(label='Input', n_pairs=500)
		plot_props = {'xlabel': 'Time [ms]', 'ylabel': 'Neuron', 'suptitle': 'Input',
		              'color': 'blue', 'linewidth': 1.0, 'linestyle': '-'}
		rp.dot_display(gids=gids, colors=['b', 'r'], with_rate=True, display=True, **plot_props)

		if save and save_path is not None:
			pl.savefig(save_path + '_input_activity')
	############################################################
	# Set-up Analysis
	# ===========================================================
	net.connect_devices()

	#############################################################
	# Connect Network
	# ===========================================================
	net.connect_populations(parameter_set.connection_pars)

	#######################################################
	# Simulate
	# ======================================================
	results = {}
	if parameter_set.kernel_pars.transient_t:
		net.simulate(parameter_set.kernel_pars.transient_t + 0.1)
		net.extract_population_activity()
		net.extract_network_activity()
		net.flush_records()

		results.update(population_state(net, parameter_set=parameter_set,
		                                nPairs=500, time_bin=1.,
		                                start=analysis_interval[0],
		                                stop=analysis_interval[1],
		                                plot=plot, display=display,
		                                save=save_path))

	t_max = parameter_set.kernel_pars.transient_t
	limit = parameter_set.kernel_pars.transient_t + 100000.
	while t_max >= (nest.GetKernelStatus()['time'] - 100.) and nest.GetKernelStatus()['time'] < limit:

		net.simulate(parameter_set.kernel_pars.sim_time)

		spk_det = [nest.GetStatus(net.device_gids[xx][0])[0]['events']['times'] for xx in range(len(
			net.population_names))]
		T_max = []
		for n_pop in list(itertools.chain(*spk_det)):
			if n_pop:
				T_max.append(np.max(n_pop))
		if T_max:
			t_max = np.max(T_max)
		else:
			t_max = 0.

	#######################################################
	# Extract and store data
	# ======================================================
	net.extract_population_activity()
	net.extract_network_activity()
	net.flush_records()

	#######################################################
	# Analyse / plot data
	# ======================================================
	results.update(ssa_lifetime(net, parameter_set,
	                            input_off=parameter_set.kernel_pars.t_off,
	                            display=display))

	if plot:
		fig = pl.figure()

		ax1 = pl.subplot2grid((30, 1), (0, 0), rowspan=23, colspan=1)
		ax2 = ax1.twinx()
		ax3 = pl.subplot2grid((30, 1), (24, 0), rowspan=5, colspan=1, sharex=ax1)

		plot_props = {'lw': 3, 'c': 'k'}
		ip = vis.InputPlots(input_obj=input_signal)
		ip.plot_input_signal(ax=ax2, save=False, display=False, **plot_props)
		ax2.set_ylim(0, parameter_set.kernel_pars.base_rate + 100)

		# Parse activity data
		gids = []
		new_SpkList = SpikeList([], [], 0.,
		                        parameter_set.kernel_pars.sim_time,
		                        np.sum(list(iterate_obj_list(net.n_neurons))))
		for n in list(iterate_obj_list(net.spiking_activity)):
			gids.append(n.id_list)
			for idd in n.id_list:
				new_SpkList.append(idd, n.spiketrains[idd])

		# Activity Plots
		rp = vis.SpikePlots(new_SpkList, start=parameter_set.kernel_pars.transient_t,
		                    stop=parameter_set.kernel_pars.sim_time)
		rp.print_activity_report(label='Self-Sustained Activity - {0}={1}', n_pairs=500)
		plot_props = {'xlabel': 'Time [ms]', 'ylabel': 'Neuron',
		              'suptitle': 'Self-Sustained Activity - ${0}={1}$'.format(r'\tau_{ssa}', str(results['ssa'][
			                                                                                          'Global_ssa'][
			                                                                                          'tau'])),
		              'color': 'blue', 'linewidth': 1.0, 'linestyle': '-'}
		rp.dot_display(gids=gids, colors=['b', 'r'], with_rate=True, display=display, ax=[ax1, ax3], fig=fig,
		               save=save_path, **plot_props)

	if save:
		with open(path + 'Results_' + parameter_set.label, 'w') as f:
			pickle.dump(results, f)
		parameter_set.save(path + 'Parameters_' + parameter_set.label)

	return results