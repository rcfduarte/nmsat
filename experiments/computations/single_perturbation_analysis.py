from __init__ import *


def run(parameter_set, plot=False, display=False, save=False, debug=False):
	"""
	"""
	if not isinstance(parameter_set, ParameterSet):
		if isinstance(parameter_set, basestring) or isinstance(parameter_set, dict):
			parameter_set = ParameterSet(parameter_set)
		else:
			raise TypeError("parameter_set must be ParameterSet, string with full path to parameter file or dictionary")

	###################################################################################
	# Setup extra variables and parameters
	# =================================================================================
	if plot:
		import modules.visualization as vis
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
	results = dict()
	##################################################################################
	# Set kernel and simulation parameters
	# ================================================================================
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

	net.connect_populations(parameter_set.connection_pars)

	#######################################################################################
	# Set-up Analysis
	# =====================================================================================
	net.connect_devices()
	net.connect_decoders(parameter_set.decoding_pars)

	###################################################
	# Build and Connect copy network
	# ==================================================
	clone_net = net.clone(parameter_set, devices=True, decoders=True)

	if debug:
		net.extract_synaptic_weights()
		net.extract_synaptic_delays()
		clone_net.extract_synaptic_weights()
		clone_net.extract_synaptic_delays()
		for idx, k in enumerate(net.synaptic_weights.keys()):
			print np.array_equal(np.array(net.synaptic_weights[k].todense()),
			                     np.array(clone_net.synaptic_weights[(k[0] + '_clone', k[1] + '_clone')].todense()))

	# if plot and debug:
	# 	fig_W = pl.figure()
	# 	topology = vis.TopologyPlots(parameter_set.connection_pars, net, colors=['b', 'r'])
	# 	#topology.print_network(depth=3)
	# 	ax1 = pl.subplot2grid((6, 6), (0, 0), rowspan=4, colspan=4)
	# 	ax2 = pl.subplot2grid((6, 6), (5, 0), rowspan=1, colspan=4)
	# 	ax3 = pl.subplot2grid((6, 6), (0, 5), rowspan=4, colspan=1)
	# 	ax4 = pl.subplot2grid((6, 6), (5, 5), rowspan=1, colspan=1)
	# 	topology.plot_connectivity(parameter_set.connection_pars.synapse_types,
	#  	                           ax=[ax1, ax3, ax2, ax4],
	#  	                           display=display, save=save_path)
	# if plot and debug:
	# 	fig_W = pl.figure()
	# 	topology = vis.TopologyPlots(parameter_set.connection_pars, list(iterate_obj_list(clone_net.populations)),
	#  	                         colors=['b', 'r'])
	# 	topology.print_network(depth=3)
	# 	ax1 = pl.subplot2grid((6, 6), (0, 0), rowspan=4, colspan=4)
	# 	ax2 = pl.subplot2grid((6, 6), (5, 0), rowspan=1, colspan=4)
	# 	ax3 = pl.subplot2grid((6, 6), (0, 5), rowspan=4, colspan=1)
	# 	ax4 = pl.subplot2grid((6, 6), (5, 5), rowspan=1, colspan=1)
	# 	topology.plot_connectivity([('E_copy', 'E_copy')],
	#  	                           ax=[ax1, ax3, ax2, ax4],
	#  	                           display=display, save=save_path)
	##########################################################
	# Build and connect input
	# =========================================================
	enc_layer = EncodingLayer()
	enc_layer.connect_clone(parameter_set.encoding_pars, net, clone_net)

	perturb_population_idx = net.population_names.index(parameter_set.kernel_pars.perturb_population)
	perturb_gids = np.random.permutation(clone_net.populations[perturb_population_idx].gids)[
	               :parameter_set.kernel_pars.perturb_n]
	perturbation_generator = nest.Create('spike_generator', 1, {'spike_times': [
		parameter_set.kernel_pars.perturbation_time + parameter_set.kernel_pars.transient_t],
		'spike_weights': [
			parameter_set.kernel_pars.perturbation_spike_weight]})
	nest.Connect(perturbation_generator, list(perturb_gids),
	             syn_spec=parameter_set.connection_pars.syn_specs[0])  # {'receptor_type': 1})
	######################################################################################
	# Simulate (Initial Transient)
	# ====================================================================================
	if parameter_set.kernel_pars.transient_t:
		print "\nTransient time = {0} ms".format(str(parameter_set.kernel_pars.transient_t))
		nest.Simulate(parameter_set.kernel_pars.transient_t)

		net.extract_population_activity(t_start=0., t_stop=parameter_set.kernel_pars.transient_t -
		                                                   parameter_set.kernel_pars.resolution)
		clone_net.extract_population_activity(t_start=0., t_stop=parameter_set.kernel_pars.transient_t -
		                                                         parameter_set.kernel_pars.resolution)
		net.extract_network_activity()
		clone_net.extract_network_activity()

		# sanity check
		activity = []
		for spikes in net.spiking_activity:
			activity.append(spikes.mean_rate())
		if not np.mean(activity) > 0:
			raise ValueError("No activity recorded in main network! Stopping simulation..")

		activity = []
		for spikes in clone_net.spiking_activity:
			activity.append(spikes.mean_rate())
		if not np.mean(activity) > 0:
			raise ValueError("No activity recorded in clone network! Stopping simulation..")

		analysis_interval = [0, parameter_set.kernel_pars.transient_t]
		results['population_activity'] = population_state(net, parameter_set=parameter_set,
		                                                  nPairs=500, time_bin=1.,
		                                                  start=analysis_interval[0],
		                                                  stop=analysis_interval[1],
		                                                  plot=plot, display=display,
		                                                  save=save_path)
		net.flush_records()
		clone_net.flush_records()

	# enc_layer.flush_records()

	print "\nSimulation time = {0} ms".format(str(parameter_set.kernel_pars.sim_time))
	nest.Simulate(parameter_set.kernel_pars.sim_time)

	######################################################################################
	# Extract and store data
	# ===================================================================================
	net.extract_population_activity(t_start=parameter_set.kernel_pars.transient_t,
	                                t_stop=parameter_set.kernel_pars.transient_t + parameter_set.kernel_pars.sim_time)
	net.extract_network_activity()
	net.flush_records(decoders=False)

	clone_net.extract_population_activity(t_start=parameter_set.kernel_pars.transient_t,
	                                      t_stop=parameter_set.kernel_pars.transient_t + parameter_set.kernel_pars.sim_time)
	clone_net.extract_network_activity()
	clone_net.flush_records(decoders=False)

	# enc_layer.extract_encoder_activity()
	# enc_layer.flush_records()

	#######################################################################################
	# Extract response matrices
	# =====================================================================================
	analysis_interval = [parameter_set.kernel_pars.transient_t,
	                     parameter_set.kernel_pars.sim_time + parameter_set.kernel_pars.transient_t]
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
					                  display=display, save=save_path)
				elif len(n_pop.response_matrix) > 1:
					for idx_nnn, nnn in enumerate(n_pop.response_matrix):
						vis.plot_response(nnn, nnn.time_axis(), n_pop, display=display, save=save_path)
	# Extract merged responses
	if not empty(clone_net.merged_populations):
		for ctr, n_pop in enumerate(clone_net.merged_populations):
			n_pop.extract_response_matrix(start=analysis_interval[0], stop=analysis_interval[1], save=True)
			n_pop.name += str(ctr)
	# Extract from populations
	if not empty(clone_net.state_extractors):
		for n_pop in clone_net.populations:
			n_pop.extract_response_matrix(start=analysis_interval[0], stop=analysis_interval[1], save=True)
			if plot and debug:
				if len(n_pop.response_matrix) == 1:
					vis.plot_response(n_pop.response_matrix[0], n_pop.response_matrix[0].time_axis(), n_pop,
					                  display=display, save=save_path)
				elif len(n_pop.response_matrix) > 1:
					for idx_nnn, nnn in enumerate(n_pop.response_matrix):
						vis.plot_response(nnn, nnn.time_axis(), n_pop, display=display, save=save_path)

	#######################################################################################
	# Analyse results
	# =====================================================================================
	results['perturbation'] = {}
	results['perturbation'].update(
		analyse_state_divergence(parameter_set, net, clone_net, plot=plot, display=display, save=save_path))

	#######################################################################################
	# Save data
	# =====================================================================================
	if save:
		with open(save_path + 'Results_' + parameter_set.label, 'w') as f:
			pickle.dump(results, f)
		parameter_set.save(save_path + 'Parameters_' + parameter_set.label)
