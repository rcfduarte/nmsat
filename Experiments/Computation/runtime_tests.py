from __init__ import *

def run(parameter_set, analysis_interval=None, plot=False, display=False, save=True):
	"""
	Tests to determine the execution time of a given computation
	:return:
	"""
	import time
	start = time.time()
	if not isinstance(parameter_set, ParameterSet):
		if isinstance(parameter_set, basestring) or isinstance(parameter_set, dict):
			parameter_set = ParameterSet(parameter_set)
		else:
			raise TypeError("parameter_set must be ParameterSet, string with full path to parameter file or dictionary")

	if plot:
		import Modules.visualization as vis

		vis.set_global_rcParams(parameter_set.kernel_pars['mpl_path'])

	np.random.seed(parameter_set.kernel_pars['grng_seed'])

	if analysis_interval is None:
		analysis_interval = [parameter_set.kernel_pars.transient_t,
		                     parameter_set.kernel_pars.sim_time]

	# #######################################################
	# Set kernel and simulation parameters
	#=======================================================
	print '\nRuning ParameterSet {0}'.format(parameter_set.label)
	nest.ResetKernel()
	nest.SetKernelStatus(extract_nestvalid_dict(parameter_set.kernel_pars.as_dict(), type='kernel'))

	setup_time = time.time() - start

	####################################################
	# Build network
	#===================================================
	start_build = time.time()
	net = Network(parameter_set.net_pars)
	########################################################################################################################
	# Randomize initial variable values
	#=======================================================================================================================
	for n in list(iterate_obj_list(net.populations)):
		n.randomize_initial_states('V_m', randomization_function=np.random.uniform, low=-70., high=-55.)
	#n.randomize_initial_states('V_th', randomization_function=np.random.uniform, low=-50., high=1.)

	########################################################
	# Build and connect input
	#=======================================================
	enc_layer = EncodingLayer(parameter_set.encoding_pars)
	enc_layer.connect(parameter_set.encoding_pars, net)
	########################################################
	# Set-up Analysis
	#=======================================================
	net.connect_devices()
	#######################################################
	# Connect Network
	#======================================================
	net.connect_populations(parameter_set.connection_pars)
	end_build = time.time() - start_build
	#######################################################
	# Simulate
	#======================================================
	sim_time = time.time()
	if parameter_set.kernel_pars.transient_t:
		net.simulate(parameter_set.kernel_pars.transient_t)
		net.flush_records()

	net.simulate(parameter_set.kernel_pars.sim_time + 1.)  # +1 to acquire last step...
	end_sim = time.time()-sim_time
	#######################################################
	# Extract and store data
	#======================================================
	read_time = time.time()
	net.extract_population_activity()
	net.extract_network_activity()
	net.flush_records()
	end_read = time.time() - read_time
	#######################################################
	# Analyse / plot data
	#======================================================
	analysis_time = time.time()
	if save:
		path = parameter_set.kernel_pars.data_path + \
		       parameter_set.kernel_pars.data_prefix + '/'
		save_path = path + parameter_set.label
		if not os.path.exists(path):
			os.mkdir(path)
	else:
		save_path = False
	results = population_state(net, parameter_set=parameter_set,
	                           nPairs=500, time_bin=1.,
	                           start=analysis_interval[0],
	                           stop=analysis_interval[1],
	                           plot=plot, display=display,
	                           save=save_path)
	#######################################################
	# Save data
	#======================================================
	if save:
		with open(path + 'Results_' + parameter_set.label, 'w') as f:
			pickle.dump(results, f)
		parameter_set.save(path + 'Parameters_' + parameter_set.label)
	end_analysis = time.time() - analysis_time

	total_time = time.time() - start

	T_results = dict()
	T_results['total'] = total_time
	T_results['build'] = end_build
	T_results['setup'] = setup_time
	T_results['data_handle'] = end_read
	T_results['analysis'] = end_analysis

	if save:
		with open(path + 'TimeResults_' + parameter_set.label, 'w') as f:
			pickle.dump(T_results, f)

	return T_results, results