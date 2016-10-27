__author__ = 'duarte'
from __init__ import *


def run(parameter_set, analysis_interval=None, plot=False, display=False, save=True):
	"""
	Analyse network dynamics when driven by Poisson input
	:param parameter_set: must be consistent with the computation, i.e. input must be poisson...
	:param net: network object (if pre-generated)
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

	np.random.seed(parameter_set.kernel_pars['np_seed'])

	if analysis_interval is None:
		analysis_interval = [parameter_set.kernel_pars.transient_t,
		                     parameter_set.kernel_pars.sim_time + parameter_set.kernel_pars.transient_t]

	########################################################
	# Set kernel and simulation parameters
	# =======================================================
	print '\nRuning ParameterSet {0}'.format(parameter_set.label)
	nest.ResetKernel()
	nest.SetKernelStatus(extract_nestvalid_dict(parameter_set.kernel_pars.as_dict(), type='kernel'))

	####################################################
	# Build network
	# ===================================================
	net = Network(parameter_set.net_pars)

	net.populations[0].randomize_initial_states('V_m',
	                                            randomization_function=np.random.uniform,
	                                            low=-70., high=-55.)
	net.populations[1].randomize_initial_states('V_m',
	                             randomization_function=np.random.uniform,
	                             low=-70., high=-55.)
	# net.populations[0].randomize_initial_states('E_L',
	#                             randomization_function=np.random.uniform,
	#                             low=-80., high=-60.)

	########################################################
	# Build and connect input
	# =======================================================
	enc_layer = EncodingLayer(parameter_set.encoding_pars)
	enc_layer.connect(parameter_set.encoding_pars, net)
	########################################################
	# Set-up Analysis
	# =======================================================
	net.connect_devices()
	#######################################################
	# Connect Network
	# ======================================================
	net.connect_populations(parameter_set.connection_pars)
	#######################################################
	# Simulate
	# ======================================================
	if parameter_set.kernel_pars.transient_t:
		net.simulate(parameter_set.kernel_pars.transient_t)
		net.flush_records()

	net.simulate(parameter_set.kernel_pars.sim_time + 0.1)  # +1 to acquire last step...
	#######################################################
	# Extract and store data
	# ======================================================
	net.extract_population_activity()
	net.extract_network_activity()
	# net.flush_records()
	#######################################################
	# Analyse / plot data
	# ======================================================
	if save:
		path = parameter_set.kernel_pars.data_path + \
		       parameter_set.kernel_pars.data_prefix + '/'
		save_path = path + parameter_set.label
		if not os.path.exists(path):
			os.mkdir(path)
	else:
		save_path = False
	results = dict()
	input_pops = ['E_input', 'I_input']
	for idd, nam in enumerate(net.population_names):
		if nam not in input_pops:
			results.update({nam: {}})
			results[nam] = single_neuron_responses(net.populations[idd],
			                                       parameter_set, pop_idx=idd,
			                                       start=analysis_interval[0],
			                                       stop=analysis_interval[1],
			                                       plot=plot, display=display,
			                                       save=save_path)
			if results[nam]['rate']:
				print 'Output Rate [{0}] = {1} spikes/s'.format(str(nam), str(results[nam]['rate']))
	#######################################################
	# Save data
	# ======================================================
	if save:
		with open(path + 'Results_' + parameter_set.label, 'w') as f:
			pickle.dump(results, f)
		parameter_set.save(path + 'Parameters_' + parameter_set.label)

	return results