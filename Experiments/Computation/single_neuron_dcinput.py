__author__ = 'duarte'
from __init__ import *


def run(parameter_set, analysis_interval=None, plot=False, display=False, save=True):
	"""
	Analyse single neuron response profile in presence of variable amplitude somatic DC injection
	:return:
	"""
	if not isinstance(parameter_set, ParameterSet):
		if isinstance(parameter_set, basestring) or isinstance(parameter_set, dict):
			parameter_set = ParameterSet(parameter_set)
		else:
			raise TypeError("parameter_set must be ParameterSet, string with "
			                "full path to parameter file or dictionary")

	if plot:
		import Modules.visualization as vis

		vis.set_global_rcParams(parameter_set.kernel_pars['mpl_path'])

	np.random.seed(parameter_set.kernel_pars['np_seed'])

	if analysis_interval is None:
		analysis_interval = [parameter_set.kernel_pars.transient_t,
		                     parameter_set.kernel_pars.sim_time]

	########################################################
	# Set kernel and simulation parameters
	# =======================================================
	print '\nRuning ParameterSet {0}'.format(parameter_set.label)
	nest.ResetKernel()
	nest.SetKernelStatus(extract_nestvalid_dict(parameter_set.kernel_pars.as_dict(),
	                                            type='kernel'))

	####################################################
	# Build network
	# ===================================================
	net = Network(parameter_set.net_pars)
	####################################################
	# Randomize initial variable values
	# ===================================================
	for idx, n in enumerate(list(iterate_obj_list(net.populations))):
		if hasattr(parameter_set.net_pars, "randomize_neuron_pars"):
			randomize = parameter_set.net_pars.randomize_neuron_pars[idx]
			for k, v in randomize.items():
				n.randomize_initial_states(k, randomization_function=v[0], **v[1])

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
	# Simulate
	# ======================================================
	if parameter_set.kernel_pars.transient_t:
		net.simulate(parameter_set.kernel_pars.transient_t)
		net.flush_records()

	net.simulate(parameter_set.kernel_pars.sim_time + 1.)
	#######################################################
	# Extract and store data
	# ======================================================
	net.extract_population_activity()
	net.extract_network_activity()
	net.flush_records()
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
	compact_results = dict()
	for idd, nam in enumerate(net.population_names):
		results.update({nam: {}})
		compact_results.update({nam: {}})
		results[nam] = single_neuron_dcresponse(net.populations[idd],
			                                    parameter_set, start=analysis_interval[0],
			                                    stop=analysis_interval[1], plot=plot,
			                                    display=display, save=save_path)
		idx = np.min(np.where(results[nam]['output_rate']))
		print "Rate range for neuron {0} = [{1}, {2}] Hz".format(str(nam), str(np.min(results[nam]['output_rate'][
			                                                                              results[nam][
				                                                                              'output_rate'] > 0.])),
		                                                         str(np.max(results[nam]['output_rate'][
			                                                                    results[nam]['output_rate'] > 0.])))
		results[nam].update({'min_rate': np.min(results[nam]['output_rate'][results[nam]['output_rate'] > 0.]),
		                     'max_rate': np.max(results[nam]['output_rate'][results[nam]['output_rate'] > 0.])})
		print "Rheobase Current for neuron {0} in [{1}, {2}]".format(str(nam), str(results[nam]['input_amplitudes'][
		                                                    idx - 1]), str(results[nam]['input_amplitudes'][idx]))

		x = np.array(results[nam]['input_amplitudes'])
		y = np.array(results[nam]['output_rate'])
		iddxs = np.where(y)
		slope2, intercept, r_value, p_value, std_err = st.linregress(x[iddxs], y[iddxs])
		print "fI Slope for neuron {0} = {1} Hz/nA [linreg method]".format(nam, str(slope2 * 1000.))

		results[nam].update({'fI_slope': slope2 * 1000., 'I_rh': [results[nam]['input_amplitudes'][idx - 1],
		                                                           results[nam]['input_amplitudes'][idx]]})

		compact_results[nam].update({'output_rate': results[nam]['output_rate'],
		                             'I_rh': results[nam]['I_rh'],
		                             'fI_slope': results[nam]['fI_slope'],
		                             'min_rate': results[nam]['min_rate'],
		                             'max_rate': results[nam]['max_rate'],
		                             'AI': results[nam]['AI']})
	#######################################################
	# Save data
	# ======================================================
	if save:
		parameter_set.save(path + 'Parameters_' + parameter_set.label)
		with open(path + 'Results_' + parameter_set.label, 'w') as f:
			pickle.dump(compact_results, f)

	return results