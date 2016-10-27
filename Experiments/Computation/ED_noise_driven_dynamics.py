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

	paths = set_storage_locations(parameter_set, save)
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

	###########################################################
	# Randomize initial states
	# =========================================================
	for idx, n in enumerate(list(iterate_obj_list(net.populations))):
		if hasattr(parameter_set.net_pars, "randomize_neuron_pars"):
			randomize = parameter_set.net_pars.randomize_neuron_pars[idx]
			for k, v in randomize.items():
				n.randomize_initial_states(k, randomization_function=v[0], **v[1])
	##########################################################
	# Build and connect input and encoder
	# =========================================================
	if hasattr(parameter_set, "input_pars"):
		input_seq = [1]
		total_stimulation_time = parameter_set.kernel_pars.sim_time + parameter_set.kernel_pars.transient_t
		input_noise = InputNoise(parameter_set.input_pars.noise,
		                         stop_time=total_stimulation_time)
		input_noise.generate()
		input_noise.re_seed(parameter_set.kernel_pars.np_seed)
		if plot:
			inp_plot = vis.InputPlots(stim_obj=None, input_obj=None, noise_obj=input_noise)
			inp_plot.plot_noise_component(display=display, save=False)
		enc_layer = EncodingLayer(parameter_set.encoding_pars, signal=input_noise)
		enc_layer.connect(parameter_set.encoding_pars, net)
	else:
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

	net.simulate(parameter_set.kernel_pars.sim_time)  # +.1 to acquire last step...
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
	if not plot:
		summary_only = True
	else:
		summary_only = False

	results = {}
	# results = population_state(net, parameter_set, nPairs=500, time_bin=1., start=analysis_interval[0],
	#                            stop=analysis_interval[1], plot=plot, display=display,
	#                            save=paths['figures'] + paths['label'])
	#
	# if 'I_ex' in results['analog_activity']['E'].keys():
	# 	I_ex = results['analog_activity']['E']['I_ex']
	# 	I_in = results['analog_activity']['E']['I_in']
	# 	time_axis = np.arange(analysis_interval[0], analysis_interval[1], 0.1)
	# 	if plot:
	# 		vis.plot_synaptic_currents(I_ex, I_in, time_axis)
	# if 'mean_I_ex' in results['analog_activity']['E'].keys():
	# 	inh = np.array(results['analog_activity']['E']['mean_I_in'])
	# 	exc = np.array(results['analog_activity']['E']['mean_I_ex'])
	# 	ei_ratios = np.abs(np.abs(inh) - np.abs(exc))
	# 	print np.mean(ei_ratios)
	# 	results['analog_activity']['E']['IE_ratio'] = np.mean(ei_ratios)
	# 	results['analog_activity']['E']['IE_ratios'] = ei_ratios
	# #######################################################
	# # Save data
	# # ======================================================
	# if save:
	# 	with open(path + 'Results_' + parameter_set.label, 'w') as f:
	# 		pickle.dump(results, f)
	# 	parameter_set.save(path + 'Parameters_' + parameter_set.label)

	results.update(characterize_population_activity(net, parameter_set, analysis_interval, epochs=None,
	                                           time_bin=1., summary_only=summary_only, complete=True,
	                                           time_resolved=False,
	                                           color_map='jet', plot=plot, display=display, save=save_path))
	if 'I_ex' in results['analog_activity']['E'].keys():
		I_ex = results['analog_activity']['E']['I_ex']
		I_in = results['analog_activity']['E']['I_in']
		time_axis = np.arange(analysis_interval[0], analysis_interval[1], 0.1)
		if plot:
			vis.plot_synaptic_currents(I_ex, I_in, time_axis)
	if 'mean_I_ex' in results['analog_activity']['E'].keys():
		inh = np.array(results['analog_activity']['E']['mean_I_in'])
		exc = np.array(results['analog_activity']['E']['mean_I_ex'])
		ei_ratios = np.abs(np.abs(inh) - np.abs(exc))
		print np.mean(ei_ratios)
		results['analog_activity']['E']['IE_ratio'] = np.mean(ei_ratios)
		results['analog_activity']['E']['IE_ratios'] = ei_ratios
	#######################################################
	# Save data
	# ======================================================
	if save:
		with open(path + 'Results_' + parameter_set.label, 'w') as f:
			pickle.dump(results, f)
		parameter_set.save(path + 'Parameters_' + parameter_set.label)
	return results