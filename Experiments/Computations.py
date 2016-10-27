__author__ = 'duarte'
import cPickle as pickle
import numpy as np
import scipy.optimize as opt
import scipy.integrate as integ
import scipy.stats as st
from scipy.sparse import coo_matrix
import os
import nest
from Modules.parameters import *
from Modules.net_architect import *
from Modules.input_architect import *
from Modules.analysis import *
from Modules.visualization import progress_bar
import pylab as pl


def noise_driven_dynamics(parameter_set, net=None, analysis_interval=None, plot=False, display=False, save=True):
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

	########################################################################################################################
	# Randomize initial variable values
	# =======================================================================================================================
	for n in list(iterate_obj_list(net.populations)):
		n.randomize_initial_states('V_m', randomization_function=np.random.uniform, low=-70., high=-55.)
	# n.randomize_initial_states('V_th', randomization_function=np.random.uniform, low=-50., high=1.)

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

	net.simulate(parameter_set.kernel_pars.sim_time)  # +.1 to acquire last step...
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
	if not plot:
		summary_only = True
	else:
		summary_only = False

	results = characterize_population_activity(net, parameter_set, analysis_interval, epochs=None,
	                                           time_bin=1., summary_only=summary_only, complete=True,
	                                           time_resolved=False,
	                                           color_map='jet', plot=plot, display=display, save=save_path)
	#######################################################
	# Save data
	# ======================================================
	if save:
		with open(path + 'Results_' + parameter_set.label, 'w') as f:
			pickle.dump(results, f)
		parameter_set.save(path + 'Parameters_' + parameter_set.label)

	return results


def ED_noise_driven_dynamics(parameter_set, analysis_interval=None, plot=False, display=False, save=True):
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


########################################################################################################################
def single_neuron_dcinput(parameter_set, analysis_interval=None, plot=False,
                                 display=False, save=True):
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


def single_neuron_noise_driven(parameter_set, analysis_interval=None, plot=False, display=False, save=True):
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


def intrinsic_timescale(parameter_set, analysis_interval=None, population='E', plot=False,
                        display=False, save=True):
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
		kernel_pars = copy_dict(extract_nestvalid_dict(parameter_set.kernel_pars.as_dict(), type='kernel'),
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


def self_sustained_activity(parameter_set, analysis_interval=None, plot=False, display=False, save=True, debug=False):
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
		import Modules.visualization as vis

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
	nest.SetKernelStatus(extract_nestvalid_dict(parameter_set.kernel_pars.as_dict(), type='kernel'))

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


def fading_memory_function(parameter_set, analysis_interval=None, plot=False, display=False, save=False, debug=False):
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


########################################################################################################################
def runtime_tests(parameter_set, analysis_interval=None, plot=False, display=False, save=True):
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


########################################################################################################################
def AD_stimulus_driven(parameter_set, plot=False, display=False, save=False, debug=False, online=True):
	"""
	Run the AD test chain.. (Simulation sequence: transient phase, unique input sequence - determine rank,
	train phase, test phase
	:param parameter_set:
	:param analysis_interval:
	:param plot:
	:param display:
	:param save:
	:param debug:
	:return:
	"""
	analysis_interval = None
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
		import matplotlib.pyplot as pl
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
		n.randomize_initial_states('V_m', randomization_function=np.random.uniform, low=0.0, high=15.)

	###################################################################################
	# Build and connect input
	# =================================================================================
	# Create StimulusSet
	stim_set_time = time.time()
	stim = StimulusSet(parameter_set, unique_set=True)
	stim.create_set(parameter_set.stim_pars.full_set_length)
	stim.discard_from_set(parameter_set.stim_pars.transient_set_length)
	stim.divide_set(parameter_set.stim_pars.transient_set_length, parameter_set.stim_pars.train_set_length,
	                parameter_set.stim_pars.test_set_length)
	print "- Elapsed Time: {0}".format(str(time.time() - stim_set_time))

	# Create InputSignalSet
	input_set_time = time.time()
	inputs = InputSignalSet(parameter_set, stim, online=online)
	if stim.transient_set_labels:
		inputs.generate_transient_set(stim)
		parameter_set.kernel_pars.transient_t = inputs.transient_stimulation_time
	inputs.generate_unique_set(stim)
	inputs.generate_train_set(stim)
	inputs.generate_test_set(stim)
	print "- Elapsed Time: {0}".format(str(time.time() - input_set_time))

	# Plot example signal
	if plot and debug and not online:
		fig_inp = pl.figure()
		ax1 = fig_inp.add_subplot(211)
		ax2 = fig_inp.add_subplot(212)
		fig_inp.suptitle('Input Stimulus / Signal')
		inp_plot = vis.InputPlots(stim_obj=stim, input_obj=inputs.train_set_signal, noise_obj=inputs.train_set_noise)
		inp_plot.plot_stimulus_matrix(set='train', ax=ax1, save=False, display=False)
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
	enc_layer = EncodingLayer(parameter_set.encoding_pars, signal=inputs.train_set_signal, online=online)
	enc_layer.connect(parameter_set.encoding_pars, net)

	# Attach decoders to input encoding populations
	if not empty(enc_layer.encoders) and hasattr(parameter_set.encoding_pars, "input_decoder"):
		enc_layer.connect_decoders(parameter_set.encoding_pars.input_decoder)

	if plot and debug:
		vis.extract_encoder_connectivity(enc_layer, net, display, save=paths['figures'] + paths['label'])

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
		                           ax=[ax1, ax3, ax2, ax4], display=display, save=paths['figures'] + paths['label'])

	######################################################################################
	# Simulate (Initial Transient)
	# ====================================================================================
	if stim.transient_set_labels:
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

	######################################################################################
	# Simulate (Unique Sequence)
	# ====================================================================================
	if not online:
		print "\nUnique Sequence time = {0} ms".format(str(inputs.unique_stimulation_time))
	iterate_input_sequence(net, inputs.unique_set_signal, enc_layer,
	                       sampling_times=parameter_set.decoding_pars.global_sampling_times,
	                       stim_set=stim, input_set=inputs, set_name='unique', store_responses=False)
	results['rank'] = get_state_rank(net)
	n_stim = len(stim.elements)
	print "State Rank: {0} / {1}".format(str(results['rank']), str(n_stim))
	for n_pop in list(itertools.chain(*[net.populations, net.merged_populations])):
		if not empty(n_pop.state_matrix):
			n_pop.flush_states()
	for n_enc in enc_layer.encoders:
		if not empty(n_enc.state_matrix):
			n_enc.flush_states()

	#######################################################################################
	# Simulate (Train period)
	# =====================================================================================
	if not online:
		print "\nTrain time = {0} ms".format(str(inputs.train_stimulation_time))
	iterate_input_sequence(net, inputs.train_set_signal, enc_layer,
	                       sampling_times=parameter_set.decoding_pars.global_sampling_times,
	                       stim_set=stim, input_set=inputs, set_name='train', store_responses=False)

	#######################################################################################
	# Train Readouts
	# =====================================================================================
	train_all_readouts(parameter_set, net, stim, inputs.train_set_signal, encoding_layer=enc_layer, flush=True,
	                   debug=debug,plot=plot, display=display, save=paths)

	#######################################################################################
	# Simulate (Test period)
	# =====================================================================================
	if not online:
		print "\nTest time = {0} ms".format(str(inputs.test_stimulation_time))
	iterate_input_sequence(net, inputs.test_set_signal, enc_layer,
	                       sampling_times=parameter_set.decoding_pars.global_sampling_times,
	                       stim_set=stim, input_set=inputs, set_name='test', store_responses=paths['activity'])
	#######################################################################################
	# Test Readouts
	# =====================================================================================
	test_all_readouts(parameter_set, net, stim, inputs.test_set_signal, encoding_layer=enc_layer, flush=False,
	                  debug=debug,
	                  plot=plot, display=display, save=paths)

	results['Performance'] = {}
	results['Performance'].update(analyse_performance_results(net, enc_layer, plot=plot, display=display, save=paths[
		                                                                                                           'figures'] +
	                                                                                                           paths[
		                                                                                                           'label']))

	# #######################################################################################
	# # Save data
	# # =====================================================================================
	if save:
		with open(paths['results'] + 'Results_' + parameter_set.label, 'w') as f:
			pickle.dump(results, f)
		parameter_set.save(paths['parameters'] + 'Parameters_' + parameter_set.label)
	return results


def discrete_input_nStep(parameter_set, plot=False, display=False, save=False, debug=False, online=True):
	"""
	Run the RC sequence processing task
	:param parameter_set:
	:param plot:
	:param display:
	:param save:
	:param debug:
	:return:
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

	###################################################################################
	# Randomize initial variable values
	# =================================================================================
	for n in list(iterate_obj_list(net.populations)):
		n.randomize_initial_states('V_m', randomization_function=np.random.uniform, low=-70., high=-55.)
	# n.randomize_initial_states('V_th', randomization_function=np.random.normal, loc=-50., scale=1.)

	###################################################################################
	# Build and connect input
	# =================================================================================
	# Create StimulusSet
	stim_set_time = time.time()
	stim = StimulusSet(parameter_set, unique_set=False)
	stim.create_set(parameter_set.stim_pars.full_set_length)
	stim.discard_from_set(parameter_set.stim_pars.transient_set_length)
	stim.divide_set(parameter_set.stim_pars.transient_set_length, parameter_set.stim_pars.train_set_length,
	                parameter_set.stim_pars.test_set_length)
	print "- Elapsed Time: {0}".format(str(time.time() - stim_set_time))

	# Create InputSignalSet
	input_set_time = time.time()
	inputs = InputSignalSet(parameter_set, stim, online=online)
	if stim.transient_set_labels:
		inputs.generate_transient_set(stim)
		parameter_set.kernel_pars.transient_t = inputs.transient_stimulation_time
	if not online:
		inputs.generate_full_set(stim)
	#inputs.generate_unique_set(stim)
	inputs.generate_train_set(stim)
	inputs.generate_test_set(stim)
	print "- Elapsed Time: {0}".format(str(time.time() - input_set_time))

	# Plot example signal
	if plot and debug and not online:
		fig_inp = pl.figure()
		ax1 = fig_inp.add_subplot(211)
		ax2 = fig_inp.add_subplot(212)
		fig_inp.suptitle('Input Stimulus / Signal')
		inp_plot = vis.InputPlots(stim_obj=stim, input_obj=inputs.train_set_signal, noise_obj=inputs.train_set_noise)
		inp_plot.plot_stimulus_matrix(set='train', ax=ax1, save=False, display=False)
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
		vis.extract_encoder_connectivity(enc_layer, net, display, save=paths['figures'] + paths['label'])

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
		                           ax=[ax1, ax3, ax2, ax4], display=display, save=paths['figures'] + paths['label'])

	######################################################################################
	# Simulate (Initial Transient)
	# ====================================================================================
	if stim.transient_set_labels:
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
		print "\nTrain time = {0} ms".format(str(inputs.train_stimulation_time))
	iterate_input_sequence(net, inputs.train_set_signal, enc_layer,
	                       sampling_times=parameter_set.decoding_pars.global_sampling_times,
	                       stim_set=stim, input_set=inputs, set_name='train', store_responses=False)

	#######################################################################################
	# Train Readouts
	# =====================================================================================
	train_all_readouts(parameter_set, net, stim, inputs.train_set_signal, encoding_layer=enc_layer, flush=True,
	                   debug=debug, plot=plot, display=display, save=paths)

	#######################################################################################
	# Simulate (Test period)
	# =====================================================================================
	if not online:
		print "\nTest time = {0} ms".format(str(inputs.test_stimulation_time))
	iterate_input_sequence(net, inputs.test_set_signal, enc_layer,
	                       sampling_times=parameter_set.decoding_pars.global_sampling_times,
	                       stim_set=stim, input_set=inputs, set_name='test',
	                       store_responses=False)

	#######################################################################################
	# Test Readouts
	# =====================================================================================
	test_all_readouts(parameter_set, net, stim, inputs.test_set_signal, encoding_layer=enc_layer, flush=False,
	                  debug=debug, plot=plot, display=display, save=paths)

	results.update(analyse_performance_results(net, enc_layer, plot=plot, display=display, save=paths['figures'] +
	                  paths['label']))

	#######################################################################################
	# Save data
	# =====================================================================================
	if save:
		with open(paths['results'] + 'Results_' + parameter_set.label, 'w') as f:
			pickle.dump(results, f)
		parameter_set.save(paths['parameters'] + 'Parameters_' + parameter_set.label)


def IPS_task(parameter_set, plot=False, display=False, save=False, debug=False, online=True, dataset_path=''):
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


def single_perturbation_analysis(parameter_set, plot=False, display=False, save=False, debug=False):
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
	results = dict()
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


def run_task(parameter_set, plot=True, display=True, save=False, online=True, debug=False):
	"""
	:param parameter_set:
	:param plot:
	:param display:
	:param save:
	:param online:
	:param debug:
	:return:
	"""
	from Projects.EncodingDecoding.StimulusGenerator.PatternGenerator import StimulusPattern
	from Projects.EncodingDecoding.Specific_extra_scripts.auxiliary_functions import train_readouts, test_readouts

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
	start_build = time.time()
	net = Network(parameter_set.net_pars)
	net.merge_subpopulations([net.populations[0], net.populations[1]], name='EI')

	########################################################################################################################
	# Randomize initial variable values
	# =======================================================================================================================
	for idx, n in enumerate(list(iterate_obj_list(net.populations))):
		if hasattr(parameter_set.net_pars, "randomize_neuron_pars"):
			randomize = parameter_set.net_pars.randomize_neuron_pars[idx]
			for k, v in randomize.items():
				n.randomize_initial_states(k, randomization_function=v[0], **v[1])

	###################################################################################
	# Build Stimulus Set
	# =================================================================================
	stim_set_startbuild = time.time()

	# Create or Load StimulusPattern
	stim_pattern = StimulusPattern(parameter_set.task_pars)
	stim_pattern.generate()

	input_sequence, output_sequence = stim_pattern.as_index()

	# Convert to StimulusSet object
	stim_set = StimulusSet(unique_set=None)
	stim_set.load_data(input_sequence, type='full_set_labels')
	stim_set.discard_from_set(parameter_set.stim_pars.transient_set_length)
	stim_set.divide_set(parameter_set.stim_pars.transient_set_length, parameter_set.stim_pars.train_set_length,
	                    parameter_set.stim_pars.test_set_length)

	# Specify target and convert to StimulusSet object
	target_set = StimulusSet(unique_set=None)
	target_set.load_data(output_sequence, type='full_set_labels')
	target_set.discard_from_set(parameter_set.stim_pars.transient_set_length)
	target_set.divide_set(parameter_set.stim_pars.transient_set_length, parameter_set.stim_pars.train_set_length,
	                      parameter_set.stim_pars.test_set_length)

	print "- Elapsed Time: {0}".format(str(time.time() - stim_set_startbuild))
	stim_set_buildtime = time.time() - stim_set_startbuild
	###################################################################################
	# Build Input Signal Set
	# =================================================================================
	input_set_time = time.time()
	parameter_set.input_pars.signal.N = len(np.unique(input_sequence))
	# Create InputSignalSet
	inputs = InputSignalSet(parameter_set, stim_set, online=online)
	if not empty(stim_set.transient_set_labels):
		inputs.generate_transient_set(stim_set)
		parameter_set.kernel_pars.transient_t = inputs.transient_stimulation_time
	if not online:
		inputs.generate_full_set(stim_set)
	# inputs.generate_unique_set(stim)
	inputs.generate_train_set(stim_set)
	inputs.generate_test_set(stim_set)

	# Plot example signal
	if plot and debug and not online:
		fig_inp = pl.figure()
		ax1 = fig_inp.add_subplot(211)
		ax2 = fig_inp.add_subplot(212)
		fig_inp.suptitle('Input Stimulus / Signal')
		inp_plot = vis.InputPlots(stim_obj=stim_set, input_obj=inputs.test_set_signal, noise_obj=inputs.test_set_noise)
		inp_plot.plot_stimulus_matrix(set='test', ax=ax1, save=False, display=False)
		inp_plot.plot_input_signal(ax=ax2, save=paths['figures'] + paths['label'], display=display)
		inp_plot.plot_input_signal(save=paths['figures'] + paths['label'], display=display)
		inp_plot.plot_signal_and_noise(save=paths['figures'] + paths['label'], display=display)
	parameter_set.kernel_pars.sim_time = inputs.train_stimulation_time + inputs.test_stimulation_time

	if save:
		stim_pattern.save(paths['inputs'])
		stim_set.save(paths['inputs'])
		if debug:
			inputs.save(paths['inputs'])

	print "- Elapsed Time: {0}".format(str(time.time() - input_set_time))
	inputs_build_time = time.time() - input_set_time
	#######################################################################################
	# Encode Input
	# =====================================================================================
	encoder_start_time = time.time()
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
		vis.extract_encoder_connectivity(enc_layer, net, display, save=paths['figures'] + paths['label'])

	#######################################################################################
	# Set-up Analysis
	# =====================================================================================
	net.connect_devices()
	net.connect_decoders(parameter_set.decoding_pars)
	encoding_time = time.time() - encoder_start_time

	######################################################################################
	# Connect Network
	# ====================================================================================
	net.connect_populations(parameter_set.connection_pars)
	build_time = time.time() - start_build
	if plot and debug:
		fig_W = pl.figure()
		topology = vis.TopologyPlots(parameter_set.connection_pars, net)
		topology.print_network(depth=3)
		ax1 = pl.subplot2grid((6, 6), (0, 0), rowspan=4, colspan=4)
		ax2 = pl.subplot2grid((6, 6), (5, 0), rowspan=1, colspan=4)
		ax3 = pl.subplot2grid((6, 6), (0, 5), rowspan=4, colspan=1)
		ax4 = pl.subplot2grid((6, 6), (5, 5), rowspan=1, colspan=1)
		topology.plot_connectivity(parameter_set.connection_pars.synapse_types,
		                           ax=[ax1, ax3, ax2, ax4], display=display, save=paths['figures'] + paths['label'])

	######################################################################################
	# Simulate (Initial Transient)
	# ====================================================================================
	if not empty(stim_set.transient_set_labels):
		if not online:
			print "\nTransient time = {0} ms".format(str(parameter_set.kernel_pars.transient_t))

		iterate_input_sequence(net, inputs.transient_set_signal, enc_layer, sampling_times=None, stim_set=stim_set,
		                       input_set=inputs, set_name='transient', store_responses=False, record=False)
		parameter_set.kernel_pars.transient_t = nest.GetKernelStatus()['time']
		inputs.transient_stimulation_time = nest.GetKernelStatus()['time']
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
			results.update(evaluate_encoding(enc_layer, parameter_set, analysis_interval,
			                                 inputs.transient_set_signal, plot=plot, display=display,
			                                 save=paths['figures'] + paths['label']))

		net.flush_records()
		enc_layer.flush_records()

	#######################################################################################
	# Simulate (Train period)
	# =====================================================================================
	train_start = time.time()
	if not online:
		print "\nFull time = {0} ms".format(str(inputs.full_stimulation_time))
	iterate_input_sequence(net, inputs.train_set_signal, enc_layer,
	                       sampling_times=parameter_set.decoding_pars.global_sampling_times,
	                       stim_set=stim_set, input_set=inputs, set_name='train', store_responses=False)

	if online:
		inputs.train_stimulation_time = nest.GetKernelStatus()['time'] - inputs.transient_stimulation_time

	#######################################################################################
	# Train Readouts
	# =====================================================================================
	train_readouts(parameter_set, net, stim_pattern, stim_set, target_set, encoding_layer=enc_layer, flush=True,
	               debug=debug, plot=plot, display=display, save=paths)
	train_time = time.time() - train_start

	#######################################################################################
	# Simulate (Test period)
	# =====================================================================================
	test_start = time.time()
	if not online:
		print "\nTest time = {0} ms".format(str(inputs.test_stimulation_time))
	iterate_input_sequence(net, inputs.test_set_signal, enc_layer,
	                       sampling_times=parameter_set.decoding_pars.global_sampling_times,
	                       stim_set=stim_set, input_set=inputs, set_name='test', store_responses=False)
	if online:
		inputs.test_stimulation_time = nest.GetKernelStatus()['time'] - (inputs.transient_stimulation_time +
		                                                                 inputs.train_stimulation_time)

	#######################################################################################
	# Test Readouts
	# =====================================================================================
	test_readouts(parameter_set, net, stim_pattern, stim_set, target_set, encoding_layer=enc_layer, flush=True,
	              debug=debug, plot=plot, display=display, save=paths)
	results['performance'] = analyse_performance_results(net, enc_layer, plot=plot, display=display, save=paths[
		                                                                                                      'figures'] +
	                                                                                                      paths[
		                                                                                                      'label'])
	test_time = time.time() - test_start

	#######################################################################################
	# Save Performance data
	# =====================================================================================
	if save:
		with open(paths['results'] + 'Results_' + parameter_set.label, 'w') as f:
			pickle.dump(results, f)
		parameter_set.save(paths['parameters'] + 'Parameters_' + parameter_set.label)

	#######################################################################################
	# Analyse population responses (during test phase)
	# =====================================================================================
	analysis_time = time.time()
	analysis_interval = [inputs.transient_stimulation_time + inputs.train_stimulation_time,
	                     inputs.transient_stimulation_time + inputs.train_stimulation_time
	                     + inputs.test_stimulation_time]
	results['activity_simple'] = population_state(net, parameter_set, nPairs=500, time_bin=1.,
	                                              start=analysis_interval[0],
	                                              stop=analysis_interval[1], plot=plot, display=display,
	                                              save=paths['figures'] + paths['label'])
	#######################################################################################
	# Save simple activity analysis data
	# =====================================================================================
	if save:
		with open(paths['results'] + 'Results_' + parameter_set.label, 'w') as f:
			pickle.dump(results, f)

	results['activity'] = characterize_population_activity(net, parameter_set, analysis_interval, epochs=None,
	                                                       time_bin=1., summary_only=True, time_resolved=False,
	                                                       window_len=100, color_map='Accent', plot=False,
	                                                       display=display, save=paths['figures'] + paths['label'])
	analysis_time = time.time() - analysis_time

	########################################################################################
	# Store time information
	# =====================================================================================
	results['time'] = {'build_time': build_time, 'stim_build': stim_set_buildtime,
	                   'inputs_build': inputs_build_time, 'train': train_time,
	                   'test': test_time, 'analysis': analysis_time}

	#######################################################################################
	# Save all data
	# =====================================================================================
	if save:
		with open(paths['results'] + 'Results_' + parameter_set.label, 'w') as f:
			pickle.dump(results, f)
		parameter_set.save(paths['parameters'] + 'Parameters_' + parameter_set.label)


def Glu_synapse_at_rest(parameter_set, plot=False, save=True):
	"""

	:return:
	"""
	from Projects.TimeScales0.Extra.analysis_functions import glutamate_synapse_analysis
	if not isinstance(parameter_set, ParameterSet):
		if isinstance(parameter_set, basestring) or isinstance(parameter_set, dict):
			parameter_set = ParameterSet(parameter_set)
		else:
			raise TypeError(
				"parameter_set must be ParameterSet, string with full path to parameter file or dictionary")
	parameter_set.randomized_pars[parameter_set.additional.neuron_type] = delete_keys_from_dict(
		parameter_set.randomized_pars[parameter_set.additional.neuron_type], ['label'])
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
	nest.SetKernelStatus({'resolution': 0.1, 'print_time': False})
	spike_times = np.arange(100., parameter_set.additional.totalT, 500.)  # np.array([20., 100., 250., 500., 800.,
	# 1200., 1500.])
	spk_gen = nest.Create('spike_generator', 1, params={'spike_times': spike_times})

	neurons = nest.Create('iaf_cond_mtime', parameter_set.additional.n_neurons, extract_nestvalid_dict(
		parameter_set.neuron))
	if parameter_set.additional.randomize:
		for k, v in parameter_set.randomized_pars[parameter_set.additional.neuron_type].items():
			randomization_function = v[0]
			try:
				nest.SetStatus(neurons, k, randomization_function(size=len(neurons), **v[1]))
			except:
				print(k)
				for n_neuron in neurons:
					success = False
					while not success:
						try:
							nest.SetStatus([n_neuron], k, randomization_function(size=1, **v[1]))
							success = True
						except:
							print(n_neuron)
							pass
	nest.Connect(spk_gen, neurons, syn_spec={'weight': parameter_set.additional.w, 'delay': parameter_set.additional.d, 'receptor_type': 1,
	                                         "model": 'static_synapse'})
	nest.Connect(spk_gen, neurons, syn_spec={'weight': parameter_set.additional.w, 'delay': parameter_set.additional.d, 'receptor_type': 2,
	                                         "model": 'static_synapse'})
	mm = nest.Create('multimeter')
	nest.SetStatus(mm, {'interval': .1, 'record_from': ['V_m', 'C1', 'C2', 'I_ex', 'I_in']})
	nest.Connect(mm, neurons)

	spkdet = nest.Create('spike_detector')
	nest.Connect(neurons, spkdet)

	nest.Simulate(parameter_set.additional.totalT + 0.1)

	events = nest.GetStatus(mm)[0]['events']
	t = events['times']

	time_window = [-10., 200.]

	results_pre = glutamate_synapse_analysis(events, spike_times, time_window, plot=plot)
	results = {'q_ratio': results_pre['q_ratio'], 'psc_ratio': results_pre['psc_ratio']}
	#######################################################################################
	# Save data
	# =====================================================================================
	if save:
		with open(paths['results'] + 'Results_' + parameter_set.label, 'w') as f:
			pickle.dump(results, f)
		parameter_set.save(paths['parameters'] + 'Parameters_' + parameter_set.label)

	return results

########################################################################################################################
# Auxiliary Functions
########################################################################################################################
def iterate_input_sequence(network_obj, input_signal, enc_layer, sampling_times=None, stim_set=None,
                           input_set=None, set_name=None, record=True,
                           store_responses=False, jitter=None, average=False):
	"""
	Run simulation iteratively, presenting one input stimulus at a time..
	:param network_obj: Network object
	:param input_signal: InputSignal object
	:param enc_layer: EncodingLayer
	:param sampling_times:
	:param stim_set: full StimulusSet object
	:param input_set: full InputSignalSet object
	:param set_name: string with the name of the current set ('transient', 'unique', 'train', 'test')
	:param record: [bool] - acquire state matrix (according to sampling_times)
	:param store_responses: [bool] - record entire population activity (memory!)
	:param jitter: if input is spike pattern, add jitter
	:param average: [bool] - if True, state vector is average of subsampled activity vectors
	"""
	if not (isinstance(network_obj, Network)):
		raise TypeError("Please provide a Network object")
	if not isinstance(enc_layer, EncodingLayer):
		raise TypeError("Please provide an EncodingLayer object")
	if input_set is not None and not isinstance(input_set, InputSignalSet):
		raise TypeError("input_set must be InputSignalSet")

	sampling_lag = 2.
	if sampling_times is None and not input_set.online:
		t_samp = np.sort(list(iterate_obj_list(input_signal.offset_times)))
	elif sampling_times is None and input_set.online:
		t_samp = [0]
	elif sampling_times is not None and input_set.online:
		t_samp = sampling_times
		t_step = [0]
	else:
		t_samp = sampling_times

	if not input_set.online:
		intervals = input_signal.intervals
		set_size = len(list(iterate_obj_list(input_signal.amplitudes)))
		set_names = ['train', 'test', 'transient', 'unique', 'full']
		set_sizes = {k: 0 for k in set_names}
		for k, v in stim_set.__dict__.items():
			for set_name in set_names:
				if k == '{0}_set_labels'.format(set_name):
					set_sizes['{0}'.format(set_name)] = len(v)
		current_set = [key for key, value in set_sizes.items() if set_size == value][0]
	else:
		assert(set_name is not None), "set_name needs to be provided in online mode.."
		current_set = set_name
		set_size = len(getattr(stim_set, '{0}_set_labels'.format(current_set)))
		signal_iterator = getattr(input_set, '{0}_set_signal_iterator'.format(current_set))
		stimulus_seq = getattr(stim_set, '{0}_set'.format(current_set))
		intervals = [0] #?

	if store_responses:
		print "\n\n!!! All responses will be stored !!!"
		labels = np.unique(getattr(stim_set, '{0}_set_labels'.format(current_set)))
		set_labels = getattr(stim_set, '{0}_set_labels'.format(current_set))
		epochs = {k: [] for k in labels}

	####################################################################################################################
	if sampling_times is None:  # one sample for each stimulus
		if input_set.online:
			print "\nSimulating {0} steps".format(str(set_size))
		else:
			print "\nSimulating {0} ms in {1} steps".format(str(
				input_signal.input_signal.t_stop-input_signal.input_signal.t_start), str(set_size))

		# ################################ Main Loop ###################################
		for idx, t in enumerate(t_samp):

			t_int = nest.GetKernelStatus()['time']

			if input_set.online and idx < set_size:
				local_signal = signal_iterator.next()
				local_signal.time_offset(t_int)
				intervals.append(local_signal.intervals)
				t_samp.append(list(itertools.chain(*local_signal.offset_times))[0])
				t = t_samp[-1]
				if intervals[-1]:
					t += intervals[-1]
				t_sim = t - t_int
				if store_responses:
					epochs[set_labels[idx]].append((t_int, t_samp[-1]))
					print idx, set_labels[idx], epochs[set_labels[idx]][-1]
			else:
				local_signal = None
				if idx < len(t_samp) - 1:
					if intervals[idx]:
						t += intervals[idx]
					if store_responses:
						epochs[set_labels[idx]].append((t_int, t_samp[idx]))
						print idx, set_labels[idx], epochs[set_labels[idx]][-1]
				t_sim = t - t_int

			if store_responses and input_set.online and idx < set_size:
				epochs[set_labels[idx]].append((t_int, t_samp[-1]))
				print idx, set_labels[idx], epochs[set_labels[idx]][-1]
				print (t_int, t_samp[-1])
			elif store_responses and input_set.online:
				epochs[set_labels[idx]].append((t_int, t))
				print (t_int, t)
			elif store_responses:
					epochs[set_labels[idx]].append((t_int, t-intervals[idx]))
					print epochs[set_labels[idx]][-1]
			if len(t_samp) <= set_size + 1 and t_sim > 0.:
				print "\nSimulating step {0} [{1} ms]".format(str(idx + 1), str(t_sim))

				# Spike templates (need to be updated online)
				if np.mean(['spike_pattern' in n for n in list(iterate_obj_list(enc_layer.generator_names))]) == 1. and \
						input_set is not None:
					assert(len(input_set.spike_patterns) == stim_set.dims), "Incorrect number of spike patterns"
					if input_set.online and local_signal is not None:
						stimulus_id = [nx for nx in range(stim_set.dims) if t_samp[-1] in local_signal.offset_times[nx]]
					else:
						stimulus_id = [nx for nx in range(stim_set.dims) if t_samp[idx] in input_signal.offset_times[nx]]
					if t_int == 0.:
						spks = input_set.spike_patterns[stimulus_id[0]].time_offset(t_int + nest.GetKernelStatus()[
							'resolution'], True)
					else:
						spks = input_set.spike_patterns[stimulus_id[0]].time_offset(t_int, True)
					if jitter is not None:
						spks.jitter(jitter)
					enc_layer.update_state(spks)

				elif input_set is not None and local_signal is not None and input_set.online:
					stim_input = coo_matrix(stimulus_seq.todense()[:, idx])
					local_signal.input_signal = local_signal.generate_single_step(stim_input)

					if t_int == 0.:
						dt = nest.GetKernelStatus()['resolution']
						local_signal.input_signal.time_offset(dt)
					enc_layer.update_state(local_signal.input_signal)

				network_obj.simulate(t_sim)
				network_obj.extract_population_activity()
				network_obj.extract_network_activity()

				if record:
					#######################################################################################
					# Extract state
					# =====================================================================================
					# Extract merged responses
					if not empty(network_obj.merged_populations):
						for ctr, n_pop in enumerate(network_obj.merged_populations):
							n_pop.extract_state_vector(time_point=t, lag=sampling_lag, save=True)
							if idx == 0 and not n_pop.name[-1].isdigit():
								n_pop.name += str(ctr)
					# Extract from populations
					if not empty(network_obj.state_extractors):
						for n_pop in network_obj.populations:
							n_pop.extract_state_vector(time_point=t, lag=sampling_lag, save=True)

					if not store_responses:
						network_obj.flush_records(decoders=True)
					else:
						epochs.update({})

					enc_layer.extract_encoder_activity()

					if not empty(enc_layer.encoders) and not empty(enc_layer.state_extractors):
						for ctr, n_enc in enumerate(enc_layer.encoders):
							n_enc.extract_state_vector(time_point=t, lag=sampling_lag, save=True)

						if not store_responses:
							enc_layer.flush_records(decoders=True)
		if record:
			# compile matrices:
			if not empty(network_obj.merged_populations):
				for ctr, n_pop in enumerate(network_obj.merged_populations):
					if not empty(n_pop.state_matrix):
						n_pop.compile_state_matrix()
			if not empty(network_obj.state_extractors):
				for n_pop in network_obj.populations:
					if not empty(n_pop.state_matrix):
						n_pop.compile_state_matrix()
			if not empty(enc_layer.encoders) and not empty(enc_layer.state_extractors):
				for ctr, n_enc in enumerate(enc_layer.encoders):
					n_enc.compile_state_matrix()

	####################################################################################################################
	elif (sampling_times is not None) and (isinstance(t_samp, list) or isinstance(t_samp, np.ndarray)): # multiple
		# sampling times per stimulus (build multiple state matrices)
		if not input_set.online:
			t_step = np.sort(list(iterate_obj_list(input_signal.offset_times)))
		if input_set.online:
			print "\nSimulating {0} steps".format(str(set_size))
		else:
			print "\nSimulating {0} ms in {1} steps".format(str(
				input_signal.input_signal.t_stop-input_signal.input_signal.t_start), str(set_size))

		# initialize state matrices
		if not empty(network_obj.merged_populations):
			for ctr, n_pop in enumerate(network_obj.merged_populations):
				if not empty(n_pop.state_extractors) and len(n_pop.state_extractors) == 1:
					n_pop.state_matrix = [[[] for _ in range(len(t_samp))]]
				elif not empty(n_pop.state_extractors) and len(n_pop.state_extractors) > 1:
					n_pop.state_matrix = [[[] for _ in range(len(t_samp))] for _ in range(len(n_pop.state_extractors))]
				if not empty(n_pop.state_extractors):
					n_pop.state_sample_times = list(sampling_times)
		if not empty(network_obj.state_extractors):
			for n_pop in network_obj.populations:
				if not empty(n_pop.state_extractors) and len(n_pop.state_extractors) == 1:
					n_pop.state_matrix = [[[] for _ in range(len(t_samp))]]
				elif not empty(n_pop.state_extractors) and len(n_pop.state_extractors) > 1:
					n_pop.state_matrix = [[[] for _ in range(len(t_samp))] for _ in range(len(n_pop.state_extractors))]
				if not empty(n_pop.state_extractors):
					n_pop.state_sample_times = list(sampling_times)
		if not empty(enc_layer.encoders) and not empty(enc_layer.state_extractors):
			for ctr, n_enc in enumerate(enc_layer.encoders):
				if not empty(n_enc.state_extractors) and len(n_enc.state_extractors) == 1:
					n_enc.state_matrix = [[[] for _ in range(len(t_samp))]]
				elif not empty(n_enc.state_extractors) and len(n_enc.state_extractors) > 1:
					n_enc.state_matrix = [[[] for _ in range(len(t_samp))] for _ in range(len(n_enc.state_extractors))]
				if not empty(n_enc.state_extractors):
					n_enc.state_sample_times = list(sampling_times)

		# ################################ Main Loop ###################################
		for idx, t in enumerate(t_step):

			t_int = nest.GetKernelStatus()['time']

			if input_set.online and idx < set_size:
				local_signal = signal_iterator.next()
				local_signal.time_offset(t_int)
				intervals.append(local_signal.intervals)
				t_step.append(list(itertools.chain(*local_signal.offset_times))[0])
				t = t_step[-1]
				if intervals[-1]:
					t += intervals[-1]
				t_sim = t - t_int
			else:
				local_signal = None
				if idx < len(t_step) - 1:
					if intervals[idx]:
						t += intervals[idx]
				t_sim = t - t_int

			if t_sim > 0.: # len(t_step) <= set_size + 1 and
				print "\nSimulating step {0} [{1} ms]".format(str(idx + 1), str(t_sim))

				# Spike templates (need to be updated online)
				if np.mean(['spike_pattern' in n for n in list(iterate_obj_list(enc_layer.generator_names))]) == 1. \
						and \
								input_set is not None:
					assert (len(input_set.spike_patterns) == stim_set.dims), "Incorrect number of spike patterns"
					if input_set.online and local_signal is not None:
						stimulus_id = [nx for nx in range(stim_set.dims) if t_step[-1] in local_signal.offset_times[
							nx]]
					else:
						stimulus_id = [nx for nx in range(stim_set.dims) if
						               t_step[idx] in input_signal.offset_times[nx]]
					if t_int == 0.:
						spks = input_set.spike_patterns[stimulus_id[0]].time_offset(t_int + nest.GetKernelStatus()[
							'resolution'], True)
					else:
						spks = input_set.spike_patterns[stimulus_id[0]].time_offset(t_int, True)
					if jitter is not None:
						spks.jitter(jitter)
					enc_layer.update_state(spks)

				elif input_set is not None and local_signal is not None and input_set.online:
					stim_input = coo_matrix(stimulus_seq.todense()[:, idx])
					local_signal.input_signal = local_signal.generate_single_step(stim_input)

					if t_int == 0.:
						dt = nest.GetKernelStatus()['resolution']
						local_signal.input_signal.time_offset(dt)
					enc_layer.update_state(local_signal.input_signal)

				network_obj.simulate(t_sim)
				network_obj.extract_population_activity()
				network_obj.extract_network_activity()

				if record:
					#######################################################################################
					# Extract state
					# =====================================================================================
					# Extract merged responses
					if not empty(network_obj.merged_populations):
						for ctr, n_pop in enumerate(network_obj.merged_populations):
							if idx == 0 and not n_pop.name[-1].isdigit():
								n_pop.name += str(ctr)
							if not empty(n_pop.state_extractors):
								print "Collecting state matrices from Population {0}".format(str(n_pop.name))
								for sample_idx, n_sample_time in enumerate(t_samp):
									assert(n_sample_time >= sampling_lag), "Minimum sampling time must be >= sampling lag"
									progress_bar(float(sample_idx+1)/float(len(t_samp)))
									sample_time = t_int + n_sample_time
									state_vectors = n_pop.extract_state_vector(time_point=sample_time, lag=sampling_lag,
									                                                       save=False)
									for state_id, state_vec in enumerate(state_vectors):
										n_pop.state_matrix[state_id][sample_idx].append(state_vec[0])
					# Extract from populations
					if not empty(network_obj.state_extractors):
						for n_pop in network_obj.populations:
							if not empty(n_pop.state_extractors):
								print "Collecting state matrices from Population {0}".format(str(n_pop.name))
								for sample_idx, n_sample_time in enumerate(t_samp):
									assert (n_sample_time >= sampling_lag), "Minimum sampling time must be >= sampling lag"
									progress_bar(float(sample_idx+1) / float(len(t_samp)))
									sample_time = t_int + n_sample_time
									state_vectors = n_pop.extract_state_vector(time_point=sample_time, lag=sampling_lag,
									                                           save=False)
									for state_id, state_vec in enumerate(state_vectors):
										n_pop.state_matrix[state_id][sample_idx].append(state_vec[0])
					if not store_responses:
						network_obj.flush_records(decoders=True)
					enc_layer.extract_encoder_activity()

					# Extract from Encoders
					if not empty(enc_layer.encoders) and not empty(enc_layer.state_extractors):
						for ctr, n_enc in enumerate(enc_layer.encoders):
							if not empty(n_enc.state_extractors):
								print "Collecting state matrices from Encoder {0}".format(str(n_enc.name))
								for sample_idx, n_sample_time in enumerate(t_samp):
									assert (n_sample_time >= sampling_lag), "Minimum sampling time must be >= sampling lag"
									progress_bar(float(sample_idx+1) / float(len(t_samp)))
									sample_time = t_int + n_sample_time
									state_vectors = n_enc.extract_state_vector(time_point=sample_time, lag=sampling_lag,
									                                           save=False)
									for state_id, state_vec in enumerate(state_vectors):
										n_enc.state_matrix[state_id][sample_idx].append(state_vec[0])
						if not store_responses:
							enc_layer.flush_records(decoders=True)
		if record:
			# compile matrices:
			if not empty(network_obj.merged_populations):
				for ctr, n_pop in enumerate(network_obj.merged_populations):
					if not empty(n_pop.state_matrix):
						n_pop.compile_state_matrix(sampling_times=sampling_times)
			if not empty(network_obj.state_extractors):
				for n_pop in network_obj.populations:
					if not empty(n_pop.state_matrix):
						n_pop.compile_state_matrix(sampling_times=sampling_times)
			if not empty(enc_layer.encoders) and not empty(enc_layer.state_extractors):
				for ctr, n_enc in enumerate(enc_layer.encoders):
					n_enc.compile_state_matrix(sampling_times=sampling_times)
	####################################################################################################################
	elif sampling_times is not None and isinstance(t_samp, float) and not average:  # sub-sampled state (and input)
		# multiple sampling times per stimulus (build multiple state matrices)
		if not input_set.online:
			t_step = np.sort(list(iterate_obj_list(input_signal.offset_times)))

		sample_every_n = int(round(t_samp ** (-1))) #* step_size  # take one sample of activity every n steps

		if store_responses:
			assert(isinstance(store_responses, str)), "Please provide a path to store the responses in"
			print "Warning: All response matrices will be stored to file!"

		if input_set.online:
			print "\nSimulating {0} steps".format(str(set_size))
		else:
			print "\nSimulating {0} ms in {1} steps".format(str(
				input_signal.input_signal.t_stop - input_signal.input_signal.t_start), str(len(t_step)))

		# initialize state matrices
		if not empty(network_obj.merged_populations):
			for n_pop in network_obj.merged_populations:
				if not empty(n_pop.state_extractors):
					n_pop.state_matrix = [[] for _ in range(len(n_pop.state_extractors))]
		if not empty(network_obj.state_extractors):
			for n_pop in network_obj.populations:
				if not empty(n_pop.state_extractors):
					n_pop.state_matrix = [[] for _ in range(len(n_pop.state_extractors))]
		if not empty(enc_layer.encoders) and not empty(enc_layer.state_extractors):
			for n_pop in enc_layer.encoders:
				if not empty(n_pop.state_extractors):
					n_pop.state_matrix = [[] for _ in range(len(n_pop.state_extractors))]

		for idx, t in enumerate(t_step):

			t_int = nest.GetKernelStatus()['time']

			if input_set.online and idx < set_size:
				local_signal = signal_iterator.next()
				local_signal.time_offset(t_int)
				intervals.append(local_signal.intervals)
				t_step.append(list(itertools.chain(*local_signal.offset_times))[0])
				t = t_step[-1]
				if intervals[-1]:
					t += intervals[-1]
				t_sim = t - t_int
			else:
				local_signal = None
				if idx < len(t_step) - 1:
					if intervals[idx]:
						t += intervals[idx]
				t_sim = t - t_int
			if t_sim > 0.:
				print "\nSimulating step {0} [{1} ms]".format(str(idx + 1), str(t_sim))
				# Spike templates (need to be updated online)
				if np.mean(['spike_pattern' in n for n in list(iterate_obj_list(enc_layer.generator_names))]) == 1. \
						and \
								input_set is not None:
					assert (len(input_set.spike_patterns) == stim_set.dims), "Incorrect number of spike patterns"
					if input_set.online and local_signal is not None:
						stimulus_id = [nx for nx in range(stim_set.dims) if t_step[-1] in local_signal.offset_times[
							nx]]
					else:
						stimulus_id = [nx for nx in range(stim_set.dims) if
						               t_step[idx] in input_signal.offset_times[nx]]
					if t_int == 0.:
						spks = input_set.spike_patterns[stimulus_id[0]].time_offset(t_int + nest.GetKernelStatus()[
							'resolution'], True)
					else:
						spks = input_set.spike_patterns[stimulus_id[0]].time_offset(t_int, True)
					if jitter is not None:
						spks.jitter(jitter)
					enc_layer.update_state(spks)

				elif input_set is not None and local_signal is not None and input_set.online:
					stim_input = coo_matrix(stimulus_seq.todense()[:, idx])
					local_signal.input_signal = local_signal.generate_single_step(stim_input)

					if t_int == 0.:
						dt = nest.GetKernelStatus()['resolution']
						local_signal.input_signal.time_offset(dt)
					enc_layer.update_state(local_signal.input_signal)

				network_obj.simulate(t_sim)
				network_obj.extract_population_activity()
				network_obj.extract_network_activity()

				if record:
					#######################################################################################
					# Extract state
					# =====================================================================================
					# Extract merged responses
					if not empty(network_obj.merged_populations):
						for ctr, n_pop in enumerate(network_obj.merged_populations):
							if idx == 0 and not n_pop.name[-1].isdigit():
								n_pop.name += str(ctr)
							if not empty(n_pop.state_extractors):
								print "Collecting response samples from Population {0} [rate = {1}]".format(str(
									n_pop.name), str(t_samp))
								responses = n_pop.extract_response_matrix(start=t_int, stop=t, save=False)
								for response_idx, n_response in enumerate(responses):
									if store_responses:
										np.save(store_responses + n_pop.name + '-StateVar{0}-Response{1}.npy'.format(str(
											response_idx), str(idx)), n_response)
									if idx == 0:
										n_pop.state_matrix[response_idx] = n_response.as_array()[:, ::sample_every_n]
										print n_pop.state_matrix[response_idx].shape
									else:
										n_pop.state_matrix[response_idx] = np.append(n_pop.state_matrix[response_idx],
										                                   n_response.as_array()[:, ::sample_every_n], axis=1)
										print n_pop.state_matrix[response_idx].shape

					# Extract from populations
					if not empty(network_obj.state_extractors):
						for n_pop in network_obj.populations:
							if not empty(n_pop.state_extractors):
								print "Collecting response samples from Population {0} [rate = {1}]".format(str(
									n_pop.name), str(t_samp))
								responses = n_pop.extract_response_matrix(start=t_int, stop=t, save=False)
								for response_idx, n_response in enumerate(responses):
									if store_responses:
										np.save(store_responses + n_pop.name + '-StateVar{0}-Response{1}.npy'.format(str(
											response_idx), str(idx)), n_response)
									if idx == 0:
										n_pop.state_matrix[response_idx] = n_response.as_array()[:, ::sample_every_n]
										#print n_pop.state_matrix[response_idx].shape
									else:
										n_pop.state_matrix[response_idx] = np.append(n_pop.state_matrix[response_idx], n_response.as_array(
										)[:, ::sample_every_n], axis=1)
										#print n_pop.state_matrix[response_idx].shape
					if not store_responses:
						network_obj.flush_records(decoders=True)
					enc_layer.extract_encoder_activity()

					# Extract from Encoders
					if not empty(enc_layer.encoders) and not empty(enc_layer.state_extractors):
						for ctr, n_enc in enumerate(enc_layer.encoders):
							if not empty(n_enc.state_extractors):
								print "Collecting response samples from Encoder {0} [rate = {1}]".format(str(
									n_enc.name), str(t_samp))
								responses = n_enc.extract_response_matrix(start=t_int, stop=t, save=False)
								for response_idx, n_response in enumerate(responses):
									if store_responses:
										np.save(store_responses + n_enc.name + '-StateVar{0}-Response{1}.npy'.format(str(
											response_idx), str(idx)), n_response)
									if idx == 0:
										n_enc.state_matrix[response_idx] = n_response.as_array()[:, ::sample_every_n]
									else:
										n_enc.state_matrix[response_idx] = np.append(n_enc.state_matrix[response_idx], n_response.as_array(
										)[:, ::sample_every_n], axis=1)
						if not store_responses:
							enc_layer.flush_records(decoders=True)
	####################################################################################################################
	elif sampling_times is not None and isinstance(t_samp, float) and average:  # sub-sampled state (and input)
		# multiple sampling times per stimulus (build multiple state matrices), state vector = average over stimulus
		# presentation time
		if not input_set.online:
			t_step = np.sort(list(iterate_obj_list(input_signal.offset_times)))

		sample_every_n = int(round(t_samp ** (-1)))  # * step_size  # take one sample of activity every n steps

		if store_responses:
			assert (isinstance(store_responses, str)), "Please provide a path to store the responses in"
			print "Warning: All response matrices will be stored to file!"

		if input_set.online:
			print "\nSimulating {0} steps".format(str(set_size))
		else:
			print "\nSimulating {0} ms in {1} steps".format(str(
				input_signal.input_signal.t_stop - input_signal.input_signal.t_start), str(len(t_step)))

		# initialize state matrices
		if not empty(network_obj.merged_populations):
			for n_pop in network_obj.merged_populations:
				if not empty(n_pop.state_extractors):
					n_pop.state_matrix = [[] for _ in range(len(n_pop.state_extractors))]
		if not empty(network_obj.state_extractors):
			for n_pop in network_obj.populations:
				if not empty(n_pop.state_extractors):
					n_pop.state_matrix = [[] for _ in range(len(n_pop.state_extractors))]
		if not empty(enc_layer.encoders) and not empty(enc_layer.state_extractors):
			for n_pop in enc_layer.encoders:
				if not empty(n_pop.state_extractors):
					n_pop.state_matrix = [[] for _ in range(len(n_pop.state_extractors))]

		inter_stim_int = 0

		for idx, t in enumerate(t_step):

			t_int = nest.GetKernelStatus()['time']

			if input_set.online and idx < set_size:
				local_signal = signal_iterator.next()
				local_signal.time_offset(t_int)
				intervals.append(local_signal.intervals)
				t_step.append(list(itertools.chain(*local_signal.offset_times))[0])
				t = t_step[-1]
				if intervals[-1]:
					t += intervals[-1]
					inter_stim_int = intervals[-1]
				t_sim = t - t_int
			else:
				local_signal = None
				if idx < len(t_step) - 1:
					if intervals[idx]:
						t += intervals[idx]
						inter_stim_int = intervals[idx]
				t_sim = t - t_int
			if t_sim > 0.:
				print "\nSimulating step {0} [{1} ms]".format(str(idx + 1), str(t_sim))
				# Spike templates (need to be updated online)
				if np.mean(['spike_pattern' in n for n in list(iterate_obj_list(enc_layer.generator_names))]) == 1. \
						and \
								input_set is not None:
					assert (len(input_set.spike_patterns) == stim_set.dims), "Incorrect number of spike patterns"
					if input_set.online and local_signal is not None:
						stimulus_id = [nx for nx in range(stim_set.dims) if t_step[-1] in local_signal.offset_times[
							nx]]
					else:
						stimulus_id = [nx for nx in range(stim_set.dims) if
						               t_step[idx] in input_signal.offset_times[nx]]
					if t_int == 0.:
						spks = input_set.spike_patterns[stimulus_id[0]].time_offset(t_int + nest.GetKernelStatus()[
							'resolution'], True)
					else:
						spks = input_set.spike_patterns[stimulus_id[0]].time_offset(t_int, True)
					if jitter is not None:
						spks.jitter(jitter)
					enc_layer.update_state(spks)

				elif input_set is not None and local_signal is not None and input_set.online:
					stim_input = coo_matrix(stimulus_seq.todense()[:, idx])
					local_signal.input_signal = local_signal.generate_single_step(stim_input)

					if t_int == 0.:
						dt = nest.GetKernelStatus()['resolution']
						local_signal.input_signal.time_offset(dt)
					enc_layer.update_state(local_signal.input_signal)

				network_obj.simulate(t_sim)
				network_obj.extract_population_activity()
				network_obj.extract_network_activity()

				if record:
					#######################################################################################
					# Extract state
					# =====================================================================================
					# Extract merged responses
					if not empty(network_obj.merged_populations):
						for ctr, n_pop in enumerate(network_obj.merged_populations):
							if idx == 0 and not n_pop.name[-1].isdigit():
								n_pop.name += str(ctr)
							if not empty(n_pop.state_extractors):
								print "Collecting response samples from Population {0} [rate = {1}]".format(str(
									n_pop.name), str(t_samp))
								responses = n_pop.extract_response_matrix(start=t_int, stop=t-inter_stim_int, save=False)
								for response_idx, n_response in enumerate(responses):
									if store_responses:
										np.save(
											store_responses + n_pop.name + '-StateVar{0}-Response{1}.npy'.format(str(
												response_idx), str(idx)), n_response)
									if idx == 0:
										subsampled_states = n_response.as_array()[:, ::sample_every_n]
										n_pop.state_matrix[response_idx] = np.array([np.mean(subsampled_states, 1)]).T
										#print n_pop.state_matrix[response_idx].shape
									else:
										subsampled_states = n_response.as_array()[:, ::sample_every_n]
										state_vec = np.array([np.mean(subsampled_states, 1)]).T
										n_pop.state_matrix[response_idx] = np.append(n_pop.state_matrix[response_idx],
										                                             state_vec,
										                                             axis=1)
										#print n_pop.state_matrix[response_idx].shape

					# Extract from populations
					if not empty(network_obj.state_extractors):
						for n_pop in network_obj.populations:
							if not empty(n_pop.state_extractors):
								print "Collecting response samples from Population {0} [rate = {1}]".format(str(
									n_pop.name), str(t_samp))
								responses = n_pop.extract_response_matrix(start=t_int, stop=t-inter_stim_int, save=False)
								for response_idx, n_response in enumerate(responses):
									if store_responses:
										np.save(
											store_responses + n_pop.name + '-StateVar{0}-Response{1}.npy'.format(str(
												response_idx), str(idx)), n_response)
									if idx == 0:
										subsampled_states = n_response.as_array()[:, ::sample_every_n]
										n_pop.state_matrix[response_idx] = np.array([np.mean(subsampled_states, 1)]).T
										#print n_pop.state_matrix[response_idx].shape
									else:
										subsampled_states = n_response.as_array()[:, ::sample_every_n]
										state_vec = np.array([np.mean(subsampled_states, 1)]).T
										n_pop.state_matrix[response_idx] = np.append(n_pop.state_matrix[response_idx],
										                                             state_vec, axis=1)
										#print n_pop.state_matrix[response_idx].shape
					if not store_responses:
						network_obj.flush_records(decoders=True)
					enc_layer.extract_encoder_activity()

					# Extract from Encoders
					if not empty(enc_layer.encoders) and not empty(enc_layer.state_extractors):
						for ctr, n_enc in enumerate(enc_layer.encoders):
							if not empty(n_enc.state_extractors):
								print "Collecting response samples from Encoder {0} [rate = {1}]".format(str(
									n_enc.name), str(t_samp))
								responses = n_enc.extract_response_matrix(start=t_int, stop=t-inter_stim_int,
								                                          save=False)
								for response_idx, n_response in enumerate(responses):
									if store_responses:
										np.save(
											store_responses + n_enc.name + '-StateVar{0}-Response{1}.npy'.format(str(
												response_idx), str(idx)), n_response)
									if idx == 0:
										subsampled_states = n_response.as_array()[:, ::sample_every_n]
										n_enc.state_matrix[response_idx] = np.array([np.mean(subsampled_states, 1)]).T
									else:
										subsampled_states = n_response.as_array()[:, ::sample_every_n]
										state_vec = np.array([np.mean(subsampled_states, 1)]).T
										n_enc.state_matrix[response_idx] = np.append(n_enc.state_matrix[response_idx],
										                                             state_vec, axis=1)
						if not store_responses:
							enc_layer.flush_records(decoders=True)
	else:
		raise NotImplementedError("Specify sampling times as None (last step sample), list or array of times, or float")

	if store_responses:
		return epochs