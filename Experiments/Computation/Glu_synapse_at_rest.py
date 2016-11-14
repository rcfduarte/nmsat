from __init__ import *


def run(parameter_set, plot=False, save=True):
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
	nest.SetKernelStatus(extract_nestvalid_dict(parameter_set.kernel_pars.as_dict(), param_type='kernel'))
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