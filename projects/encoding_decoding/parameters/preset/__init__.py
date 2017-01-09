__author__ = 'duarte'
import numpy as np
import sys
from modules.parameters import ParameterSet, copy_dict
from defaults.paths import paths


def set_kernel_defaults(run_type='local', data_label='', **system_pars):
	"""
	Return pre-defined kernel parameters dictionary
	:param default_set:
	:return:
	"""
	if run_type == 'local':
		run = True
	else:
		run = False
	keys = ['nodes', 'ppn', 'mem', 'walltime', 'queue', 'sim_time', 'transient_time']
	if not np.mean(np.sort(system_pars.keys()) == np.sort(keys)).astype(bool):
		raise TypeError("system parameters dictionary must contain the following keys {0}".format(str(keys)))

	N_vp = system_pars['nodes'] * system_pars['ppn']
	np_seed = np.random.randint(1000000000) + 1
	np.random.seed(np_seed)
	msd = np.random.randint(100000000000)

	kernel_pars = {
		'resolution': 0.1,
		'sim_time': system_pars['sim_time'],
		'transient_t': system_pars['transient_time'],
		'data_prefix': data_label,
		'data_path': paths[run_type]['data_path'],
		'mpl_path': paths[run_type]['matplotlib_rc'],
		'overwrite_files': True,
		'print_time': run,
		'rng_seeds': range(msd + N_vp + 1, msd + 2 * N_vp + 1),
		'grng_seed': msd + N_vp,
		'total_num_virtual_procs': N_vp,
		'local_num_threads': system_pars['ppn'],
		'np_seed': np_seed,

		'system': {
			'local': run,
			'system_label': run_type,
			'jdf_template': paths[run_type]['jdf_template'],
			'remote_directory': paths[run_type]['remote_directory'],
			'jdf_fields': {'{{ script_folder }}': '',
			               '{{ nodes }}': str(system_pars['nodes']),
			               '{{ ppn }}': str(system_pars['ppn']),
			               '{{ mem }}': str(system_pars['mem']),
			               '{{ walltime }}': system_pars['walltime'],
			               '{{ queue }}': system_pars['queue'],
			               '{{ computation_script }}': ''}
		}
	}
	return ParameterSet(kernel_pars)


def rec_device_defaults(start=0., stop=sys.float_info.max, resolution=0.1, record_to='memory',
                        device_type='spike_detector', label=''):
	"""
	Standard device parameters
	:param default_set:
	:return:
	"""
	rec_devices = {
		'start': start,
		'stop': stop,
		'origin': 0.,
		'interval': resolution,
		'record_to': [record_to],
		'label': label,
		'model': device_type,
		'close_after_simulate': False,
		'flush_after_simulate': False,
		'flush_records': False,
		'close_on_reset': True,
		'withtime': True,
		'withgid': True,
		'withweight': False,
		'time_in_steps': False,
		'scientific': False,
		'precision': 3,
		'binary': False,
	}
	return ParameterSet(rec_devices)


def set_neuron_defaults(default_set=0):
	"""
	Default single neuron parameters

	:param default_set: (int) - if applicable
	:return: ParameterSet
	"""
	if default_set == 0:
		print("\nLoading Default Neuron Set 7.3 - aeif_cond_exp, fixed voltage threshold, fixed absolute refractory "
		      "time, Fast, conductance-based exponential synapses")
		neuron_pars = {
			'E': {
				'model': 'aeif_cond_exp',
				'C_m': 250.0,
				'Delta_T': 2.0,
				'E_L': -70.,
				'E_ex': 0.0,
				'E_in': -75.0,
				'I_e': 0.,
				'V_m': -70.,
				'V_th': -50.,
				'V_reset': -60.0,
				'V_peak': 0.0,
				'a': 4.0,
				'b': 80.5,
				'g_L': 16.7,
				'g_ex': 1.0,
				'g_in': 1.0,
				't_ref': 2.0,
				'tau_minus': 20.,
				'tau_minus_triplet': 200.,
				'tau_w': 144.0,
				'tau_syn_ex': 2.,
				'tau_syn_in': 6.0,
			},
			'I': {
				'model': 'aeif_cond_exp',
				'C_m': 250.0,
				'Delta_T': 2.0,
				'E_L': -70.,
				'E_ex': 0.0,
				'E_in': -75.0,
				'I_e': 0.,
				'V_m': -70.,
				'V_th': -50.,
				'V_reset': -60.0,
				'V_peak': 0.0,
				'a': 4.0,
				'b': 80.5,
				'g_L': 16.7,
				'g_ex': 1.0,
				'g_in': 1.0,
				't_ref': 2.0,
				'tau_minus': 20.,
				'tau_minus_triplet': 200.,
				'tau_w': 144.0,
				'tau_syn_ex': 2.,
				'tau_syn_in': 6.0,
			},
		}
	else:
		raise NotImplementedError("Default set %s not implemented!" % str(default_set))
	return neuron_pars


def set_network_defaults(neuron_set=0, connection_set=0, N=1250, kernel_pars=None, **synapse_pars):
	"""
	Network default parameters
	:param default_set:
	:return:
	"""
	print "\nLoading Default Network Set - Standard BRN, no topology"
	syn_pars = ParameterSet(synapse_pars)
	nE = 0.8 * N
	nI = 0.2 * N

	rec_devices = rec_device_defaults(start=0.)
	neuron_pars = set_neuron_defaults(default_set=neuron_set)

	#############################################################################################################
	# NETWORK Parameters
	# ===========================================================================================================
	net_pars = {
		'n_populations': 2,
		'pop_names': ['E', 'I'],
		'n_neurons': [int(nE), int(nI)],
		'neuron_pars': [neuron_pars['E'], neuron_pars['I']],
		'randomize_neuron_pars': [{'V_m': (np.random.uniform, {'low': -70., 'high': -50.})},
		                          {'V_m': (np.random.uniform, {'low': -70., 'high': -50.})}],
		'topology': [False, False],
		'topology_dict': [None, None],
		'record_spikes': [True, True],
		'spike_device_pars': [copy_dict(rec_devices,
		                                {'model': 'spike_detector',
		                                 'record_to': ['memory'],
		                                 'interval': 0.1,
		                                 'label': 'E_Spikes'}),
		                      copy_dict(rec_devices,
		                                {'model': 'spike_detector',
		                                 'record_to': ['memory'],
		                                 'interval': 0.1,
		                                 'label': 'I_Spikes'})],
		'record_analogs': [False, False],
		'analog_device_pars': [None, None],
	}
	#############################################################################################################
	# SYNAPSE Parameters
	# ============================================================================================================
	connection_pars, neuron_pars = set_connection_defaults(default_set=connection_set, syn_pars=syn_pars,
	                                                       neuron_pars=neuron_pars)

	return ParameterSet(neuron_pars), ParameterSet(net_pars), ParameterSet(connection_pars)


def set_connection_defaults(syn_pars=None):
	"""
	Connection parameter defaults
	:param default_set:
	:return:
	"""
	print("\nLoading Default Connection Set - E/I populations, no topology, fast synapses")

	#############################################################################################################
	# SYNAPSE Parameters
	# ============================================================================================================
	synapses = syn_pars.connected_populations
	synapse_names = [n[1]+n[0] for n in synapses]
	synapse_models = syn_pars.synapse_models
	model_pars = syn_pars.synapse_model_parameters

	assert (
	np.mean([n in synapses for n in syn_pars.connected_populations]).astype(bool)), "Inconsistent Parameters"
	connection_pars = {
		'n_synapse_types': len(synapses),
		'synapse_types': synapses,
		'synapse_names': synapse_names,
		'topology_dependent': [False for _ in range(len(synapses))],
		'models': synapse_models,
		'model_pars': model_pars,
		'pre_computedW': syn_pars.pre_computedW,
		'weight_dist': syn_pars.weights,
		'delay_dist': syn_pars.delays,
		'conn_specs': syn_pars.conn_specs,
		'syn_specs': syn_pars.syn_specs,
	}

	return connection_pars
