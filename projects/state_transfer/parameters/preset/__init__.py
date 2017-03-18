__author__ = 'duarte'
import numpy as np
import sys
from modules.parameters import ParameterSet, copy_dict
from defaults.paths import paths
import itertools
from modules.parameters import Parameter, compare_dict


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


def set_neuron_defaults(default_set=1):
	"""
		Default single neuron parameters

		:param default_set: (int) - if applicable
		:return: ParameterSet
		"""
	if default_set == 1:
		print "\nLoading Default Neuron Set 2.1 - iaf_cond_exp, fixed voltage threshold, fixed absolute refractory " \
			  "time, Fast, exponential synapses (AMPA, GABAa), homogeneous parameters"
		neuron_pars = {
			'description': {
				'neurons': 'Leaky integrate-and-fire, fixed voltage threshold, fixed absolute refractory time',
				'synapses': r'Fast, exponential synapses ($\mathrm{AMPA}$, $\mathrm{GABA}_{A}$, homogeneous parameters'},
			'E': {
				'model': 'iaf_cond_exp',
				'C_m': 250.,
				'E_L': -70.0,
				'I_e': 0.,
				'tau_m': 15.,
				'V_m': -70.0,
				'V_th': -50.0,
				'V_reset': -60.0,
				'g_L': 16.7,
				't_ref': 2.,
				'tau_minus': 20., # What are these and why we bother setting them?
				'tau_minus_triplet': 200.,
				'E_ex': 0.,
				'g_ex': 1.8,
				'tau_syn_ex': 5.,
				'E_in': -70.,
				'g_in': 21.6,
				'tau_syn_in': 10.
			},
			'I': {
				'model': 'iaf_cond_exp',
				'C_m': 250.,
				'E_L': -70.0,
				'I_e': 0.,
				'tau_m': 15.,
				'V_m': -70.0,
				'V_th': -50.0,
				'V_reset': -60.0,
				'g_L': 16.7,
				't_ref': 2.,
				'tau_minus': 20.,
				'tau_minus_triplet': 200.,
				'E_ex': 0.,
				'g_ex': 1.8,
				'tau_syn_ex': 5.,
				'E_in': -70.,
				'g_in': 21.6,
				'tau_syn_in': 10.
			}
		}
	elif default_set == 2:
		print "\nLoading Default Neuron Set 2 (two pools, E1, I1, E2, I2 neurons) - iaf_cond_exp, fixed voltage " \
		      "threshold, fixed absolute refractory time, Fast, exponential synapses (AMPA, GABAa), " \
		      "homogeneous parameters"

		neuron_pars = {
			'description': {
				'neurons': 'Leaky integrate-and-fire, fixed voltage threshold, fixed absolute refractory time',
				'synapses': r'Fast, exponential synapses ($\mathrm{AMPA}$, $\mathrm{GABA}_{A}$, homogeneous parameters'},
			'E1': {
				'model': 'iaf_cond_exp',
				'C_m': 250.,
				'E_L': -70.0,
				'I_e': 0.,
				'tau_m': 15.,
				'V_m': -70.0,
				'V_th': -50.0,
				'V_reset': -60.0,
				'g_L': 16.7,
				't_ref': 2.,
				'tau_minus': 20., # What are these and why we bother setting them?
				'tau_minus_triplet': 200.,
				'E_ex': 0.,
				'g_ex': 1.8,
				'tau_syn_ex': 5.,
				'E_in': -70.,
				'g_in': 21.6,
				'tau_syn_in': 10.
			},
			'E2': {
				'model': 'iaf_cond_exp',
				'C_m': 250.,
				'E_L': -70.0,
				'I_e': 0.,
				'tau_m': 15.,
				'V_m': -70.0,
				'V_th': -50.0,
				'V_reset': -60.0,
				'g_L': 16.7,
				't_ref': 2.,
				'tau_minus': 20., # What are these and why we bother setting them?
				'tau_minus_triplet': 200.,
				'E_ex': 0.,
				'g_ex': 1.8,
				'tau_syn_ex': 5.,
				'E_in': -70.,
				'g_in': 21.6,
				'tau_syn_in': 10.
			},
			'I1': {
				'model': 'iaf_cond_exp',
				'C_m': 250.,
				'E_L': -70.0,
				'I_e': 0.,
				'tau_m': 15.,
				'V_m': -70.0,
				'V_th': -50.0,
				'V_reset': -60.0,
				'g_L': 16.7,
				't_ref': 2.,
				'tau_minus': 20.,
				'tau_minus_triplet': 200.,
				'E_ex': 0.,
				'g_ex': 1.8,
				'tau_syn_ex': 5.,
				'E_in': -70.,
				'g_in': 21.6,
				'tau_syn_in': 10.
			},
			'I2': {
				'model': 'iaf_cond_exp',
				'C_m': 250.,
				'E_L': -70.0,
				'I_e': 0.,
				'tau_m': 15.,
				'V_m': -70.0,
				'V_th': -50.0,
				'V_reset': -60.0,
				'g_L': 16.7,
				't_ref': 2.,
				'tau_minus': 20.,
				'tau_minus_triplet': 200.,
				'E_ex': 0.,
				'g_ex': 1.8,
				'tau_syn_ex': 5.,
				'E_in': -70.,
				'g_in': 21.6,
				'tau_syn_in': 10.
			},
		}
	else:
		raise NotImplementedError("Default set %s not implemented!" % str(default_set))
	return neuron_pars


def set_network_defaults(default_set=1, neuron_set=0, N=1250, **synapse_pars):
	"""
	Network default parameters
	:param default_set:
	:return:
	"""
	#TODO this can be simplified, based on the keys in the neuron dictionary...

	syn_pars = ParameterSet(synapse_pars)
	nE = 0.8 * N
	nI = 0.2 * N

	if default_set == 1:
		print("\nLoading Default Network Set 1 - One pool, standard BRN, no topology")

		rec_devices = rec_device_defaults(start=0.)
		neuron_pars = set_neuron_defaults(default_set=neuron_set)
		neuron_pars['description']['synapses'] += ' Non-adapting'
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
			                                 'label': ''}),
			                      copy_dict(rec_devices,
			                                {'model': 'spike_detector',
			                                 'record_to': ['memory'],
			                                 'interval': 0.1,
			                                 'label': ''})],
			'record_analogs': [False, False],
			'analog_device_pars': [None, None],
			'description': {'topology': 'None'}
		}
		#############################################################################################################
		# SYNAPSE Parameters
		# ============================================================================================================
		connection_pars = set_connection_defaults(syn_pars=syn_pars)

	elif default_set == 2:
		print("\nLoading Default Network Set 2 - Two-pool system")

		rec_devices = rec_device_defaults(start=0.)
		neuron_pars = set_neuron_defaults(default_set=neuron_set)

		#############################################################################################################
		# NETWORK Parameters
		# ===========================================================================================================
		net_pars = {
			'n_populations': 4,
			'pop_names': ['E1', 'I1', 'E2', 'I2'],
			'n_neurons': [int(nE), int(nI), int(nE), int(nI)],
			'neuron_pars': [neuron_pars['E1'], neuron_pars['I1'], neuron_pars['E2'], neuron_pars['I2']],
			'topology': [False, False, False, False],
			'topology_dict': [None, None, None, None],
			'record_spikes': [True, True, True, True],
			'spike_device_pars': [copy_dict(rec_devices,
			                                {'model': 'spike_detector',
			                                 'record_to': ['memory'],
			                                 'interval': 0.1,
			                                 'label': ''}),
			                      copy_dict(rec_devices,
			                                 {'model': 'spike_detector',
			                                  'record_to': ['memory'],
			                                  'interval': 0.1,
			                                  'label': ''}),
			                      copy_dict(rec_devices,
			                                {'model': 'spike_detector',
			                                 'record_to': ['memory'],
			                                 'interval': 0.1,
			                                 'label': ''}),
			                      copy_dict(rec_devices,
			                                 {'model': 'spike_detector',
			                                  'record_to': ['memory'],
			                                  'interval': 0.1,
			                                  'label': ''}),],
			'record_analogs': [False, False, False, False],
			'analog_device_pars': [None, None, None, None],
			'description': {'topology': 'None'}
		}
		#############################################################################################################
		# SYNAPSE Parameters
		# ============================================================================================================
		connection_pars = set_connection_defaults(syn_pars=syn_pars)

	elif default_set == 3:
		print "\nLoading Default Network Set 3 - Single Neuron synaptic input"
		syn_pars = ParameterSet(synapse_pars)
		if isinstance(neuron_set, int):
			neuron_pars = set_neuron_defaults(default_set=neuron_set)
		elif isinstance(neuron_set, dict) or isinstance(neuron_set, ParameterSet):
			neuron_pars = neuron_set
		parrots = {'model': 'parrot_neuron'}

		n_pars = {k: v for k, v in neuron_pars.items() if k != 'description'}
		pop_names = ['{0}'.format(str(n)) for n in n_pars.keys()]

		randomized_neuron_pars = [{} for _ in pop_names]
		randomized_neuron_pars.extend([{} for _ in n_pars.keys()])

		if len(neuron_pars.keys()) > 1:
			neuron_params = [n_pars[n] for n in n_pars.keys()]
		else:
			neuron_params = [n_pars[n_pars.keys()[0]]]
		neuron_params.extend([parrots for _ in n_pars.keys()])

		n_neurons = [1 for _ in n_pars.keys()]
		n_neurons.extend([eval("int(n{0})".format(str(n))) for n in n_pars.keys()])

		devices_flags = [True for _ in range(len(pop_names))]
		devices_flags.extend([False for _ in range(len(pop_names))])

		sds = [syn_pars.devices[1] for _ in range(len(pop_names))]
		sds.extend([None for _ in range(len(pop_names))])

		mms = [syn_pars.devices[0] for _ in range(len(pop_names))]
		mms.extend([None for _ in range(len(pop_names))])

		pop_names.extend(['{0}_inputs'.format(str(n)) for n in n_pars.keys()])

		net_pars = {
			'n_populations': len(pop_names),
			'pop_names': pop_names,
			'n_neurons': n_neurons,
			'neuron_pars': neuron_params,
			'randomize_neuron_pars': randomized_neuron_pars,
			'topology': [False for _ in range(len(pop_names))],
			'topology_dict': [None for _ in range(len(pop_names))],
			'record_spikes': devices_flags,
			'spike_device_pars': sds,
			'record_analogs': devices_flags,
			'analog_device_pars': mms,
			'description': {'topology': 'None'}
		}
		##############################################################
		connection_pars = set_connection_defaults(syn_pars=syn_pars)
	else:
		raise NotImplementedError("Default set %s not implemented!" % str(default_set))

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
		'description': {
			'connectivity': 'Sparse, Random with density 0.1 (all connections)',
			'plasticity': 'None'}
	}

	return connection_pars


def set_encoding_defaults(default_set=1, input_dimensions=1, n_encoding_neurons=0, encoder_neuron_pars=None,
                          **synapse_pars):
	"""

	:param default_set:
	:return:
	"""
	if default_set == 0:
		print("\nLoading Default Encoding Set 0 - Empty Settings (add background noise)")
		encoding_pars = {
			'description': {'general': r'',
							'specific': r'',
							'parameters': r''},
			'encoder': {
				'N': 0,
				'labels': [],
				'models': [],
				'model_pars': [],
				'n_neurons': [],
				'neuron_pars': [],
				'topology': [],
				'topology_dict': [],
				'record_spikes': [],
				'spike_device_pars': [],
				'record_analogs': [],
				'analog_device_pars': []},
			'generator': {
				'N': 0,
				'labels': [],
				'models': [],
				'model_pars': [],
				'topology': [],
				'topology_pars': []},
			'connectivity': {
				'synapse_name': [],
				'connections': [],
				'topology_dependent': [],
				'conn_specs': [],
				'syn_specs': [],
				'models': [],
				'model_pars': [],
				'weight_dist': [],
				'delay_dist': [],
				'preset_W': []},
			'input_decoder': None}

	elif default_set == 1:
		# ###################################################################
		# Encoding Type 1 - DC injection to target populations
		# ###################################################################
		gen_label = 'DC_input'
		keys = ['target_population_names', 'conn_specs', 'syn_specs', 'models', 'model_pars',
		        'weight_dist', 'delay_dist', 'preset_W']
		if not np.mean([n in synapse_pars.keys() for n in keys]).astype(bool):
			raise TypeError("Incorrect Synapse Parameters")
		syn_pars = ParameterSet(synapse_pars)
		# n_connections = len(syn_pars.target_population_names)
		connections = [(n, gen_label) for n in syn_pars.target_population_names]
		synapse_names = [gen_label+'syn' for _ in syn_pars.target_population_names]
		print("\nLoading Default Encoding Set 1 - DC input to {0}".format(str(syn_pars.target_population_names)))
		encoding_pars = {
			'encoder': {
				'N': 0,
				'labels': [],
				'models': [],
				'model_pars': [],
				'n_neurons': [],
				'neuron_pars': [],
				'topology': [],
				'topology_dict': [],
				'record_spikes': [],
				'spike_device_pars': [],
				'record_analogs': [],
				'analog_device_pars': []},
			'generator': {
				'N': input_dimensions,
				'labels': [gen_label],
				'models': ['step_current_generator'],
				'model_pars': [{'start': 0., 'stop': sys.float_info.max, 'origin': 0.}],
				'topology': [False for _ in range(input_dimensions)],
				'topology_pars': [None for _ in range(input_dimensions)]},
			'connectivity': {
				'synapse_name': synapse_names,
				'connections': connections,
				'topology_dependent': [False, False],
				'conn_specs': syn_pars.conn_specs,
				'syn_specs': syn_pars.syn_specs,
				'models': syn_pars.models,
				'model_pars': syn_pars.model_pars,
				'weight_dist': syn_pars.weight_dist,
				'delay_dist': syn_pars.delay_dist,
				'preset_W': syn_pars.preset_W},
			'input_decoder': None
		}

	elif default_set == 2:
		# ###################################################################
		# Encoding Type 2 - Deterministic spike encoding layer
		# ###################################################################
		rec_devices = rec_device_defaults()
		enc_label = 'NEF'
		keys = ['target_population_names', 'conn_specs', 'syn_specs', 'models', 'model_pars',
		        'weight_dist', 'delay_dist', 'preset_W']
		if not np.mean([n in synapse_pars.keys() for n in keys]).astype(bool):
			raise TypeError("Incorrect Synapse Parameters")
		syn_pars = ParameterSet(synapse_pars)
		# n_connections = len(syn_pars.target_population_names) + 1
		labels = [enc_label + '{0}'.format(str(n)) for n in range(input_dimensions)]
		connections = [(n, enc_label) for n in syn_pars.target_population_names]
		conn_specs = syn_pars.conn_specs
		conn_specs.insert(0, {'rule': 'all_to_all'})#None)
		synapse_names = [enc_label+'_'+n for n in syn_pars.target_population_names]
		connections.insert(0, ('NEF', 'StepGen'))
		synapse_names.insert(0, 'Gen_Enc')
		spike_device_pars = [copy_dict(rec_devices, {'model': 'spike_detector',
		                                             'label': 'input_Spikes'}) for _ in range(input_dimensions)]
		models = syn_pars.models
		models.insert(0, 'static_synapse')
		model_pars = syn_pars.model_pars
		model_pars.insert(0, {})
		weight_dist = syn_pars.weight_dist
		weight_dist.insert(0, 1.)
		delay_dist = syn_pars.delay_dist
		delay_dist.insert(0, 0.1)
		syn_specs = syn_pars.syn_specs
		syn_specs.insert(0, {})
		if hasattr(syn_pars, 'gen_to_enc_W'):
			preset_W = syn_pars.preset_W
			preset_W.insert(0, syn_pars.gen_to_enc_W)
		else:
			preset_W = syn_pars.preset_W

		print("\nLoading Default Encoding Set 2 - Deterministic spike encoding, {0} input populations of {1} [{2} " \
		      "neurons] connected to {3}".format(
				str(input_dimensions), str(n_encoding_neurons), str(encoder_neuron_pars['model']), str(
				syn_pars.target_population_names)))

		encoding_pars = {
			'encoder': {
				'N': input_dimensions,
				'labels': [enc_label for _ in range(input_dimensions)],
				'models': [enc_label for _ in range(input_dimensions)],
				'model_pars': [None for _ in range(input_dimensions)],
				'n_neurons': [n_encoding_neurons for _ in range(input_dimensions)],
				'neuron_pars': [encoder_neuron_pars for _ in range(input_dimensions)],
				'topology': [False for _ in range(input_dimensions)],
				'topology_dict': [None for _ in range(input_dimensions)],
				'record_spikes': [True for _ in range(input_dimensions)],
				'spike_device_pars': spike_device_pars,
				'record_analogs': [False for _ in range(input_dimensions)],
				'analog_device_pars': [None for _ in range(input_dimensions)]},
			'generator': {
				'N': 1,
				'labels': ['StepGen'],
				'models': ['step_current_generator'],
				'model_pars': [{'start': 0., 'stop': sys.float_info.max, 'origin': 0.}],
				'topology': [False],
				'topology_pars': [None]},
			'connectivity': {
				'connections': connections,
				'synapse_name': synapse_names,
				'topology_dependent': [False for _ in range(len(synapse_names))],
				'conn_specs': conn_specs,
				'models': models,
				'model_pars': model_pars,
				'weight_dist': weight_dist,
				'delay_dist': delay_dist,
				'preset_W': preset_W,
				'syn_specs': syn_specs},
			'input_decoder': {'encoder_label': enc_label}
		}
	elif default_set == 3:
		# ###################################################################
		# Encoding Type 3 - Stochastic spike encoding layer
		# ###################################################################
		gen_label = 'inh_poisson'
		keys = ['target_population_names', 'conn_specs', 'syn_specs', 'models', 'model_pars',
				'weight_dist', 'delay_dist', 'preset_W']
		if not all([n in synapse_pars.keys() for n in keys]):
			raise TypeError("Incorrect Synapse Parameters")

		syn_pars = ParameterSet(synapse_pars)
		connections = [(n, gen_label) for n in syn_pars.target_population_names]
		synapse_names = [gen_label + 'syn' for _ in syn_pars.target_population_names]

		print("\nLoading Default Encoding Set 3 - Stochastic spike encoding, independent realizations of "
		      "inhomogeneous Poisson processes connected to {0}".format(str(syn_pars.target_population_names)))

		encoding_pars = {
			'encoder': {
				'N': 0,
				'labels': [],
				'models': [],
				'model_pars': [],
				'n_neurons': [],
				'neuron_pars': [],
				'topology': [],
				'topology_dict': [],
				'record_spikes': [],
				'spike_device_pars': [],
				'record_analogs': [],
				'analog_device_pars': []},
			'generator': {
				'N': 1, #input_dimensions,
				'labels': ['inh_poisson'],
				'models': ['inh_poisson_generator'],
				'model_pars': [{'start': 0., 'stop': sys.float_info.max, 'origin': 0.}],
				'topology': [False],
				'topology_pars': [None]},
			'connectivity': {
				'synapse_name': synapse_names,
				'connections': connections,
				'topology_dependent': [False, False],
				'conn_specs': syn_pars.conn_specs,
				'syn_specs': syn_pars.syn_specs,
				'models': syn_pars.models,
				'model_pars': syn_pars.model_pars,
				'weight_dist': syn_pars.weight_dist,
				'delay_dist': syn_pars.delay_dist,
				'preset_W': syn_pars.preset_W}
		}
	elif default_set == 4:
		# ###################################################################
		# Encoding Type 3 - Precise Spatiotemporal spike encoding (Frozen noise)
		# ###################################################################
		gen_label = 'spike_pattern'
		keys = ['target_population_names', 'conn_specs', 'syn_specs', 'models', 'model_pars',
		        'weight_dist', 'delay_dist', 'preset_W']
		if not np.mean([n in synapse_pars.keys() for n in keys]).astype(bool):
			raise TypeError("Incorrect Synapse Parameters")
		syn_pars = ParameterSet(synapse_pars)
		connections = [(n, gen_label) for n in syn_pars.target_population_names]
		synapse_names = [gen_label+'syn' for _ in syn_pars.target_population_names]
		conn_tp = [False for _ in range(len(connections))]
		if hasattr(syn_pars, 'jitter'):
			jitter = syn_pars.jitter
		else:
			jitter = None

		print(("\nLoading Default Encoding Set 4 - Stochastic spike encoding, {0} fixed spike pattern templates "
		      "composed of {1} independent spike trains connected to {2}".format(
				str(input_dimensions), str(n_encoding_neurons), str(syn_pars.target_population_names))))

		encoding_pars = {
			'encoder': {
				'N': 0,
				'labels': [],
				'models': [],
				'model_pars': [],
				'n_neurons': [],
				'neuron_pars': [],
				'topology': [],
				'topology_dict': [],
				'record_spikes': [],
				'spike_device_pars': [],
				'record_analogs': [],
				'analog_device_pars': []},
			'generator': {
				'N': 1,
				'labels': ['spike_pattern'],
				'models': ['spike_generator'],
				'model_pars': [{'start': 0., 'stop': sys.float_info.max, 'origin': 0.}],
				'jitter': jitter,
				'topology': [False],
				'topology_pars': [None],
				'gen_to_enc_W': syn_pars.gen_to_enc_W},
			'connectivity': {
				'synapse_name': synapse_names,
				'connections': connections,
				'topology_dependent': conn_tp,
				'conn_specs': syn_pars.conn_specs,
				'syn_specs': syn_pars.syn_specs,
				'models': syn_pars.models,
				'model_pars': syn_pars.model_pars,
				'weight_dist': syn_pars.weight_dist,
				'delay_dist': syn_pars.delay_dist,
				'preset_W': syn_pars.preset_W},
			'input_decoder': {}
		}
	elif default_set == 5:
		# ###################################################################
		# Encoding Type 3 - Stochastic spike encoding layer - JUST TRYIN' @barni
		# ###################################################################
		gen_labels = ['inh_poisson{0}'.format(n) for n in range(input_dimensions)]
		keys = ['target_population_names', 'conn_specs', 'syn_specs', 'models', 'model_pars',
		        'weight_dist', 'delay_dist', 'preset_W']
		if not all([n in synapse_pars.keys() for n in keys]):
			raise TypeError("Incorrect Synapse Parameters")

		syn_pars 		= ParameterSet(synapse_pars)
		connections 	= [(n, gen_labels[n]) for n in syn_pars.target_population_names]
		synapse_names 	= [gen_labels[n]+'syn' for _ in syn_pars.target_population_names]
		print("\nLoading Default Encoding Set 3 - Stochastic spike encoding, independent realizations of "
		      "inhomogeneous Poisson processes connected to {0}".format(str(syn_pars.target_population_names)))

		encoding_pars = {
			'encoder': {
				'N': 0,
				'labels': [],
				'models': [],
				'model_pars': [],
				'n_neurons': [],
				'neuron_pars': [],
				'topology': [],
				'topology_dict': [],
				'record_spikes': [],
				'spike_device_pars': [],
				'record_analogs': [],
				'analog_device_pars': []},
			'generator': {
				'N': input_dimensions,
				'labels': gen_labels,
				'models': ['inh_poisson_generator'  for _ in range(input_dimensions)],
				'model_pars': [{'start': 0., 'stop': sys.float_info.max, 'origin': 0.} for _ in range(input_dimensions)],
				'topology': [False for _ in range(input_dimensions)],
				'topology_pars': [None for _ in range(input_dimensions)]},
			'connectivity': {
				'synapse_name': synapse_names,
				'connections': connections,
				'topology_dependent': [False, False],
				'conn_specs': syn_pars.conn_specs,
				'syn_specs': syn_pars.syn_specs,
				'models': syn_pars.models,
				'model_pars': syn_pars.model_pars,
				'weight_dist': syn_pars.weight_dist,
				'delay_dist': syn_pars.delay_dist,
				'preset_W': syn_pars.preset_W}
		}
	else:
		raise IOError("default_set not defined")

	return ParameterSet(encoding_pars)


def set_decoding_defaults(output_resolution=1., to_memory=True, **decoder_pars):
	"""

	:return:
	"""
	keys = ['decoded_population', 'state_variable', 'filter_time', 'readouts', 'sampling_times', 'reset_states',
	        'average_states', 'standardize']
	if not all([n in decoder_pars.keys() for n in keys]) or len(decoder_pars['decoded_population']) != \
			len(decoder_pars['state_variable']):
		raise TypeError("Incorrect Decoder Parameters")

	dec_pars = ParameterSet(decoder_pars)
	n_decoders = len(dec_pars.decoded_population)
	if to_memory:
		rec_device = rec_device_defaults(start=0., resolution=output_resolution)
	else:
		rec_device = rec_device_defaults(start=0., resolution=output_resolution, record_to='file')
	state_specs = []
	for state_var in dec_pars.state_variable:
		if state_var == 'spikes':
			state_specs.append({'tau_m': dec_pars.filter_time, 'interval': output_resolution})
		else:
			state_specs.append(copy_dict(rec_device, {'model': 'multimeter',
			                                          'record_n': None,
			                                          'record_from': [state_var]}))

	if 'N' in decoder_pars.keys():
		N = decoder_pars['N']
	else:
		N = len(dec_pars.readouts)
	if len(dec_pars.readout_algorithms) == N:
		readouts = [{'N': N, 'labels': dec_pars.readouts, 'algorithm': dec_pars.readout_algorithms} for _ in
		            range(n_decoders)]
	else:
		readouts = [{'N': N, 'labels': dec_pars.readouts, 'algorithm': [
			dec_pars.readout_algorithms[n]]} for n in range(n_decoders)]

	decoding_pars = {
		'state_extractor': {
			'N': n_decoders,
			'filter_tau': dec_pars.filter_time,
			'source_population': dec_pars.decoded_population,
			'state_variable': dec_pars.state_variable,
			'state_specs': state_specs,
			'reset_states': dec_pars.reset_states,
			'average_states': dec_pars.average_states,
			'standardize': dec_pars.standardize},
		'readout': readouts,
		'sampling_times': dec_pars.sampling_times,
		'output_resolution': output_resolution
	}
	return ParameterSet(decoding_pars)


def add_background_noise(encoding_pars, noise_pars):
	"""
	Adds a source of Poisson input to the specified populations (by modifying the encoding parameters)
	:param encoding_pars: original encoding parameters
	:param noise_pars: parameters of the noise process
	:return: encoding ParameterSet
	"""
	extra_pars = noise_pars.pop('additional_parameters')
	target_populations = noise_pars.pop('target_population_names')
	if 'generator_label' in noise_pars.keys():
		label = noise_pars['generator_label']
	else:
		label = 'X_noise'
	encoding_pars.generator.N += 1
	encoding_pars.generator.labels.append(label)
	encoding_pars.generator.models.append('poisson_generator')
	encoding_pars.generator.model_pars.append(noise_pars)
	encoding_pars.generator.topology.append(False)
	encoding_pars.generator.topology_pars.append(None)

	connections = [(x, label) for x in target_populations]
	encoding_pars.connectivity.synapse_name.extend([label for _ in range(len(connections))])
	encoding_pars.connectivity.connections.extend(connections)
	encoding_pars.connectivity.topology_dependent.extend([False for _ in range(len(connections))])
	if 'conn_specs' in extra_pars.keys():
		encoding_pars.connectivity.conn_specs.extend([extra_pars['conn_specs'] for _ in range(len(connections))])
	else:
		encoding_pars.connectivity.conn_specs.extend([{'rule': 'all_to_all'} for _ in range(len(connections))])
	if 'syn_specs' in extra_pars.keys():
		encoding_pars.connectivity.syn_specs.extend([extra_pars['syn_specs'] for _ in range(len(connections))])
	else:
		encoding_pars.connectivity.syn_specs.extend([{} for _ in range(len(connections))])
	if 'models' in extra_pars.keys():
		encoding_pars.connectivity.models.extend([extra_pars['models'] for _ in range(len(connections))])
	else:
		encoding_pars.connectivity.models.extend(['static_synapse' for _ in range(len(connections))])
	if 'model_pars' in extra_pars.keys():
		encoding_pars.connectivity.model_pars.extend([extra_pars['model_pars'] for _ in range(len(connections))])
	else:
		encoding_pars.connectivity.model_pars.extend([{} for _ in range(len(connections))])
	if 'weight_dist' in extra_pars.keys():
		encoding_pars.connectivity.weight_dist.extend([extra_pars['weight_dist'] for _ in range(len(connections))])
	else:
		encoding_pars.connectivity.weight_dist.extend([1.0 for _ in range(len(connections))])
	if 'delay_dist' in extra_pars.keys():
		encoding_pars.connectivity.delay_dist.extend([extra_pars['delay_dist'] for _ in range(len(connections))])
	else:
		encoding_pars.connectivity.delay_dist.extend([{} for _ in range(len(connections))])
	encoding_pars.connectivity.preset_W.extend([None for _ in range(len(connections))])


def set_report_defaults(default_set, run_type, paths, kernel_pars, neuron_pars, net_pars, connection_pars,
                        encoding_pars, decoding_pars):
	"""

	:param default_set:
	:return:
	"""
	# ==============================================================================================================
	if net_pars.items():
		topology_description = net_pars.description.topology
	else:
		topology_description = ''
	if connection_pars.items():
		connectivity_description = connection_pars.description.connectivity
		plasticity_description = connection_pars.description.plasticity
	else:
		connectivity_description = ''
		plasticity_description = ''
	if neuron_pars.items():
		neurons_description = neuron_pars.description.neurons
		synapse_description = neuron_pars.description.synapses
	else:
		neurons_description = ''
		synapse_description = ''

	input_full_description = ''
	connectivity_full_description = ''
	if encoding_pars.items():
		input_description = encoding_pars.description.general
		input_specific_description = encoding_pars.description.specific
		input_parameters = encoding_pars.description.parameters
		input_generators = list(itertools.chain(encoding_pars.generator.models))
		for n_inp, inp in enumerate(input_generators):
			targets = [xx[0] for xx in encoding_pars.connectivity.connections if xx[1] ==
			           encoding_pars.generator.labels[n_inp]]
			input_full_description += '{0} & {1} & {2}'.format(r'\verb+' + inp + '+', str(targets),
			                                                   input_specific_description + r'\tabularnewline' + ' \hline ')
	else:
		input_description = ''
		input_specific_description = ''
		input_parameters = ''
		input_full_description = ''
	if decoding_pars.items():
		measurements_description = decoding_pars.description.measurements
	else:
		measurements_description = ''

	if isinstance(net_pars['neuron_pars'][0], list):
		neuron_models = [x['model'] for x in list(itertools.chain(*net_pars['neuron_pars'])) if x != 'global' and
		x.has_key('model')]
		N_neurons = list(itertools.chain(net_pars['n_neurons']))
		pops = list(itertools.chain(net_pars['pop_names']))
	else:
		neuron_models = [x['model'] for x in list(itertools.chain(net_pars['neuron_pars'])) if x != 'global' and
		                 x.has_key('model')]
		N_neurons = list(itertools.chain(net_pars['n_neurons']))
		pops = list(itertools.chain(net_pars['pop_names']))

	#### network description (populations) ##############################
	population_full_description = ''
	population_parameters = ''
	for idxxx, nnn in enumerate(pops):
		population_full_description += ('{0} & {1} & {2}'.format(str(nnn), r'\verb+' + str(neuron_models[idxxx]) + '+',
		                                                         str(N_neurons[
			                                                             idxxx])) + r'\tabularnewline' + ' \hline ')
		population_parameters += ('$N^{0}$ & {1} & {2}'.format('{' + str(nnn) + '}', str(N_neurons[idxxx]),
		                                                       str(
			                                                       nnn) + ' Population Size') + r'\tabularnewline' + ' \hline ')
	if connection_pars.items():
		#### Connectivity description ########################################
		connectivity_parameters = r'\epsilon & $0.1$ & Connection Probability (for all synapses)' + r'\tabularnewline' + ' \hline '
		for idx in range(connection_pars['n_synapse_types']):
			connectivity_full_description += ('{0} & {1} & {2} & {3}'.format(str(connection_pars['synapse_names'][idx]),
		                                                                 str(connection_pars['synapse_types'][idx][1]),
		                                                                 str(connection_pars['synapse_types'][idx][0]),
		                                                                 'Random, ') + r'\tabularnewline' + ' \hline ')
			connectivity_parameters += '$w^{0}$ & {1} & Connection strength'.format('{' + str(connection_pars[
				                                                                                  'synapse_names'][
				                                                                                  idx]) + '}', str(
																				connection_pars['weight_dist'][idx])) + \
		                                                                         r'\tabularnewline' + ' \hline '
	else:
		connectivity_parameters = r'' + r'\tabularnewline' + ' \hline '

	if default_set == 5:
		# ### Extract single neuron parameters from dictionaries ###############
		keys = neuron_pars.keys()
		keys.remove('description')
		keys.remove('label')
		neuron_pars_description = {}
		for neuron_name in keys:
			neuron_pars_description.update({neuron_name: {
				'Parameters': [
					Parameter(name='C_{m}', value=neuron_pars[neuron_name]['C_m'], units=r'\pF'),
					Parameter(name='V_{\mathrm{rest}}', value=neuron_pars[neuron_name]['E_L'], units=r'\mV'),
					Parameter(name='V_{\mathrm{reset}}', value=neuron_pars[neuron_name]['V_reset'], units=r'\mV'),
					Parameter(name='V_{\mathrm{th}}', value=neuron_pars[neuron_name]['V_th'], units=r'\mV'),
					Parameter(name=r'g_{\mathrm{leak}}', value=neuron_pars[neuron_name]['g_L'], units=r'\ms'),
					Parameter(name=r'E_{E}', value=neuron_pars[neuron_name]['E_ex'], units=r'\mV'),
					Parameter(name=r'E_{I}', value=neuron_pars[neuron_name]['E_in'], units=r'\mV'),
					Parameter(name=r'\tau_{E}', value=neuron_pars[neuron_name]['tau_syn_ex'], units=r'\ms'),
					Parameter(name=r'\tau_{I}', value=neuron_pars[neuron_name]['tau_syn_in'], units=r'\ms'),
					Parameter(name='t_{\mathrm{ref}}', value=neuron_pars[neuron_name]['t_ref'], units=r'\ms')],
				'Descriptions': [
					'Membrane Capacitance',
					'Resting Membrane Potential',
					'Reset Potential',
					'Firing Threshold',
					'Membrane leak conductance',
					'Excitatory Reversal potential',
					'Inhibitory Reversal potential',
					'Excitatory Synaptic time constants',
					'Inhibitory Synaptic time constants',
					'Absolute Refractory time']}
			})

		neuron_pars = list(itertools.chain(net_pars['neuron_pars']))
		if len(neuron_pars) == 1:
			neuron_name = [pops[0]]
		else:
			if np.mean([compare_dict(x, y) for x, y in zip(neuron_pars, neuron_pars[1:])]) == 1.:
				neuron_name = [pops[0]]
			else:
				neuron_name = [n_pop for idx_pop, n_pop in enumerate(pops) if neuron_pars[idx_pop][
					'model'] != 'parrot_neuron']

		neuron_parameters = ''
		for idd, nam in enumerate(neuron_name):
			neuron_parameters += (
				'{0} & {1} & {2}'.format('Name', str(nam), 'Neuron name') + r'\tabularnewline ' + ' \hline ')

			for iid, par in enumerate(neuron_pars_description[nam]['Parameters']):
				name = par.name
				value = str(par.value)
				units = str(par.units)

				neuron_parameters += ('{0} & {1} & {2}'.format('$' + name + '$', '$' + value + units + '$',
				                                               neuron_pars_description[nam]['Descriptions'][iid]) +
				                      r'\tabularnewline ' + ' \hline ')

		#### Synapse parameters ######################################
		synapse_parameters = 'None & None & None' + r'\tabularnewline' + ' \hline '

		#### Plasticity Parameters ###################################
		plasticity_parameters = 'None & None & None' + r'\tabularnewline' + ' \hline '

		########################################
		report_pars = {
			'report_templates_path': paths[run_type]['report_templates_path'],
			'report_path': paths[run_type]['report_path'],
			'report_filename': kernel_pars['data_prefix'] + '.tex',
			'table_fields': {
				'{{ net_populations_description }}': '{0} Main Populations: {1}'.format(str(net_pars['n_populations']),
				                                                                        str(net_pars['pop_names'])),
				'{{ net_populations_topology }}': topology_description,
				'{{ net_connectivity_description }}': connectivity_description,
				'{{ net_neuron_models }}': neurons_description,
				'{{ net_synapse_models }}': synapse_description,
				'{{ net_plasticity }}': plasticity_description,
				'{{ input_general_description }}': input_description,
				'{{ input_description }}': input_full_description,
				'{{ measurements_description }}': measurements_description,

				'{{ population_names_elements_sizes }}': population_full_description,
				'{{ connectivity_name_src_tgt_pattern }}': connectivity_full_description,
				'{{ measurements }}': measurements_description,
				'{{ population_parameters }}': population_parameters,
				'{{ connectivity_parameters }}': connectivity_parameters,
				'{{ neuron_parameters }}': neuron_parameters,
				'{{ synapse_parameters }}': synapse_parameters,
				'{{ plasticity_parameters }}': plasticity_parameters,
				'{{ input_parameters }}': input_parameters,
			},
		}

	return ParameterSet(report_pars)