__author__ = 'duarte'
import numpy as np
import sys
from modules.parameters import ParameterSet, copy_dict
from defaults.paths import paths


def set_kernel_defaults(run_type='local', data_label='', **system_pars):
	"""
	Return pre-defined kernel parameters dictionary
	:param run_type: string with system specificity
	:param data_label: label of current experiment
	:param project_label:
	:param system_pars: dictionary with the most relevant system parameters
	:return: kernel ParameterSet
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
		'resolution': 0.01,
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


def set_neuron_defaults():
	"""
	Default single neuron parameters

	:param default_set: (int)
	:return: ParameterSet
	"""
	print("\nLoading Default Neuron Set - iaf_psc_exp_ps, fixed voltage threshold, fixed absolute refractory "
	      "time, Fast, exponential current-based synapses, homogeneous parameters")
	neuron_pars = {
		'E': {
			'model': 'iaf_psc_exp_ps',
			'C_m': 250.0,
			'E_L': 0.0,
			'V_reset': 0.0,
			'V_th': 15.,
			't_ref': 2.,
			'tau_syn_ex': 2.,
			'tau_syn_in': 2.,
			'tau_m': 20.
		},
		'I': {
			'model': 'iaf_psc_exp_ps',
			'C_m': 250.0,
			'E_L': 0.0,
			'V_reset': 0.0,
			'V_th': 15.,
			't_ref': 2.,
			'tau_syn_ex': 2.,
			'tau_syn_in': 2.,
			'tau_m': 20.
		},
	}
	return neuron_pars


def set_network_defaults(N=1250, kernel_pars=None, **synapse_pars):
	"""
	Network default parameters
	:param default_set:
	:return:
	"""
	print("\nLoading Default Network Set - Standard E/I BRN")
	syn_pars = ParameterSet(synapse_pars)
	nE = 0.8 * N
	nI = 0.2 * N
	if kernel_pars is not None:
		rec_devices = rec_device_defaults(start=0., resolution=kernel_pars['resolution'])
	else:
		rec_devices = rec_device_defaults(start=0.)

	neuron_pars = set_neuron_defaults()
	keys = neuron_pars.keys()
	if len(keys) > 1:
		neuron_params = [neuron_pars[n] for n in keys]
	else:
		neuron_params = [neuron_pars[keys[0]], neuron_pars[keys[0]]]
	#############################################################################################################
	# NETWORK Parameters
	# ===========================================================================================================
	net_pars = {
		'n_populations': 2,
		'pop_names': ['E', 'I'],
		'n_neurons': [int(nE), int(nI)],
		'neuron_pars': neuron_params,
		'topology': [False, False],
		'topology_dict': [None, None],
		'record_spikes': [True, True],
		'spike_device_pars': [copy_dict(rec_devices,
		                                {'model': 'spike_detector',
		                                 'record_to': ['memory'],
		                                 'label': ''}),
		                      copy_dict(rec_devices,
		                                {'model': 'spike_detector',
		                                 'record_to': ['memory'],
		                                 'label': ''})],
		'record_analogs': [False, False],
		'analog_device_pars': [None, None],
		'description': {'topology': 'None'}
	}
	#############################################################################################################
	# SYNAPSE Parameters
	# ============================================================================================================
	connection_pars = set_connection_defaults(syn_pars=syn_pars)

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


def set_encoding_defaults(default_set=1, input_dimensions=1, n_encoding_neurons=0, **synapse_pars):
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
	else:
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

		print("\nLoading Default Encoding Set 4 - Stochastic spike encoding, {0} fixed spike pattern templates "
		      "composed of {1} independent spike trains connected to {2}".format(
				str(input_dimensions), str(n_encoding_neurons), str(syn_pars.target_population_names)))

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
			                                          'record_from': [state_var],
			                                          }))

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


def add_parrots(encoding_pars, n_parrots, decode=True, **extra_pars):
	"""
	Attaches a layer of parrot neurons to the encoder (for cases when the generator is a spike-emmitting device)
	:param encoding_pars: original encoding parameters
	:param n_parrots: number of parrot neurons to attach (should be the same as the number of unique generators)
	:param decode: attach decoders and readouts to parrot neurons.. (only spikes can be read!)
	"""
	if extra_pars.items():
		conn_specs = extra_pars['conn_specs']
		presetW = extra_pars['preset_W']
	else:
		conn_specs = {'rule': 'one_to_one'}
		presetW = None
	rec_devices = rec_device_defaults()
	encoding_pars.encoder.N += 1
	encoding_pars.encoder.labels.extend(['parrots'])
	encoding_pars.encoder.models.extend(['parrot_neuron'])
	encoding_pars.encoder.model_pars.extend([None])
	encoding_pars.encoder.n_neurons.extend([n_parrots])
	encoding_pars.encoder.neuron_pars.extend([{'model': 'parrot_neuron'}])
	encoding_pars.encoder.topology.extend([False])
	encoding_pars.encoder.topology_dict.extend([None])
	encoding_pars.encoder.record_spikes.extend([True])
	encoding_pars.encoder.spike_device_pars.extend([copy_dict(rec_devices, {'model': 'spike_detector',
	                                             'label': 'input_Spikes'})])
	encoding_pars.encoder.record_analogs.extend([False])
	encoding_pars.encoder.analog_device_pars.extend([None])
	syn_name = encoding_pars.connectivity.synapse_name[0]  # all synapses from a device must have the same name!!
	encoding_pars.connectivity.synapse_name.extend([syn_name])
	encoding_pars.connectivity.connections.extend([('parrots', encoding_pars.generator.labels[0])])
	encoding_pars.connectivity.topology_dependent.extend([False])
	encoding_pars.connectivity.conn_specs.extend([conn_specs])
	encoding_pars.connectivity.syn_specs.extend([{}])
	encoding_pars.connectivity.models.extend(['static_synapse'])
	encoding_pars.connectivity.model_pars.extend([{}])
	encoding_pars.connectivity.weight_dist.extend([1.])
	encoding_pars.connectivity.delay_dist.extend([0.1])
	encoding_pars.connectivity.preset_W.extend([presetW])
	if decode:
		encoding_pars.input_decoder = {'encoder_label': 'parrots'}


def add_input_decoders(encoding_pars, input_decoder_pars, kernel_pars):
	"""
	Updates encoding parameters, adding decoders to the encoding layer
	:param encoding_pars: original encoding parameters to update
	:param encoder_label: label of encoder to readout
	:param input_decoder_pars: parameters of decoder to attach
	:param kernel_pars: main system parameters
	:return: updates encoding ParameterSet
	"""
	if not isinstance(input_decoder_pars, ParameterSet):
		input_decoder_pars = ParameterSet(input_decoder_pars)
	if isinstance(encoding_pars, ParameterSet):
		encoding_pars = encoding_pars.as_dict()
	enc_label = encoding_pars['input_decoder'].pop('encoder_label')
	encoding_pars['input_decoder'] = {}
	decoder_dict = copy_dict(input_decoder_pars.as_dict(), {'decoded_population': [enc_label for _ in range(len(
		input_decoder_pars.state_variable))]})
	resolution = decoder_dict.pop('output_resolution')

	input_decoder = set_decoding_defaults(output_resolution=resolution, kernel_pars=kernel_pars, **decoder_dict)
	encoding_pars.update({'input_decoder': input_decoder.as_dict()})

	return ParameterSet(encoding_pars)


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
