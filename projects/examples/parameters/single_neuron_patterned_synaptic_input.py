__author__ = 'duarte'
import sys
from preset import *
import numpy as np

"""
single_neuron_patterned_synaptic_input
- main spike_input stimulus processing
"""

run = 'local'
data_label = 'example2'

# ######################################################################################################################
# PARAMETER RANGE declarations
# ======================================================================================================================
parameter_range = {
}


def build_parameters():
	# ##################################################################################################################
	# System / Kernel Parameters
	# ##################################################################################################################
	system = dict(
		nodes=1,
		ppn=16,
		mem=32,
		walltime='01-00:00:00',
		queue='batch',
		transient_time=1000.,
		sim_time=2000.)

	kernel_pars = set_kernel_defaults(run_type=run, data_label=data_label, **system)

	# ##################################################################################################################
	# Neuron, Synapse and Network Parameters
	# ##################################################################################################################
	neuron_pars = {
		'AeifCondExp': {
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
				'tau_syn_in': 6.0,}
	}

	pop_names = ['{0}'.format(str(n)) for n in neuron_pars.keys()]
	n_neurons = [1 for _ in neuron_pars.keys()]
	if len(neuron_pars.keys()) > 1:
		neuron_params = [neuron_pars[n] for n in neuron_pars.keys()]
	else:
		neuron_params = [neuron_pars[neuron_pars.keys()[0]]]
	net_pars = ParameterSet({
		'n_populations': len(neuron_pars.keys()),
		'pop_names': pop_names,
		'n_neurons': n_neurons,
		'neuron_pars': neuron_params,
		'randomize_neuron_pars': [{'V_m': (np.random.uniform, {'low': -70., 'high': -50.})}],
		'topology': [False for _ in neuron_pars.keys()],
		'topology_dict': [None for _ in neuron_pars.keys()],
		'record_spikes': [True for _ in neuron_pars.keys()],
		'spike_device_pars': [None for _ in neuron_pars.keys()],#[copy_dict(multimeter, {'model': 'spike_detector'}) for _ in neuron_pars.keys()],
		'record_analogs': [False for _ in neuron_pars.keys()],
		'analog_device_pars': [None for _ in neuron_pars.keys()]
		# 'analog_device_pars': [copy_dict(multimeter, {'record_from': ['V_m'], 'record_n': 1}) for _ in
		# 					   neuron_pars.keys()],
	})
	neuron_pars = ParameterSet(neuron_pars)
	# ######################################################################################################################
	# Input Parameters
	# ######################################################################################################################
	inp_resolution = 0.1

	input_pars = {
		'noise':
			{'N': 2,
			 'noise_source': ['OU'],
			 'noise_pars': {'amplitude': 5., 'mean': 1., 'std': 0.25},
			 'rectify': False,
			 'start_time': 0.,
			 'stop_time': sys.float_info.max,
			 'resolution': inp_resolution, }}

	# ##################################################################################################################
	# Encoding Parameters
	# ##################################################################################################################
	n_afferents = 2
	encoder_delay = 0.1
	w_in = 90.

	encoding_pars = {
		'encoder': {
			'N': 2,
			'labels': ['parrot_blue', 'parrot_red'],
			'models': ['parrot_neuron', 'parrot_neuron'],
			'model_pars': [None, None],
			'n_neurons': [10, 10],
			'neuron_pars': [{'model': 'parrot_neuron'}, {'model': 'parrot_neuron'}],
			'topology': [False, False],
			'topology_dict': [None, None],
			'record_spikes': [True, True],
			'spike_device_pars': [None, None],
			'record_analogs': [False, False],
			'analog_device_pars': [None, None]},
		'generator': {
			'N': 2,
			'labels': ['inh_poisson_blue', 'inh_poisson_red'],
			'models': ['inh_poisson_generator', 'inh_poisson_generator'],
			'model_pars': [{'start': 0., 'stop': sys.float_info.max, 'origin': 0.}],
			'topology': [False, False],
			'topology_pars': [None, None]},
		'connectivity': {
			'synapse_name': ['static_synapse_blue', 'static_synapse_red'],
			'connections': [('parrot_blue', 'inh_poisson_blue'), ('parrot_red', 'inh_poisson_red')],
			'topology_dependent': [False, False],
			'conn_specs': [{'rule': 'one_to_one'}, {'rule': 'one_to_one'}],
			'syn_specs': [{}, {}],
			'models': ['static_synapse', 'static_synapse'],
			'model_pars': [{}],
			'weight_dist': [w_in, w_in],
			'delay_dist': [encoder_delay, encoder_delay],
			'preset_W': [None, None]}
	}

	encoding_pars['encoder']['n_neurons'] = [n_afferents]
	encoding_pars = ParameterSet(encoding_pars)

	# ##################################################################################################################
	# RETURN dictionary of Parameters dictionaries
	# ==================================================================================================================
	return dict([('kernel_pars', 	kernel_pars),
				 ('input_pars', 	input_pars),
				 ('neuron_pars', 	neuron_pars),
				 ('net_pars',	 	net_pars),
				 ('encoding_pars', 	encoding_pars),
				 # ('decoding_pars', 	decoding_pars),
				 # ('stim_pars', 		stim_pars),
				 # ('connection_pars',connection_pars)
				 ])


