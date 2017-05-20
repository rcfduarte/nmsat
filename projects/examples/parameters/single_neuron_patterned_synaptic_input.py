__author__ = 'duarte'
import copy
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
		transient_time=0.,
		sim_time=1000.)

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

	# create a multimeter recording device
	multimeter = copy_dict(rec_device_defaults(), {'model': 'multimeter',
												   'label': 'single_neuron_Vm',
												   'record_from': ['V_m', 'g_ex', 'g_in'],
												   'record_n': 1})

	net_pars = ParameterSet({
		'n_populations': len(neuron_pars.keys()),
		'pop_names': pop_names,
		'n_neurons': n_neurons,
		'neuron_pars': neuron_params,
		'randomize_neuron_pars': [{'V_m': (np.random.uniform, {'low': -70., 'high': -50.})}],
		'topology': [False for _ in neuron_pars.keys()],
		'topology_dict': [None for _ in neuron_pars.keys()],
		'record_spikes': [True for _ in neuron_pars.keys()],
		'spike_device_pars': [rec_device_defaults(device_type='spike_detector', label='single_neuron_spikes') for _ in
		                      neuron_pars.keys()],
		'record_analogs': [True for _ in neuron_pars.keys()],
		'analog_device_pars': [copy_dict(multimeter) for _ in neuron_pars.keys()]

	})
	neuron_pars = ParameterSet(neuron_pars)
	# ##################################################################################################################
	# Input Parameters
	# ##################################################################################################################
	inp_resolution = 1.

	input_pars = {
		'noise':
			{'N': 1,
			 'label': 'OU_generator',
			 'noise_source': ['OU'],
			 'noise_pars': {'dt': inp_resolution, 'tau': 30., 'sigma': 20.,
							'y0': 15., 'label': 'OU_generator'},
			 'rectify': False,
			 'start_time': 0.,
			 'stop_time': sys.float_info.max,
			 'resolution': inp_resolution, }}

	# ##################################################################################################################
	# Encoding Parameters
	# ##################################################################################################################
	encoder_delay = 0.1
	w_in = 1.
	w_E = 20.
	w_I = w_E * -8.
	n_parrots = 10000

	encoding_pars_ch1 = {
		'encoder': {
			'N': 1,
			'labels': ['parrot_exc'],
			'models': ['parrot_neuron'],
			'model_pars': [None],
			'n_neurons': [int(0.8 * n_parrots)],
			'neuron_pars': [{'model': 'parrot_neuron'}],
			'topology': [False],
			'topology_dict': [None, ],
			'record_spikes': [False,],
			'spike_device_pars': [{},],
			'record_analogs': [False,],
			'analog_device_pars': [{}]},

		'generator': {
			'N': 1,
			'labels': ['inh_generator_ch1'],
			'models': ['inh_poisson_generator'],
			'model_pars': [{'start': 0., 'stop': sys.float_info.max, 'origin': 0.},],
			'topology': [False],
			'topology_pars': [None]},

		'connectivity': {
			'synapse_name': ['static_synapse', 'static_synapse'],
			'connections': [('parrot_exc', 'inh_generator_ch1'), (pop_names[0], 'parrot_exc')],
			'topology_dependent': [False, False],
			'conn_specs': [{'rule': 'all_to_all'}, {'rule': 'all_to_all'}],
			'syn_specs': [{}, {}],
			'models': ['static_synapse', 'static_synapse'],
			'model_pars': [{}, {}],
			'weight_dist': [w_in, w_E],
			'delay_dist': [encoder_delay, encoder_delay],
			'preset_W': [None, None]}
	}

	# copy and update the encoding parameters for the second (inhibitory) channel
	encoding_pars_ch2 = copy.deepcopy(encoding_pars_ch1)
	encoding_pars_ch2['encoder'].update({'labels': ['parrot_inh'],
										 'n_neurons': [int(0.2 * n_parrots)]})
	encoding_pars_ch2['generator'].update({'labels': ['inh_generator_ch2']})
	encoding_pars_ch2['connectivity'].update({'connections': [('parrot_inh', 'inh_generator_ch2'),
															  (pop_names[0], 'parrot_inh')],
											  'weight_dist': [w_in, w_I]})

	encoding_pars_ch1 = ParameterSet(encoding_pars_ch1)
	encoding_pars_ch2 = ParameterSet(encoding_pars_ch2)
	# ##################################################################################################################
	# RETURN dictionary of Parameters dictionaries
	# ==================================================================================================================
	return dict([('kernel_pars', 	kernel_pars),
				 ('input_pars', 	input_pars),
				 ('neuron_pars', 	neuron_pars),
				 ('net_pars',	 	net_pars),
				 ('encoding_ch1_pars', 	encoding_pars_ch1),
				 ('encoding_ch2_pars', 	encoding_pars_ch2)])


