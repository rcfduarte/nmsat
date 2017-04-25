import sys
sys.path.insert(0, "../")
from preset import *
import numpy as np
from auxiliary_fcns import determine_lognormal_parameters

"""
spike_triggered_average
- run with spike_ta in computations
- debug with script single_neuron_sta.py
"""
__author__ = 'duarte'

run = 'local'
data_label = 'spike_triggered_average_test'
project_label = 'Timescales'
E_spike_times = []#np.arange(600., 10000., 200.)
I1_spike_times = np.arange(700., 10000.-200., 200.)
I2_spike_times = []
window = [-10., 20.]
neuron_type = 'E'
n_neurons = 1
heterogeneity = {'neuron': True, 'structural': False, 'synaptic': False}

# ######################################################################################################################
# PARAMETER RANGE declarations
# ======================================================================================================================
parameter_range = {
	}


########################################################################################################################
def build_parameters():
	# ######################################################################################################################
	# System / Kernel Parameters
	# ######################################################################################################################
	system = dict(
		nodes=1,
		ppn=8,
		mem=32,
		walltime='00:20:00:00',
		queue='singlenode',
		transient_time=0.,
		sim_time=10000.)

	kernel_pars = set_kernel_defaults(run_type=run, data_label=data_label, **system)
	kernel_pars.update({'window': window})
	# ######################################################################################################################
	# Neuron, Synapse and Network Parameters
	# ######################################################################################################################
	randomized_pars = randomize_neuron_parameters(heterogeneity['neuron'])
	if heterogeneity['neuron']:
		neuron_set = 1.2
	else:
		neuron_set = 1.1
		randomized_pars.update({neuron_type: {}})

	N = 25000
	nE = 0.8 * N
	nI = 0.2 * N
	nI1 = 0.35 * nI
	nI2 = 0.65 * nI

	# connection densities
	epsilon = {'EE': 0.168, 'I1E': 0.575, 'I2E': 0.242,
	           'EI1': 0.6, 'EI2': 0.465, 'I1I1': 0.55,
	           'I2I1': 0.241, 'I1I2': 0.379, 'I2I2': 0.381,}
	# connection delays
	if heterogeneity['synaptic']:
		delays = {'EE': {'distribution': 'lognormal', 'mu': 1.8, 'sigma': 0.25},
		          'I1E': {'distribution': 'lognormal', 'mu': 1.2, 'sigma': 0.2},
		          'I2E': {'distribution': 'lognormal', 'mu': 1.5, 'sigma': 0.2},
		          'EI1': {'distribution': 'lognormal', 'mu': 0.8, 'sigma': 0.1},
		          'EI2': {'distribution': 'lognormal', 'mu': 1.5, 'sigma': 0.2},
		          'I1I1': {'distribution': 'lognormal', 'mu': 1., 'sigma': 0.1},
		          'I2I1': {'distribution': 'lognormal', 'mu': 1.2, 'sigma': 0.3},
		          'I1I2': {'distribution': 'lognormal', 'mu': 1.5, 'sigma': 0.5},
		          'I2I2': {'distribution': 'lognormal', 'mu': 1.5, 'sigma': 0.3}}
	else:
		delays = {'EE': 1.8, 'I1E': 1.2, 'I2E': 1.5,
		          'EI1': 0.8, 'EI2': 1.5, 'I1I1': 1.,
		          'I2I1': 1.2, 'I1I2': 1.5, 'I2I2': 1.5, }

	# connection weights
	gamma_E = 4.0
	gamma_I1 = 2.0
	gamma_I2 = 0.1

	wEE = 1.
	wEI1 = gamma_E * wEE
	wEI2 = gamma_E * wEE
	wI1E = 1.
	wI1I1 = gamma_I1 * wI1E
	wI1I2 = gamma_I1 * wI1E
	wI2E = 1.
	wI2I1 = gamma_I2 * wI2E
	wI2I2 = gamma_I2 * wI2E

	if heterogeneity['synaptic']:
		weights = {'EE': {'distribution': 'lognormal_clipped', 'mu': wEE, 'sigma': 1., 'low': 0.0001, 'high': 100*wEE},
		           'I1E': {'distribution': 'lognormal_clipped', 'mu': wI1E, 'sigma': 1., 'low': 0.0001, 'high': 100*wI1E},
		           'I2E': {'distribution': 'lognormal_clipped', 'mu': wI2E, 'sigma': 1., 'low': 0.0001, 'high': 100*wI2E},
		           'EI1': {'distribution': 'lognormal_clipped', 'mu': wEI1, 'sigma': 1., 'low': 0.0001, 'high': 100*wEI1},
		           'EI2': {'distribution': 'lognormal_clipped', 'mu': wEI2, 'sigma': 1., 'low': 0.0001, 'high': 100*wEI2},
		           'I1I1': {'distribution': 'lognormal_clipped', 'mu': wI1I1, 'sigma': 1., 'low': 0.0001,
		                    'high': 100*wI1I1},
		           'I2I1': {'distribution': 'lognormal_clipped', 'mu': wI2I1, 'sigma': 1., 'low': 0.0001,
		                    'high': 100*wI2I1},
		           'I1I2': {'distribution': 'lognormal_clipped', 'mu': wI1I2, 'sigma': 1., 'low': 0.0001,
		                    'high': 100*wI1I2},
		           'I2I2': {'distribution': 'lognormal_clipped', 'mu': wI2I2, 'sigma': 1., 'low': 0.0001,
		                    'high': 100*wI2I2},}
	else:
		weights = {'EE': wEE, 'I1E': wI1E, 'I2E': wI2E,
		           'EI1': wEI1, 'EI2': wEI2, 'I1I1': wI1I1,
		           'I2I1': wI2I1, 'I1I2': wI1I2, 'I2I2': wI2I2,}

	synapse_keys = ['EE', 'I1E', 'I2E', 'EI1', 'EI2', 'I1I1', 'I2I1', 'I1I2', 'I2I2']

	neuron_pars = set_neuron_defaults(default_set=neuron_set)
	multimeter = rec_device_defaults(device_type='multimeter')
	multimeter.update({'record_from': ['V_m', 'I_ex', 'I_in'], 'record_n': 1})
	sd = rec_device_defaults(device_type='spike_detector')
	net_pars = ParameterSet({
		'n_populations': 1,
		'pop_names': [neuron_type],
		'n_neurons': [1],
		'neuron_pars': [neuron_pars[neuron_type]],
		'randomize_neuron_pars': [randomized_pars[neuron_type]],
		'topology': [False],
		'topology_dict': [None],
		'record_spikes': [True],
		'spike_device_pars': [sd],
		'record_analogs': [True],
		'analog_device_pars': [multimeter],
		'description': {'topology': 'None'}})
	connection_pars = {}

	# ######################################################################################################################
	# Input Parameters
	# ######################################################################################################################
	nu_x = 5.
	k_X = epsilon['{0}E'.format(neuron_type)] * nE
	k_E = epsilon['{0}E'.format(neuron_type)] * nE
	k_I1 = epsilon['{0}I1'.format(neuron_type)] * nI1
	k_I2 = epsilon['{0}I2'.format(neuron_type)] * nI2
	poiss_gen = dict(start=0., stop=sys.float_info.max, origin=0., rate=5.)

	encoding_pars = ParameterSet({
		'generator': {
			'N': 4,
			'labels': ['X_Noise', 'E_Noise', 'I1_Noise', 'I2_Noise'],
			'models': ['poisson_generator', 'poisson_generator', 'poisson_generator', 'poisson_generator'],
			'model_pars': [copy_dict(poiss_gen, {'rate': nu_x * k_X}),
			               copy_dict(poiss_gen, {'rate': nu_x * k_E}),
			               copy_dict(poiss_gen, {'rate': nu_x * k_I1}),
			               copy_dict(poiss_gen, {'rate': nu_x * k_I2})],
			'topology': [False, False, False, False],
			'topology_pars': [None, None, None, None]},

		'encoder': {
			'N': 4,
			'labels': ['X_inputs', 'E_inputs', 'I1_inputs', 'I2_inputs'],
			'models': ['parrot_neuron', 'parrot_neuron', 'parrot_neuron', 'parrot_neuron'],
			'model_pars': [None, None, None, None],
			'n_neurons': [1, 1, 1, 1],
			'neuron_pars': [{'model': 'parrot_neuron'}, {'model': 'parrot_neuron'}, {'model': 'parrot_neuron'},
			                {'model': 'parrot_neuron'}],
			'topology': [False, False, False, False],
			'topology_dict': [None, None, None, None],
			'record_spikes': [False, False, False, False],
			'spike_device_pars': [{}, {}, {}, {}],
			'record_analogs': [False, False, False, False],
			'analog_device_pars': [{}, {}, {}, {}]},

		'connectivity': {
			'connections': [('X_inputs', 'X_Noise'), ('E_inputs', 'E_Noise'), ('I1_inputs', 'I1_Noise'), ('I2_inputs',
			                                                                                              'I2_Noise'),
			                (neuron_type, 'X_inputs'), (neuron_type, 'E_inputs'),
			                (neuron_type, 'I1_inputs'), (neuron_type, 'I2_inputs')],
			'synapse_name': ['XnoiseX', 'EnoiseE', 'I1noiseI1', 'I2noiseI2',
			                 'Xinputs', 'Einputs', 'I1inputs', 'I2inputs'],
			'topology_dependent': [False for _ in range(8)],
			'conn_specs': [{'rule': 'all_to_all'} for _ in range(8)],
			'syn_specs': [{}, {}, {}, {},
			              {'receptor_type': 1}, {'receptor_type': 1},
			              {'receptor_type': 1}, {'receptor_type': 1}],
			'models': ['static_synapse', 'static_synapse', 'static_synapse', 'static_synapse',
			           'multiport_synapse', 'multiport_synapse', 'multiport_synapse', 'multiport_synapse'],
			'model_pars': [{}, {}, {}, {},
			               {'receptor_types': [1., 3.]}, {'receptor_types': [1., 3.]},
			               {'receptor_types': [2., 4.]}, {'receptor_types': [2., 4.]}],
			'weight_dist': [1., 1., 1., 1.,
			                weights['{0}E'.format(neuron_type)], weights['{0}E'.format(neuron_type)],
			                weights['{0}I1'.format(neuron_type)], weights['{0}I2'.format(neuron_type)],],
			'delay_dist': [0.1, 0.1, 0.1, 0.1,
			                delays['{0}E'.format(neuron_type)], delays['{0}E'.format(neuron_type)],
			                delays['{0}I1'.format(neuron_type)], delays['{0}I2'.format(neuron_type)], ],
			'preset_W': [None for _ in range(8)]},
	})
	###########
	perturbation_pars = set_encoding_defaults(default_set=0)

	perturbation_pars.generator.update({
		'N': 3,
		'labels': ['E_input', 'I1_input', 'I2_input'],
		'models': ['spike_generator', 'spike_generator', 'spike_generator'],
		'model_pars': [{'spike_times': E_spike_times}, {'spike_times': I1_spike_times}, {'spike_times': I2_spike_times}],
		'topology': [False, False, False],
		'topology_pars': [None, None, None],
	})
	perturbation_pars.encoder.update({
		'N': 3,
		'labels': ['E_parr', 'I1_parr', 'I2_parr'],
		'models': ['parrot_neuron', 'parrot_neuron', 'parrot_neuron'],
		'models_pars': [None, None, None],
		'n_neurons': [1, 1, 1],
		'neuron_pars': [{'model': 'parrot_neuron'}, {'model': 'parrot_neuron'}, {'model': 'parrot_neuron'}],
		'topology': [False, False, False],
		'topology_dict': [None, None, None],
		'record_spikes': [False, False, False],
		'spike_device_pars': [{}, {}, {}],
		'record_analogs': [False, False, False],
		'analog_device_pars': [{}, {}, {}]})

	perturbation_pars.connectivity.update({
		'connections': [('E_parr', 'E_input'), ('I1_parr', 'I1_input'), ('I2_parr', 'I2_input'),
			(neuron_type + '_clone', 'E_parr'), (neuron_type + '_clone', 'I1_parr'),
		    (neuron_type + '_clone', 'I2_parr'), ],
		'synapse_name': ['EpPar', 'I1pPar', 'I2pPar', 'Eextra', 'I1extra', 'I2extra'],
		'topology_dependent': [False for _ in range(6)],
		'conn_specs': [{'rule': 'all_to_all'} for _ in range(6)],
		'syn_specs': [{}, {}, {}, {'receptor_type': 1}, {'receptor_type': 1}, {'receptor_type': 1}],
		'models': ['static_synapse', 'static_synapse', 'static_synapse',
		           'multiport_synapse', 'multiport_synapse', 'multiport_synapse'],
		'model_pars': [{}, {}, {},
		               {'receptor_types': [1., 3.]}, {'receptor_types': [2., 4.]}, {'receptor_types': [2., 4.]}],
		'weight_dist': [1., 1., 1.,
		                weights['{0}E'.format(neuron_type)],
		                weights['{0}I1'.format(neuron_type)],
		                weights['{0}I2'.format(neuron_type)],],
		'delay_dist': [0.1, 0.1, 0.1,
		               delays['{0}E'.format(neuron_type)],
		                delays['{0}I1'.format(neuron_type)],
		                delays['{0}I2'.format(neuron_type)],],
		'preset_W': [None for _ in range(6)]})
	# ##################################################################################################################
	# RETURN dictionary of Parameters dictionaries
	# ==================================================================================================================
	return dict([('kernel_pars', kernel_pars),
	             ('neuron_pars', neuron_pars),
	             ('net_pars', net_pars),
	             ('connection_pars', connection_pars),
	             ('encoding_pars', encoding_pars),
	             ('perturbation_pars', perturbation_pars)])