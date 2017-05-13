import sys
sys.path.insert(0, "../")
from preset import *
import numpy as np
# from auxiliary_fcns import determine_lognormal_parameters

"""
single_neuron_synaptic_input
- to run with function single_neuron_synaptic_input in Computation
or with SingleNeuron_SynapticInput.py (debugging)
"""
__author__ = 'duarte'

run = 'local'
data_label = 'HT_singleneuron_RTF_homogeneous'
heterogeneity = {'synaptic': False, 'neuronal': False, 'structural': False}


# ######################################################################################################################
# PARAMETER RANGE declarations
# ======================================================================================================================
parameter_range = {
	'nu_x': np.arange(0., 50., 1.)
	}


########################################################################################################################
def build_parameters(nu_x):
	# ######################################################################################################################
	# System / Kernel Parameters
	# ######################################################################################################################
	system = dict(
		nodes=1,
		ppn=8,
		mem=32,
		walltime='00:20:00:00',
		queue='singlenode',
		transient_time=1000.,
		sim_time=10000.)

	kernel_pars = set_kernel_defaults(run_type=run, data_label=data_label, **system)

	# ######################################################################################################################
	# Neuron, Synapse and Network Parameters
	# ######################################################################################################################
	randomized_pars = randomize_neuron_parameters(heterogeneity['neuronal'])
	if heterogeneity['neuronal']:
		neuron_set = 1.2
	else:
		neuron_set = 1.1
		# check that Vm is randomized

	# Population sizes
	N = 25000
	nE = 0.8 * N
	nI = 0.2 * N
	nI1 = 0.35 * nI
	nI2 = 0.65 * nI

	epsilon, weights, delays = connection_parameters(heterogeneous=heterogeneity['synaptic'])

	# connection weights
	# gamma_E = 4.0
	# gamma_I1 = 2.0
	# gamma_I2 = 0.1
	#
	# wEE = 1.
	# wEI1 = gamma_E * wEE
	# wEI2 = gamma_E * wEE
	# wI1E = 1.
	# wI1I1 = gamma_I1 * wI1E
	# wI1I2 = gamma_I1 * wI1E
	# wI2E = 1.
	# wI2I1 = gamma_I2 * wI2E
	# wI2I2 = gamma_I2 * wI2E

	# if heterogeneity['synaptic']:
	# 	weights = {'EE': {'distribution': 'lognormal_clipped', 'mu': wEE, 'sigma': 1., 'low': 0.0001, 'high': 100*wEE},
	# 	           'I1E': {'distribution': 'lognormal_clipped', 'mu': wI1E, 'sigma': 1., 'low': 0.0001, 'high': 100*wI1E},
	# 	           'I2E': {'distribution': 'lognormal_clipped', 'mu': wI2E, 'sigma': 1., 'low': 0.0001, 'high': 100*wI2E},
	# 	           'EI1': {'distribution': 'lognormal_clipped', 'mu': wEI1, 'sigma': 1., 'low': 0.0001, 'high': 100*wEI1},
	# 	           'EI2': {'distribution': 'lognormal_clipped', 'mu': wEI2, 'sigma': 1., 'low': 0.0001, 'high': 100*wEI2},
	# 	           'I1I1': {'distribution': 'lognormal_clipped', 'mu': wI1I1, 'sigma': 1., 'low': 0.0001,
	# 	                    'high': 100*wI1I1},
	# 	           'I2I1': {'distribution': 'lognormal_clipped', 'mu': wI2I1, 'sigma': 1., 'low': 0.0001,
	# 	                    'high': 100*wI2I1},
	# 	           'I1I2': {'distribution': 'lognormal_clipped', 'mu': wI1I2, 'sigma': 1., 'low': 0.0001,
	# 	                    'high': 100*wI1I2},
	# 	           'I2I2': {'distribution': 'lognormal_clipped', 'mu': wI2I2, 'sigma': 1., 'low': 0.0001,
	# 	                    'high': 100*wI2I2},}
	# else:
	# 	weights = {'EE': wEE, 'I1E': wI1E, 'I2E': wI2E,
	# 	           'EI1': wEI1, 'EI2': wEI2, 'I1I1': wI1I1,
	# 	           'I2I1': wI2I1, 'I1I2': wI1I2, 'I2I2': wI2I2,}

	# devices
	multimeter = rec_device_defaults(device_type='multimeter')
	multimeter.update({'record_from': ['V_m', 'I_ex', 'I_in', 'G_syn_tot'], 'record_n': 1})
	sd = rec_device_defaults(device_type='spike_detector')

	# recurrent synapses
	# receptor order = (1) AMPA, (2) GABAa, (3) NMDA, (4) GABAb
	synapse_keys = ['EE', 'I1E', 'I2E', 'EI1', 'EI2', 'I1I1', 'I2I1', 'I1I2', 'I2I2']
	recurrent_synapses = dict(
	connected_populations=[('E', 'E_inputs'), # EE
						   ('I1', 'E_inputs'), ('I2', 'E_inputs'),  # EI
						   ('E', 'I1_inputs'), ('E', 'I2_inputs'), # IE
						   ('I1', 'I1_inputs'), ('I2', 'I1_inputs'),
						   ('I1', 'I2_inputs'), ('I2', 'I2_inputs'),],  # II
	synapse_names = ['EE', 'I1E', 'I2E', 'EI1', 'EI2', 'I1I1', 'I2I1', 'I1I2', 'I2I2'],
	synapse_models = ['multiport_synapse' for _ in range(len(synapse_keys))],
	synapse_model_parameters = [{'receptor_types': [1., 3.]},
	                            {'receptor_types': [1., 3.]}, {'receptor_types': [1., 3.]},
	                            {'receptor_types': [2., 4.]}, {'receptor_types': [2., 4.]},
	                            {'receptor_types': [2., 4.]}, {'receptor_types': [2., 4.]},
	                            {'receptor_types': [2., 4.]}, {'receptor_types': [2., 4.]},],
	syn_specs = [{'receptor_type': 1}, # this is still needed due to initial tests..
	             {'receptor_type': 1}, {'receptor_type': 1},
	             {'receptor_type': 1}, {'receptor_type': 1},
	             {'receptor_type': 1}, {'receptor_type': 1},
	             {'receptor_type': 1}, {'receptor_type': 1}],
	pre_computedW = [None for _ in range(len(synapse_keys))],
	weights = [weights[k] for k in synapse_keys],
	delays = [delays[k] for k in synapse_keys],
	conn_specs =[{'rule': 'pairwise_bernoulli', 'p': epsilon[k]} for k in synapse_keys],
	devices = [multimeter, sd])
	##############################################################

	neuron_pars, net_pars, connection_pars = set_network_defaults(default_set=2, neuron_set=neuron_set, connection_set=1,
	                                                              N=N, kernel_pars=kernel_pars,
	                                                              random_parameters=randomized_pars, **recurrent_synapses)
	# ######################################################################################################################
	# Encoding Parameters
	# ######################################################################################################################
	# nu_x = 5.
	k_x = epsilon['EE'] * nE

	encoding_pars = set_encoding_defaults(default_set=0)

	background_noise = dict(
		start=0., stop=sys.float_info.max, origin=0.,
		generator_label='X_Noise',
		rate=nu_x*k_x, target_population_names=['X_inputs'],
		additional_parameters={
			'syn_specs': {},
			'models': 'static_synapse',
			'model_pars': {},
			'weight_dist': 1.,
			'delay_dist': 0.1})
	add_background_noise(encoding_pars, background_noise)

	encoding_pars.encoder.update({
	'N': 1,
	  'analog_device_pars': [{}],
	  'labels': ['X_inputs'],
	  'model_pars': [{}],
	  'models': ['parrot_neuron'],
	  'n_neurons': [1],
	  'neuron_pars': [{'model': 'parrot_neuron'}],
	  'record_analogs': [False],
	  'record_spikes': [False],
	  'spike_device_pars': [{}],
	  'topology': [False],
	  'topology_dict': [None]
	})
	encoding_pars.connectivity.connections.extend([('E', 'X_inputs'), ('I1', 'X_inputs'), ('I2', 'X_inputs')])
	encoding_pars.connectivity.conn_specs.extend([{'rule': 'all_to_all'}, {'rule': 'all_to_all'}, {'rule': 'all_to_all'}])
	encoding_pars.connectivity.models.extend(['multiport_synapse', 'multiport_synapse', 'multiport_synapse'])
	encoding_pars.connectivity.model_pars.extend([{'receptor_types': [1., 3.]}, {'receptor_types': [1., 3.]}, {'receptor_types': [1., 3.]}])
	encoding_pars.connectivity.preset_W.extend([None, None, None])
	encoding_pars.connectivity.syn_specs.extend([{'receptor_type': 1}, {'receptor_type': 1}, {'receptor_type': 1}])
	encoding_pars.connectivity.synapse_name.extend(['XinE', 'XinI1', 'XinI2'])
	encoding_pars.connectivity.topology_dependent.extend([False, False, False])
	encoding_pars.connectivity.weight_dist.extend([weights['EE'], weights['I1E'], weights['I2E']])
	encoding_pars.connectivity.delay_dist.extend([delays['EE'], delays['I1E'], delays['I2E']])

	background_noise = dict(
		start=0., stop=sys.float_info.max, origin=0.,
		generator_label='E_Noise',
		rate=nu_x, target_population_names=['E_inputs'],
		additional_parameters={
			'syn_specs': {},
			'models': 'static_synapse',
			'model_pars': {},
			'weight_dist': 1.,
			'delay_dist': 0.1})
	add_background_noise(encoding_pars, background_noise)

	background_noise = dict(
		start=0., stop=sys.float_info.max, origin=0.,
		generator_label='I1_Noise',
		rate=nu_x, target_population_names=['I1_inputs'],
		additional_parameters={
			'syn_specs': {},
			'models': 'static_synapse',
			'model_pars': {},
			'weight_dist': 1.,
			'delay_dist': 0.1})
	add_background_noise(encoding_pars, background_noise)

	background_noise = dict(
		start=0., stop=sys.float_info.max, origin=0.,
		generator_label='I2_Noise',
		rate=nu_x, target_population_names=['I2_inputs'],
		additional_parameters={
			'syn_specs': {},
			'models': 'static_synapse',
			'model_pars': {},
			'weight_dist': 1.,
			'delay_dist': 0.1})
	add_background_noise(encoding_pars, background_noise)

	# ##################################################################################################################
	# RETURN dictionary of Parameters dictionaries
	# ==================================================================================================================
	return dict([('kernel_pars', kernel_pars),
	             ('neuron_pars', neuron_pars),
	             ('net_pars', net_pars),
	             ('connection_pars', connection_pars),
	             ('encoding_pars', encoding_pars)])