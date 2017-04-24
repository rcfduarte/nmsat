import sys
sys.path.insert(0, "../")
from preset import *
import numpy as np
from auxiliary_fcns import determine_lognormal_parameters

"""
noise_driven_dynamics
- run with noisedriven_dynamics in computations
- debug with script noise_driven
"""
__author__ = 'duarte'

run = 'local'
data_label = 'TS0_noisedrivendynamics_test'
project_label = 'Timescales'
heterogeneity = {'neuron': True, 'synapse': True, 'structural': False}

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
		walltime='01-00:00:00',
		queue='defqueue',
		transient_time=1000.,
		sim_time=2000.)

	kernel_pars = set_kernel_defaults(run_type=run, data_label=data_label, **system)

	# ######################################################################################################################
	# Neuron, Synapse and Network Parameters
	# ######################################################################################################################
	randomized_pars = randomize_neuron_parameters(heterogeneity['neuron'])
	if heterogeneity['neuron']:
		neuron_set = 1.2
	else:
		neuron_set = 1.1

	# Population sizes
	N = 1250
	nE = 0.8 * N
	nI = 0.2 * N
	nI1 = 0.35 * nI
	nI2 = 0.65 * nI

	# connection densities
	epsilon = {'EE': 0.168, 'I1E': 0.575, 'I2E': 0.242,
	           'EI1': 0.6, 'EI2': 0.465, 'I1I1': 0.55,
	           'I2I1': 0.241, 'I1I2': 0.379, 'I2I2': 0.381,}
	# connection delays
	mean_delays = {'EE': 1.8, 'I1E': 1.2, 'I2E': 1.5,
	          'EI1': 0.8, 'EI2': 1.5, 'I1I1': 1.,
	          'I2I1': 1.2, 'I1I2': 1.5, 'I2I2': 1.5, }
	if heterogeneity['synapse']:
		# delays = {}
		# for k, v in mean_delays.items():
		# 	ln_pars = determine_lognormal_parameters(v, 2.*v, median=v)
		# 	delays.update({k: {'distribution': 'lognormal_clipped', 'mu': ln_pars[0], 'sigma': ln_pars[1], 'low': 0.1,
		# 	                  'high': 100.*v}})
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
		delays = {k: v for k, v in mean_delays.items()}
	# connection weights
	gamma_E = 3.
	gamma_I1 = 1.
	gamma_I2 = 1.

	wEE = 0.45
	wEI1 = gamma_E * wEE #5.148#
	wEI2 = gamma_E * wEE #4.85#
	wI1E = 1.65
	wI1I1 = gamma_I1 * wI1E #2.22 #
	wI1I2 = gamma_I1 * wI1E #1.47 #
	wI2E = 0.638
	wI2I1 = gamma_I2 * wI2E #1.4 #
	wI2I2 = gamma_I2 * wI2E #0.83 #

	if heterogeneity['synapse']:
		# weights = {}
		# for k in epsilon.keys():
		# 	w_var = locals()['w{0}'.format(k)]
		# 	ln_pars = determine_lognormal_parameters(w_var, 4*w_var, median=w_var)
		# 	weights.update({k: {'distribution': 'lognormal_clipped', 'mu': ln_pars[0], 'sigma': ln_pars[1],
		# 	                    'low': 0.00001, 'high': 100*w_var}})
		weights = {'EE': {'distribution': 'lognormal_clipped', 'mu': wEE, 'sigma': 0.5*wEE, 'low': 0.0001, 'high': 100*wEE},
		           'I1E': {'distribution': 'lognormal_clipped', 'mu': wI1E, 'sigma': 0.5*wI1E, 'low': 0.0001,
		                   'high': 100*wI1E},
		           'I2E': {'distribution': 'lognormal_clipped', 'mu': wI2E, 'sigma': 0.5*wI2E, 'low': 0.0001,
		                   'high': 100*wI2E},
		           'EI1': {'distribution': 'lognormal_clipped', 'mu': wEI1, 'sigma': 0.5*wEI1, 'low': 0.0001,
		                   'high': 100*wEI1},
		           'EI2': {'distribution': 'lognormal_clipped', 'mu': wEI2, 'sigma': 0.5*wEI2, 'low': 0.0001,
		                   'high': 100*wEI2},
		           'I1I1': {'distribution': 'lognormal_clipped', 'mu': wI1I1, 'sigma': 0.5*wI1I1, 'low': 0.0001,
		                    'high': 100*wI1I1},
		           'I2I1': {'distribution': 'lognormal_clipped', 'mu': wI2I1, 'sigma': 0.5*wI2I1, 'low': 0.0001,
		                    'high': 100*wI2I1},
		           'I1I2': {'distribution': 'lognormal_clipped', 'mu': wI1I2, 'sigma': 0.5*wI1I2, 'low': 0.0001,
		                    'high': 100*wI1I2},
		           'I2I2': {'distribution': 'lognormal_clipped', 'mu': wI2I2, 'sigma': 0.5*wI2I2, 'low': 0.0001,
		                    'high': 100*wI2I2},}
	else:
		weights = {'EE': wEE, 'I1E': wI1E, 'I2E': wI2E,
		           'EI1': wEI1, 'EI2': wEI2, 'I1I1': wI1I1,
		           'I2I1': wI2I1, 'I1I2': wI1I2, 'I2I2': wI2I2,}
	# if heterogeneity['synapse']:
	# 	weights = {'EE': {'distribution': 'lognormal_clipped', 'mu': 0.45, 'sigma': 1., 'low': 0.0001, 'high': 50.},
	# 	           'I1E': {'distribution': 'lognormal_clipped', 'mu': 1.65, 'sigma': 1., 'low': 0.0001, 'high': 150.},
	# 	           'I2E': {'distribution': 'lognormal_clipped', 'mu': 0.638, 'sigma': 1., 'low': 0.0001, 'high': 60.},
	# 	           'EI1': {'distribution': 'lognormal_clipped', 'mu': 5.148, 'sigma': 1., 'low': 0.0001, 'high': 500.},
	# 	           'EI2': {'distribution': 'lognormal_clipped', 'mu': 4.85, 'sigma': 1., 'low': 0.0001, 'high': 500.},
	# 	           'I1I1': {'distribution': 'lognormal_clipped', 'mu': 2.22, 'sigma': 1., 'low': 0.0001,
	# 	                    'high': 250.},
	# 	           'I2I1': {'distribution': 'lognormal_clipped', 'mu': 1.4, 'sigma': 1., 'low': 0.0001,
	# 	                    'high': 150.},
	# 	           'I1I2': {'distribution': 'lognormal_clipped', 'mu': 1.47, 'sigma': 1., 'low': 0.0001,
	# 	                    'high': 150.},
	# 	           'I2I2': {'distribution': 'lognormal_clipped', 'mu': 0.83, 'sigma': 1., 'low': 0.0001,
	# 	                    'high': 100.},}
	# else:
	# 	weights = {'EE': 0.45, 'I1E': 1.65, 'I2E': 0.638,
	# 	           'EI1': 5.148, 'EI2': 4.85, 'I1I1': 2.22,
	# 	           'I2I1': 1.4, 'I1I2': 1.47, 'I2I2': 0.83,}

	# recurrent synapses
	# receptor order = (1) AMPA, (2) GABAa, (3) NMDA, (4) GABAb
	synapse_keys = ['EE', 'I1E', 'I2E', 'EI1', 'EI2', 'I1I1', 'I2I1', 'I1I2', 'I2I2']
	recurrent_synapses = dict(
	connected_populations=[('E', 'E'), # EE
						   ('I1', 'E'), ('I2', 'E'),  # EI
						   ('E', 'I1'), ('E', 'I2'), # IE
						   ('I1', 'I1'), ('I2', 'I1'),
						   ('I1', 'I2'), ('I2', 'I2'),],  # II
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
	conn_specs =[{'rule': 'pairwise_bernoulli', 'p': epsilon[k]} for k in synapse_keys])
	##############################################################

	neuron_pars, net_pars, connection_pars = set_network_defaults(default_set=3, neuron_set=neuron_set, connection_set=1,
	                                                              N=N, kernel_pars=kernel_pars,
	                                                              random_parameters=randomized_pars, **recurrent_synapses)
	net_pars['record_analogs'] = [True, False, False]
	multimeter = rec_device_defaults(device_type='multimeter')
	multimeter.update({'record_from': ['V_m', 'I_ex', 'I_in'], 'record_n': 1})
	net_pars['analog_device_pars'] = [copy_dict(multimeter, {'label': 'E_analogs'}),
	                                  {}, {}]

	# ######################################################################################################################
	# Encoding Parameters
	# ######################################################################################################################
	nu_x = 5.
	k_x = 1000. #epsilon['EE'] * nE
	# w_in = weights['EE']
	# d_in = delays['EE']

	encoding_pars = set_encoding_defaults(default_set=0)

	background_noise = dict(
		start=0., stop=sys.float_info.max, origin=0.,
		generator_label='X_Noise',
		rate=nu_x*k_x, target_population_names=['E_inputs', 'I1_inputs', 'I2_inputs'],
		additional_parameters={
			'syn_specs': {},
			'models': 'static_synapse',
			'model_pars': {},
			'weight_dist': 1.,
			'delay_dist': 0.1})
	add_background_noise(encoding_pars, background_noise)

	encoding_pars.encoder.update({
	'N': 3,
	  'analog_device_pars': [{}],
	  'labels': ['E_inputs', 'I1_inputs', 'I2_inputs'],
	  'model_pars': [{}, {}, {}],
	  'models': ['parrot_neuron', 'parrot_neuron', 'parrot_neuron'],
	  'n_neurons': [nE, nI1, nI2],
	  'neuron_pars': [{'model': 'parrot_neuron'}, {'model': 'parrot_neuron'}, {'model': 'parrot_neuron'}],
	  'record_analogs': [False, False, False],
	  'record_spikes': [False, False, False],
	  'spike_device_pars': [{}, {}, {}],
	  'topology': [False, False, False],
	  'topology_dict': [None, None, None]
	})
	encoding_pars.connectivity.connections.extend([('E', 'E_inputs'), ('I1', 'I1_inputs'), ('I2', 'I2_inputs')])
	encoding_pars.connectivity.conn_specs.extend([{'rule': 'one_to_one'}, {'rule': 'one_to_one'}, {'rule': 'one_to_one'}])
	encoding_pars.connectivity.models.extend(['multiport_synapse', 'multiport_synapse', 'multiport_synapse'])
	encoding_pars.connectivity.model_pars.extend([{'receptor_types': [1., 3.]}, {'receptor_types': [1., 3.]}, {'receptor_types': [1., 3.]}])
	encoding_pars.connectivity.preset_W.extend([None, None, None])
	encoding_pars.connectivity.syn_specs.extend([{'receptor_type': 1}, {'receptor_type': 1}, {'receptor_type': 1}])
	encoding_pars.connectivity.synapse_name.extend(['XinE', 'XinI1', 'XinI2'])
	encoding_pars.connectivity.topology_dependent.extend([False, False, False])
	encoding_pars.connectivity.weight_dist.extend([weights['EE'], weights['I1E'], weights['I2E']])
	encoding_pars.connectivity.delay_dist.extend([delays['EE'], delays['I1E'], delays['I2E']])

	# ##################################################################################################################
	# Extra analysis parameters (specific for this experiment)
	# ==================================================================================================================
	analysis_pars = {
		# analysis depth
		'depth': 1,  	# 1: save only summary of data, use only fastest measures
						# 2: save all data, use only fastest measures
						# 3: save only summary of data, use all available measures
						# 4: save all data, use all available measures

		'store_activity': False,  	# [int] - store all population activity in the last n steps of the test
									# phase; if set True the entire test phase will be stored;

		'population_activity': {
			'time_bin': 1.,  		# bin width for spike counts, fano factors and correlation coefficients
			'n_pairs': 500,  		# number of spike train pairs to consider in correlation coefficient
			'tau': 20.,  			# time constant of exponential filter (van Rossum distance)
			'window_len': 100,  	# length of sliding time window (for time_resolved analysis)
			'time_resolved': False, # perform time-resolved analysis
		}
	}
	# ##################################################################################################################
	# RETURN dictionary of Parameters dictionaries
	# ==================================================================================================================
	return dict([('kernel_pars', kernel_pars),
	             ('neuron_pars', neuron_pars),
	             ('net_pars', net_pars),
	             ('encoding_pars', encoding_pars),
	             ('connection_pars', connection_pars),
	             ('analysis_pars', analysis_pars)])