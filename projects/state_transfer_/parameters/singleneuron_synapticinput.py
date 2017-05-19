__author__ = 'duarte'
from preset import *
from defaults.paths import paths
from modules.parameters import ParameterSet
import numpy as np

"""
single_neuron_synaptic_input
- simulate a single, point neuron, driven by Poisson input (external and recurrent)
- run with single_neuron in computations
- debug with single_neuron script
"""

run = 'local'
data_label = 'amat2_test_one_pool'


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
		transient_time=1000.,
		sim_time=5000.)

	kernel_pars = set_kernel_defaults(run_type=run, data_label=data_label, **system)


	# ##################################################################################################################
	# Neuron, Synapse and Network Parameters
	# ##################################################################################################################
	neuron_pars = {
		'E': {
			'model': 'amat2_psc_exp',
			'alpha_1': -0.5,
			'alpha_2': 0.35,
			'beta': -0.3,
			'omega': -65.,
			'I_e': 91.,
			'tau_m': 10.,
			'tau_1': 10.,
			'tau_2': 200.,
			't_ref': 2.,
			'C_m': 200., },
		'I': {
			'model': 'amat2_psc_exp',
			'alpha_1': -0.5,
			'alpha_2': 0.35,
			'beta': -0.3,
			'omega': -65.,
			'I_e': 91.,
			'tau_m': 10.,
			'tau_1': 10.,
			'tau_2': 200.,
			't_ref': 2.,
			'C_m': 200., }
	}

	N = 1250
	delay = 1.5
	epsilon = 0.1

	gamma = 8.
	wE = 20.
	wI = -gamma * wE

	# devices
	multimeter = rec_device_defaults(device_type='multimeter')
	multimeter.update({'record_from': ['V_m', 'V_th'], 'record_n': 1})
	sd = rec_device_defaults(device_type='spike_detector')

	recurrent_synapses = dict(
		connected_populations=[('E', 'E_inputs'), ('E', 'I_inputs'), ('I', 'E_inputs'), ('I', 'I_inputs')],
		synapse_models=['static_synapse', 'static_synapse', 'static_synapse', 'static_synapse'],
		synapse_model_parameters=[{}, {}, {}, {}],
		pre_computedW=[None, None, None, None],
		weights=[wE, wI, wE, wI],
		delays=[delay, delay, delay, delay],
		conn_specs=[{'autapses': False, 'multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon},
		            {'autapses': False, 'multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon},
		            {'autapses': False, 'multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon},
		            {'autapses': False, 'multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon}],
		syn_specs=[{}, {}, {}, {}],
		devices = [multimeter, sd])

	##############################################################
	neuron_pars, net_pars, connection_pars = set_network_defaults(default_set=3, neuron_set=neuron_pars,
	                                                              connection_set=1,
	                                                              N=N, #kernel_pars=kernel_pars,
	                                                              **recurrent_synapses)

	# ######################################################################################################################
	# Encoding Parameters
	# ######################################################################################################################
	nu_x = 5.
	k_x = epsilon * (0.8 * N)

	encoding_pars = set_encoding_defaults(default_set=0)

	background_noise = dict(
		start=0., stop=sys.float_info.max, origin=0.,
		generator_label='X_Noise',
		rate=nu_x * k_x, target_population_names=['E', 'I'],
		additional_parameters={
			'syn_specs': {},
			'models': 'static_synapse',
			'model_pars': {},
			'weight_dist': wE,
			'delay_dist': 0.1}) #delay}) - to avoid numerical errors
	add_background_noise(encoding_pars, background_noise)

	background_noise = dict(
		start=0., stop=sys.float_info.max, origin=0.,
		generator_label='E_Noise',
		rate=nu_x, target_population_names=['E_inputs'],
		additional_parameters={
			'syn_specs': {},
			'models': 'static_synapse',
			'model_pars': {},
			'weight_dist': wE,
			'delay_dist': delay})
	add_background_noise(encoding_pars, background_noise)
	background_noise = dict(
		start=0., stop=sys.float_info.max, origin=0.,
		generator_label='I_Noise',
		rate=nu_x, target_population_names=['I_inputs'],
		additional_parameters={
			'syn_specs': {},
			'models': 'static_synapse',
			'model_pars': {},
			'weight_dist': wI,
			'delay_dist': delay})
	add_background_noise(encoding_pars, background_noise)

	# ##################################################################################################################
	# RETURN dictionary of Parameters dictionaries
	# ==================================================================================================================
	return dict([('kernel_pars', kernel_pars),
	             ('neuron_pars', neuron_pars),
	             ('net_pars', net_pars),
	             ('connection_pars', connection_pars),
	             ('encoding_pars', encoding_pars)])