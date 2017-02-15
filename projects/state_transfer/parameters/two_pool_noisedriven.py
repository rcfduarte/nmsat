__author__ = 'duarte'
import sys
from preset import *
import numpy as np

"""
two_pool_noisedriven
- 2 pool network setup (2 BRN), driven with noisy, Poissonian input
- quantify and set population state
- run with noise_driven_dynamics in computations
- debug with noise_driven_dynamics script
"""

run = 'Blaustein'
data_label = 'state_transfer_twopool_noisedriven'


# ######################################################################################################################
# PARAMETER RANGE declarations
# ======================================================================================================================
parameter_range = {
	'nu_x':		[20.],
	'gamma': 	[8., 14.]
}


def build_parameters(nu_x, gamma):
	# ##################################################################################################################
	# System / Kernel Parameters
	# ##################################################################################################################
	system = dict(
		nodes=1,
		ppn=16,
		mem=32,
		walltime='00-20:00:00',
		queue='defqueue',
		transient_time=1000.,
		sim_time=2000.)

	kernel_pars = set_kernel_defaults(run_type=run, data_label=data_label, **system)

	# ##################################################################################################################
	# Neuron, Synapse and Network Parameters
	# ##################################################################################################################
	N = 12500
	nE = 0.8 * N
	delay = 1.5
	epsilon = 0.1

	# gamma = 8.
	wE = 20.
	wI = -gamma * wE

	recurrent_synapses = dict(
		connected_populations=[('E1', 'E1'), ('E2', 'E1'), ('E2', 'E2'), ('E1', 'E2'),
		                       ('I1', 'I1'), ('E1', 'I1'), ('I1', 'E1'), ('I2', 'I2'),
		                       ('E2', 'I2'), ('I2', 'E2')],
		synapse_models=['static_synapse', 'static_synapse', 'static_synapse', 'static_synapse',
		                'static_synapse', 'static_synapse', 'static_synapse', 'static_synapse',
		                'static_synapse', 'static_synapse'],
		synapse_model_parameters=[{}, {}, {}, {}, {}, {}, {}, {}, {}, {}],
		pre_computedW=[None, None, None, None, None, None, None, None, None, None],
		weights=[wE, wE, wE, wE, wI, wI, wE, wI, wI, wE],
		delays=[delay, delay, delay, delay, delay, delay, delay, delay, delay, delay],
		conn_specs=[{'autapses': False, 'multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon/2.},
		            {'autapses': False, 'multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon/2.},
		            {'autapses': False, 'multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon/2.},
		            {'autapses': False, 'multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon/2.},
		            {'autapses': False, 'multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon},
		            {'autapses': False, 'multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon},
		            {'autapses': False, 'multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon},
		            {'autapses': False, 'multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon},
		            {'autapses': False, 'multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon},
		            {'autapses': False, 'multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon}],
		syn_specs=[{}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
	)
	neuron_pars, net_pars, connection_pars = set_network_defaults(default_set=2, neuron_set=2, N=N,
	                                                              **recurrent_synapses)

	net_pars['record_analogs'] = [True, False, False, False]
	multimeter = rec_device_defaults(device_type='multimeter')
	multimeter.update({'record_from': ['V_m', 'V_th'], 'record_n': 1})
	net_pars['analog_device_pars'] = [copy_dict(multimeter, {'label': ''}), {}, {}, {}]

	# ##################################################################################################################
	# Encoding Parameters
	# ##################################################################################################################
	# nu_x = 20.
	k_x = epsilon * nE
	# w_in = 90.

	encoding_pars = set_encoding_defaults(default_set=0)

	background_noise = dict(
		start=0., stop=sys.float_info.max, origin=0.,
		rate=nu_x * k_x, target_population_names=['E1', 'I1', 'E2', 'I2'],
		additional_parameters={
			'syn_specs': {},
			'models': 'static_synapse',
			'model_pars': {},
			'weight_dist': wE,
			'delay_dist': delay})
	add_background_noise(encoding_pars, background_noise)

	# ##################################################################################################################
	# RETURN dictionary of Parameters dictionaries
	# ==================================================================================================================
	return dict([('kernel_pars', kernel_pars),
	             ('neuron_pars', neuron_pars),
	             ('net_pars', net_pars),
	             ('encoding_pars', encoding_pars),
	             ('connection_pars', connection_pars)])