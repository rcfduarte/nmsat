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

run = 'local'
data_label = 'ST_twopool_noisedriven_ud_metrics_75_155'


# ######################################################################################################################
# PARAMETER RANGE declarations
# ======================================================================================================================
parameter_range = {
	# 'nu_x': np.arange(2, 12.1, .5),
	# 'gamma': np.arange(9, 17.1, .5)
}


# def build_parameters(nu_x, gamma):
def build_parameters():
	gamma 	= 15.5
	nu_x 	= 7.5
	# ##################################################################################################################
	# System / Kernel Parameters
	# ##################################################################################################################
	system = dict(
		nodes=1,
		ppn=24,
		mem=64000,
		walltime='00-20:00:00',
		queue='defqueue',
		transient_time=1000.,
		sim_time=1500.)

	kernel_pars = set_kernel_defaults(run_type=run, data_label=data_label, **system)

	# ##################################################################################################################
	# Neuron, Synapse and Network Parameters
	# ##################################################################################################################
	N = 10000
	nE = 0.8 * N
	delay = 1.5
	epsilon = 0.1

	wE = 1.
	wI = -gamma * wE

	syn_pars_dict = dict(
		connected_populations=[('E1', 'E1'), ('E2', 'E1'), ('I1', 'E1'), ('I2', 'E1'),
		                       ('E1', 'I1'), ('E2', 'I1'), ('I1', 'I1'), ('I2', 'I1'),
		                       ('E1', 'E2'), ('E2', 'E2'), ('I1', 'E2'), ('I2', 'E2'),
		                       ('E1', 'I2'), ('E2', 'I2'), ('I1', 'I2'), ('I2', 'I2')],
		synapse_models=['static_synapse', 'static_synapse', 'static_synapse', 'static_synapse',
		                'static_synapse', 'static_synapse', 'static_synapse', 'static_synapse',
		                'static_synapse', 'static_synapse', 'static_synapse', 'static_synapse',
		                'static_synapse', 'static_synapse', 'static_synapse', 'static_synapse', ],
		synapse_model_parameters=[{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}],
		pre_computedW=[None, None, None, None,
		               None, None, None, None, None, None, None, None, None, None, None, None],
		weights=[wE, wE, wE, wE,
		         wI, wI, wI, wI,
		         wE, wE, wE, wE,
		         wI, wI, wI, wI],
		delays=[delay, delay, delay, delay,
		        delay, delay, delay, delay,
		        delay, delay, delay, delay,
		        delay, delay, delay, delay],
		conn_specs=[{'autapses': False, 'multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon},  # E1<-E1
		            {'autapses': False, 'multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon},  # E2<-E1
		            {'autapses': False, 'multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon},  # I1<-E1
		            {'autapses': False, 'multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon},  # I2<-E1

		            {'autapses': False, 'multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon},  # E1<-I1
		            {'autapses': False, 'multapses': False, 'rule': 'pairwise_bernoulli', 'p': 0.},  # E2<-I1
		            {'autapses': False, 'multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon},  # I1<-I1
		            {'autapses': False, 'multapses': False, 'rule': 'pairwise_bernoulli', 'p': 0.},  # I2<-I1

		            {'autapses': False, 'multapses': False, 'rule': 'pairwise_bernoulli', 'p': 0.},  # E1<-E2
		            {'autapses': False, 'multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon},  # E2<-E2
		            {'autapses': False, 'multapses': False, 'rule': 'pairwise_bernoulli', 'p': 0.},  # I1<-E2
		            {'autapses': False, 'multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon},  # I2<-E2

		            {'autapses': False, 'multapses': False, 'rule': 'pairwise_bernoulli', 'p': 0.},  # E1<-I2
		            {'autapses': False, 'multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon},  # E2<-I2
		            {'autapses': False, 'multapses': False, 'rule': 'pairwise_bernoulli', 'p': 0.},  # I1<-I2
		            {'autapses': False, 'multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon}, ],# I2<-I2
		syn_specs=[{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}])

	neuron_pars, net_pars, connection_pars = set_network_defaults(default_set=2, neuron_set=2, N=N, **syn_pars_dict)

	net_pars['record_analogs'] = [True, True, True, True]
	multimeter = rec_device_defaults(device_type='multimeter')
	multimeter.update({'record_from': ['V_m'], 'record_n': 100})
	net_pars['analog_device_pars'] = [copy_dict(multimeter, {'label': 'mmE1'}),
	                                  copy_dict(multimeter, {'label': 'mmI1'}),
	                                  copy_dict(multimeter, {'label': 'mmE2'}),
	                                  copy_dict(multimeter, {'label': 'mmI2'})]

	# ##################################################################################################################
	# Encoding Parameters
	# ##################################################################################################################
	k_x = epsilon * nE

	encoding_pars = set_encoding_defaults(default_set=0)

	background_noise = dict(
		start=0.,
		stop=sys.float_info.max,
		origin=0.,
		rate=nu_x * k_x,
		target_population_names=['E1', 'I1'],
		additional_parameters={
			'syn_specs': {},
			'models': 'static_synapse',
			'model_pars': {},
			'weight_dist': wE,
			'delay_dist': delay})
	add_background_noise(encoding_pars, background_noise)

	analysis_pars = {
		# analysis depth
		'depth': 3,	# 1: save only summary of data, use only fastest measures
					# 2: save all data, use only fastest measures
					# 3: save only summary of data, use all available measures
					# 4: save all data, use all available measures

		'store_activity': False,	# [int] - store all population activity in the last n steps of the test
									# phase; if set True the entire test phase will be stored;

		'population_activity': {
			'time_bin': 	1.,  	# bin width for spike counts, fano factors and correlation coefficients
			'n_pairs': 		500,  	# number of spike train pairs to consider in correlation coefficient
			'tau': 			20., 	# time constant of exponential filter (van Rossum distance)
			'window_len': 	100, 	# length of sliding time window (for time_resolved analysis)
			'time_resolved': False, # perform time-resolved analysis
		},
		'nu_x': nu_x,
		'gamma': gamma
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
