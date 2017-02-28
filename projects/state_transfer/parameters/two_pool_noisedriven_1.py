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
	'nu_x':		[1200.],
	'gamma': 	[5.]

	# 'nu_x': np.arange(4, 30, 1),
	# 'gamma': np.arange(4., 30, 0.5)

}


def build_parameters(nu_x, gamma):
	# ##################################################################################################################
	# System / Kernel Parameters
	# ##################################################################################################################
	system = dict(
		nodes=1,
		ppn=16,
		mem=64000,
		walltime='00-20:00:00',
		queue='defqueue',
		transient_time=1000.,
		sim_time=2000.)

	kernel_pars = set_kernel_defaults(run_type=run, data_label=data_label, **system)

	# ##################################################################################################################
	# Neuron, Synapse and Network Parameters
	# ##################################################################################################################
	N = 1250
	nE = 0.8 * N
	delay = 1.5
	epsilon = 0.1

	# gamma = 8.
	wE = 20.
	wI = -gamma * wE

	recurrent_synapses = dict(
		connected_populations=[('E1', 'E1'), ('E2', 'E1'), ('I1', 'E1'), ('I2', 'E1'),
							   ('I1', 'I1'), ('E1', 'I1'), ('I2', 'I1'),
							   ('E2', 'E2'), ('E1', 'E2'), ('I2', 'E2'),
							   ('I2', 'I2'), ('E2', 'I2'), ('I1', 'I2'),],
		synapse_models=['static_synapse', 'static_synapse', 'static_synapse', 'static_synapse',
						'static_synapse', 'static_synapse', 'static_synapse',
						'static_synapse', 'static_synapse', 'static_synapse',
		                'static_synapse', 'static_synapse', 'static_synapse'],
		synapse_model_parameters=[{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}],
		pre_computedW=[None, None, None, None, None, None, None, None, None, None, None, None, None],
		weights=[wE, wE, wE, wE,
				 wI, wI, wI,
				 wE, wE, wE,
				 wI, wI, wI],
		delays=[delay, delay, delay, delay, delay, delay, delay, delay, delay, delay, delay, delay, delay],

		# @Renato please take a look at this
		# each neuron in E1 gets, on average, 100(X) + 50(E1) + 50(E2) = 200 ex; 25(I1) + 0(I2)  	 = 25 in
		# each neuron in I1 gets, on average, 100(X) + 100(E1) + 0(E2) = 200 ex; 12.5(I1) + 12.5(I2) = 25 in
		# each neuron in E2 gets, on average, 0(X) + 100(E1) + 100(E2) = 200 ex; 0(I1) + 25(I2)  	 = 25 in
		# each neuron in I2 gets, on average, 100(X) + 50(E1) + 50(E2) = 200 ex; 12.5(I1) + 12.5(I2) = 25 in
		# --- one other possibility would be to have 50 I-I connections instead of 25, so 25 I1<-I1 and 25 I1<-I2, and
		# --- vice versa. Not sure how many I-I connections there is usually..
		conn_specs=[{'autapses': False, 'multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon/2.}, # E1<-E1
		            {'autapses': False, 'multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon},	# E2<-E1
		            {'autapses': False, 'multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon},	# I1<-E1
		            {'autapses': False, 'multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon},	# I2<-E1

		            {'autapses': False, 'multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon/2.},	# I1<-I1
		            {'autapses': False, 'multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon},	# E1<-I1
		            {'autapses': False, 'multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon/2.},	# I2<-I1

					{'autapses': False, 'multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon}, 	# E2<-E2
					{'autapses': False, 'multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon/2.}, # E1<-E2
		            {'autapses': False, 'multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon}, 	# I2<-E2

		            {'autapses': False, 'multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon/2.},	# I2<-I2
		            {'autapses': False, 'multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon},	# E2<-I2
		            {'autapses': False, 'multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon/2.}],# I1<-I2
		syn_specs=[{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
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
	w_in = 1900.

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
		'depth': 4,	# 1: save only summary of data, use only fastest measures
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