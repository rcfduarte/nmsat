__author__ = 'duarte'
import sys
from preset import *
import numpy as np

"""
one_pool_noisedriven
- standard network setup (single BRN), driven with noisy, Poissonian input
- quantify and set population state
- run with noise_driven_dynamics in computations
- debug with noise_driven_dynamics script
"""

run = 'Blaustein'
data_label = 'state_transfer_onepool_noisedriven'

# ######################################################################################################################
# PARAMETER RANGE declarations
# ======================================================================================================================
parameter_range = {
	'nu_x': np.arange(4, 25, 1),
	'gamma': np.arange(4., 30, 0.5)
	# 'nu_x': [15.],
	# 'gamma': [10.]
}


def build_parameters(nu_x, gamma):
	# ##################################################################################################################
	# System / Kernel Parameters
	# ##################################################################################################################
	system = dict(
		nodes=1,
		ppn=16,
		mem=32000,
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
		connected_populations=[('E', 'E'), ('E', 'I'), ('I', 'E'), ('I', 'I')],
		synapse_models=['static_synapse', 'static_synapse', 'static_synapse', 'static_synapse'],
		synapse_model_parameters=[{}, {}, {}, {}],
		pre_computedW=[None, None, None, None],
		weights=[wE, wI, wE, wI],
		delays=[delay, delay, delay, delay],
		conn_specs=[{'autapses': False, 'multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon},
		            {'autapses': False, 'multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon},
		            {'autapses': False, 'multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon},
		            {'autapses': False, 'multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon}],
		syn_specs=[{}, {}, {}, {}]
	)
	neuron_pars, net_pars, connection_pars = set_network_defaults(default_set=1, neuron_set=1, N=N, **recurrent_synapses)

	net_pars['record_analogs'] = [True, False]
	multimeter = rec_device_defaults(device_type='multimeter')
	multimeter.update({'record_from': ['V_m', 'V_th'], 'record_n': 1})
	net_pars['analog_device_pars'] = [copy_dict(multimeter, {'label': ''}), {}]

	# ##################################################################################################################
	# Encoding Parameters
	# ##################################################################################################################
	# nu_x = 20.
	k_x = epsilon * nE
	w_in = 70.

	encoding_pars = set_encoding_defaults(default_set=0)

	background_noise = dict(
		start=0., stop=sys.float_info.max, origin=0.,
		rate=nu_x * k_x, target_population_names=['E', 'I'],
		additional_parameters={
			'syn_specs': {},
			'models': 'static_synapse',
			'model_pars': {},
			'weight_dist': w_in,#wE,
			'delay_dist': delay})
	add_background_noise(encoding_pars, background_noise)

	# ##################################################################################################################
	# Extra analysis parameters (specific for this experiment)
	# ==================================================================================================================
	analysis_pars = {'time_bin': 1.,  # bin width for spike counts, fano factors and correlation coefficients
					 'n_pairs': 500,  # number of spike train pairs to consider in correlation coefficient
					 'tau': 20.,  # time constant of exponential filter (van Rossum distance)
					 'window_len': 100,  # length of sliding time window (for time_resolved analysis)
					 'summary_only': True,  # how to save the data (only mean and std - True) or entire data set (False)
					 'complete': False,  # use all existing measures or just the fastest / simplest ones
					 'time_resolved': False}  # perform time-resolved analysis

	# analysis_pars = {
	# 				# !!! analysis depth, or level, or something else..
	# 				'depth': 1,			# 1: save only summary of data, use only fastest measures
	# 									# 2: save all data, use only fastest measures
	# 									# 3: save only summary of data, use all available measures
	# 									# 4: save all data, use all available measures
	#
	# 				'numerics': {
	# 					'time_bin': 	1.,  	# bin width for spike counts, fano factors and correlation coefficients
	# 					'n_pairs': 		500,  	# number of spike train pairs to consider in correlation coefficient
	# 					'tau': 			20., 	# time constant of exponential filter (van Rossum distance)
	# 					'window_len': 	100}, 	# length of sliding time window (for time_resolved analysis)
	#
	# 				'stats': {
	# 					# other options are possible here...
	# 					'epochs': None,
	# 					'time_resolved': False,  # perform time-resolved analysis
	#
	# 					# !!! for ainess, we could instead enforce more generic options, such as synchrony, regularity,
	# 					# etc., as you suggested in the email.
	# 					'ainess': ['ISI_distance', 'SPIKE_distance', 'ccs_pearson',  # compute level of asynchronous,
	# 							   'cvs', 'cvs_log', 'd_vp', 'd_vr', 'ents', 'ffs']},  # irregular population activity
	#
	# 				'meta': {
	# 					'save_path': 	"",  # if set to None, then we're not saving anything
	# 					'plot': 		True,
	# 					'display': 		True,
	# 					'color_map': 	'jet'}
	# }

	# ##################################################################################################################
	# RETURN dictionary of Parameters dictionaries
	# ==================================================================================================================
	return dict([('kernel_pars', 	 kernel_pars),
	             ('neuron_pars', 	 neuron_pars),
	             ('net_pars', 		 net_pars),
	             ('encoding_pars', 	 encoding_pars),
	             ('connection_pars', connection_pars),
				 ('analysis_pars', 	 analysis_pars)])
