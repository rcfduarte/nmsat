__author__ = 'duarte'
import sys
from preset import *
import numpy as np

"""
spike_noise_input
- standard network setup, driven with noisy, Poissonian input
- quantify and set population state
- run with noise_driven_dynamics in computations
- debug with noise_driven_dynamics script
"""

run = 'Blaustein'
data_label = 'ED_spike_noise_input'


def build_parameters(g, nu_x):
	# ##################################################################################################################
	# System / Kernel Parameters
	# ##################################################################################################################
	system = dict(
		nodes=1,
		ppn=16,
		mem=32,
		walltime='01-00:00:00',
		queue='defqueue',
		transient_time=1000.,
		sim_time=10000.)

	kernel_pars = set_kernel_defaults(run_type=run, data_label=data_label, **system)
	np.random.seed(kernel_pars['np_seed'])
	# ##################################################################################################################
	# Neuron, Synapse and Network Parameters
	# ##################################################################################################################
	N = 10000
	nE = 0.8 * N
	nI = 0.2 * N
	dE = 1.0
	dI = 0.8

	# Connection probabilities
	pEE = 0.1
	pEI = 0.2
	pIE = 0.1
	pII = 0.2

	# connection weights
	# g = 13.5
	wE = 1.2
	wI = -g * wE

	recurrent_synapses = dict(
		connected_populations=[('E', 'E'), ('E', 'I'), ('I', 'E'), ('I', 'I')],
		synapse_models=['static_synapse', 'static_synapse', 'static_synapse', 'static_synapse'],
		synapse_model_parameters=[{}, {}, {}, {}],
		pre_computedW=[None, None, None, None],
		weights=[{'distribution': 'normal_clipped', 'mu': wE, 'sigma': 0.5 * wE, 'low': 0.0001, 'high': 10. * wE},
		         {'distribution': 'normal_clipped', 'mu': wI, 'sigma': np.abs(0.5 * wI), 'low': 10. * wI, 'high': 0.0001},
		         {'distribution': 'normal_clipped', 'mu': wE, 'sigma': 0.5 * wE, 'low': 0.0001, 'high': 10. * wE},
		         {'distribution': 'normal_clipped', 'mu': wI, 'sigma': np.abs(0.5 * wI), 'low': 10. * wI, 'high': 0.0001}],
		delays=[{'distribution': 'normal_clipped', 'mu': dE, 'sigma': 0.5 * dE, 'low': 0.1, 'high': 10. * dE},
		        {'distribution': 'normal_clipped', 'mu': dI, 'sigma': 0.5 * dI, 'low': 0.1, 'high': 10. * dI},
		        {'distribution': 'normal_clipped', 'mu': dE, 'sigma': 0.5 * dE, 'low': 0.1, 'high': 10. * dE},
		        {'distribution': 'normal_clipped', 'mu': dI, 'sigma': 0.5 * dI, 'low': 0.1, 'high': 10. * dI}],
		conn_specs=[{'rule': 'pairwise_bernoulli', 'p': pEE},
		            {'rule': 'pairwise_bernoulli', 'p': pEI},
		            {'rule': 'pairwise_bernoulli', 'p': pIE},
		            {'rule': 'pairwise_bernoulli', 'p': pII}],
		syn_specs=[{}, {}, {}, {}])
	neuron_pars, net_pars, connection_pars = set_network_defaults(N=N, **recurrent_synapses)

	net_pars['record_analogs'] = [True, False]
	multimeter = rec_device_defaults(device_type='multimeter')
	multimeter.update({'record_from': ['V_m', 'g_ex', 'g_in'], 'record_n': 100})
	net_pars['analog_device_pars'] = [copy_dict(multimeter, {'label': ''}), {}]
	# ######################################################################################################################
	# Encoding Parameters
	# ######################################################################################################################
	# nu_x = 20.
	k_x = pEE * nE
	w_in = 1.

	encoding_pars = set_encoding_defaults(default_set=0)

	background_noise = dict(
		start=0., stop=sys.float_info.max, origin=0.,
		rate=nu_x*k_x, target_population_names=['E', 'I'],
		additional_parameters={
			'syn_specs': {},
			'models': 'static_synapse',
			'model_pars': {},
			'weight_dist': {'distribution': 'normal_clipped', 'mu': w_in, 'sigma': 0.5*w_in, 'low': 0.0001,
			                'high': 10.*w_in},
			'delay_dist': 0.1})
	add_background_noise(encoding_pars, background_noise)

	# nu_x = 12.
	# k_x = pEE * nE
	# w_in = 1.
	#
	# background_noise2 = dict(
	# 	start=0., stop=sys.float_info.max, origin=0.,
	# 	rate=nu_x*k_x, target_population_names=['E', 'I'],
	# 	generator_label='background',
	# 	additional_parameters={
	# 		'syn_specs': {},
	# 		'models': 'static_synapse',
	# 		'model_pars': {},
	# 		'weight_dist': {'distribution': 'normal_clipped', 'mu': w_in, 'sigma': 0.5*w_in, 'low': 0.0001,
	# 		                'high': 10.*w_in},
	# 		'delay_dist': 0.1})
	# add_background_noise(encoding_pars, background_noise2)

	# ##################################################################################################################
	# Extra analysis parameters (specific for this experiment)
	# ==================================================================================================================
	analysis_pars = {'time_bin': 1.,        # bin width for spike counts, fano factors and correlation coefficients
	                 'n_pairs': 500,        # number of spike train pairs to consider in correlation coefficient
	                 'tau': 20.,            # time constant of exponential filter (van Rossum distance)
	                 'window_len': 100,     # length of sliding time window (for time_resolved analysis)
	                 'summary_only': True, # how to save the data (only mean and std - True) or entire data set (False)
	                 'complete': True,      # use all existing measures or just the fastest / simplest ones
	                 'time_resolved': False}# perform time-resolved analysis

	# ##################################################################################################################
	# RETURN dictionary of Parameters dictionaries
	# ==================================================================================================================
	return dict([('kernel_pars', kernel_pars),
	             ('neuron_pars', neuron_pars),
	             ('net_pars', net_pars),
	             ('encoding_pars', encoding_pars),
	             ('connection_pars', connection_pars),
	             ('analysis_pars', analysis_pars)])

# ######################################################################################################################
# PARAMETER RANGE declarations
# ======================================================================================================================
parameter_range = {
	'g': np.arange(4., 25.5, 0.2),
	'nu_x': np.arange(2, 21, 1)
}