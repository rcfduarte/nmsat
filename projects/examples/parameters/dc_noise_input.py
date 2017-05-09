__author__ = 'duarte'
import sys
from preset import *
import numpy as np

"""
dc_noise_input
- standard network setup, driven with noisy direct current injection
- quantify and set population state
- run with noise_driven_dynamics in computations
- debug with noise_driven_dynamics script
"""

run = 'local'
data_label = 'example_uniformDC'

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
		mem=32000,
		walltime='01-00:00:00',
		queue='defqueue',
		transient_time=1000.,
		sim_time=5000.)

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
	g = 15.
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

	# net_pars['record_analogs'] = [True, True]
	# multimeter = rec_device_defaults(device_type='multimeter')
	# multimeter.update({'record_from': ['V_m'], 'record_n': 1})
	# net_pars['analog_device_pars'] = [copy_dict(multimeter, {'label': ''}), {}]
	# ##################################################################################################################
	# Input Parameters
	# ##################################################################################################################
	ro_in = 1500.
	inp_resolution = 1.

	input_pars = {'noise':
		                {'N': 1,
		                 'noise_source': [np.random.uniform],
		                 'noise_pars': {'low': 0., 'high': 0.5, 'amplitude': ro_in},
		                 # 'amplitude': ro_in,
		                 'rectify': False,
		                 'start_time': 0.,
		                 'stop_time': sys.float_info.max,
		                 'resolution': inp_resolution}
	}
	# ##################################################################################################################
	# Encoding Parameters
	# ##################################################################################################################
	w_in = 1.
	input_synapses = dict(
		target_population_names=['E', 'I'],
		conn_specs=[{'rule': 'pairwise_bernoulli', 'p': pEE}, {'rule': 'pairwise_bernoulli', 'p': pEE}],
		syn_specs=[{}, {}],
		models=['static_synapse', 'static_synapse'],
		model_pars=[{}, {}],
		weight_dist=[{'distribution': 'normal_clipped', 'mu': w_in, 'sigma': 0.5*w_in, 'low': 0.0001, 'high': 10.*w_in},
		             {'distribution': 'normal_clipped', 'mu': w_in, 'sigma': 0.5*w_in, 'low': 0.0001, 'high': 10.*w_in}],
		delay_dist=[0.1, 0.1],
		preset_W=[None, None],
		gen_to_enc_W=None)

	encoding_pars = set_encoding_defaults(default_set=1, input_dimensions=1, n_encoding_neurons=0., **input_synapses)

	# ##################################################################################################################
	# Decoding / Readout Parameters
	# ##################################################################################################################
	out_resolution = inp_resolution # (since the target is a function of the input)
	filter_tau = 20.  # time constant of exponential filter (applied to spike trains)
	state_sampling = None#np.arange(0., inp_duration + 200., 50.)#None  # 1.(cannot start at 0)
	readout_labels = ['ridge_classifier']
	readout_algorithms = ['ridge']

	decoders = dict(
		decoded_population=[['E', 'I'], ['E', 'I']],
		state_variable=['spikes', 'V_m'],
		filter_time=filter_tau,
		readouts=readout_labels,
		readout_algorithms=readout_algorithms,
		sampling_times=state_sampling,
		reset_states=[False, False],
		average_states=[False, False],
		standardize=[False, False])

	decoding_pars = set_decoding_defaults(output_resolution=out_resolution, to_memory=True, **decoders)

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
	             ('input_pars', input_pars),
	             ('decoding_pars', decoding_pars),
	             ('analysis_pars', analysis_pars)])
