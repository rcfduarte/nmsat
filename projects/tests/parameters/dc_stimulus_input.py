__author__ = 'duarte'
import sys
from preset import *
import numpy as np

"""
dc_input
- test dc_input stimulus processing
"""

run = 'local'
data_label = 'DCinput_scaling_{0}'.format(run)

# ######################################################################################################################
# PARAMETER RANGE declarations
# ======================================================================================================================
parameter_range = {
	'ppn': np.arange(2, 24, 2),
	# 'n_th': np.arange(1, 10, 1)
}


def build_parameters(ppn):
	# ######################################################################################################################
	# System / Kernel Parameters
	# ######################################################################################################################
	system = dict(
		nodes=1,
		ppn=ppn,
		n_th=1,
		mem=8,
		walltime='01-00:00:00',
		queue='batch',
		transient_time=1000.,
		sim_time=1000.)

	kernel_pars = set_kernel_defaults(run_type=run, data_label=data_label, **system)

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

	# net_pars['record_spikes'] = [True, True]

	# net_pars['record_analogs'] = [True, False]
	# multimeter = rec_device_defaults(device_type='multimeter')
	# multimeter.update({'record_from': ['V_m', 'g_ex', 'g_in'], 'record_n': 1000})
	# net_pars['analog_device_pars'] = [copy_dict(multimeter, {'label': ''}), {}]

	# ######################################################################################################################
	# Task and Stimulus Parameters
	# ######################################################################################################################
	task = 1  # Accepted tasks:
	# - identity mapping (1);
	# - delayed identity mapping (2);
	# - delayed identity mapping with distractors (3);
	# - adjacent dependencies (4);
	# - non-adjacent dependencies (5);
	# - pattern mapping with cross dependencies (6);
	# - hierarchical dependencies (7);

	lexicon_size = 10
	n_distractors = 0  # (if applicable)
	T = 1000
	T_discard = 2  # number of elements to discard (>=1, for some weird reasons..)

	random_dt = False  # if True, dt becomes maximum distance (?)
	dt = 3  # delay (if applicable)

	pause_t = 1  # pause between 2 items

	C_len = 2  # length of patterns (if applicable) - only 2! (?)

	task_pars = {'task': task,
	             'lexicon_size': lexicon_size,
	             'T': T + T_discard,
	             'random_dt': random_dt,
	             'dt': dt,
	             'pause_t': pause_t,
	             'C_len': C_len,
	             'n_distractors': n_distractors}

	stim_pars = dict(
		n_stim=lexicon_size,
		full_set_length=int(T + T_discard),
		transient_set_length=int(T_discard),
		train_set_length=int(0.8 * T),
		test_set_length=int(0.2 * T))

	# ######################################################################################################################
	# Input Parameters
	# ######################################################################################################################
	inp_resolution = 0.1
	inp_amplitude = 1500.
	inp_duration = 200.
	inter_stim_interval = 0.

	input_pars = {'signal': {
		'N': lexicon_size,
		'durations': [inp_duration],
		'i_stim_i': [inter_stim_interval],
		'kernel': ('box', {}),
		'start_time': 0.,
		'stop_time': sys.float_info.max,
		'max_amplitude': [inp_amplitude],
		'min_amplitude': 0.,
		'resolution': inp_resolution},
		# 'noise': {
		# 	'N': 0,
		# 	'noise_source': ['GWN'],
		# 	'noise_pars': {'amplitude': 5., 'mean': 1., 'std': 0.25},
		# 	'rectify': True,
		# 	'start_time': 0.,
		# 	'stop_time': sys.float_info.max,
		# 	'resolution': inp_resolution, }
	}

	# ######################################################################################################################
	# Encoding Parameters
	# ######################################################################################################################
	gamma_in = pEE
	r = 0.5
	w_in = 1.
	sig_w = 0.5 * w_in

	# Input connectivity
	input_synapses = dict(
		target_population_names=['E', 'I'],
		conn_specs=[{'rule': 'pairwise_bernoulli', 'p': gamma_in},
		            {'rule': 'pairwise_bernoulli', 'p': gamma_in}],
		syn_specs=[{}, {}],
		models=['static_synapse', 'static_synapse'],
		model_pars=[{}, {}],
		weight_dist=[
			{'distribution': 'normal_clipped', 'mu': w_in, 'sigma': 0.5 * w_in, 'low': 0.0001, 'high': 10. * w_in},
			{'distribution': 'normal_clipped', 'mu': w_in, 'sigma': 0.5 * w_in, 'low': 0.0001, 'high': 10. * w_in}],
		delay_dist=[0.1, 0.1],
		preset_W=[None, None],
		gen_to_enc_W=None)

	encoding_pars = set_encoding_defaults(default_set=1, input_dimensions=1, n_encoding_neurons=0.,
	                                      **input_synapses)

	# ##################################################################################################################
	# Decoding / Readout Parameters
	# ##################################################################################################################
	out_resolution = 0.1
	filter_tau = 20.  # time constant of exponential filter (applied to spike trains)
	state_sampling = None  # 1.(cannot start at 0)
	readout_labels = ['ridge_classifier', 'pinv_classifier']
	readout_algorithms = ['ridge', 'pinv']

	decoders = dict(
		decoded_population=[['E', 'I'], ['E', 'I']],
		state_variable=['V_m', 'spikes'],
		filter_time=filter_tau,
		readouts=readout_labels,
		readout_algorithms=readout_algorithms,
		sampling_times=state_sampling,
		reset_states=[False, False],
		average_states=[False, False],
		standardize=[False, False]
	)

	decoding_pars = set_decoding_defaults(output_resolution=out_resolution, origin=0.0, to_memory=True, **decoders)

	# ##################################################################################################################
	# Extra analysis parameters (specific for this experiment)
	# ==================================================================================================================
	analysis_pars = {
		# analysis depth
		'depth': 1,  	# 1: save only summary of data, use only fastest measures
						# 2: save all data, use only fastest measures
						# 3: save only summary of data, use all available measures
						# 4: save all data, use all available measures

		'store_activity': False,  		# [int] - store all population activity in the last n steps of the test
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
	             ('task_pars', task_pars),
	             ('decoding_pars', decoding_pars),
	             ('stim_pars', stim_pars),
	             ('analysis_pars', analysis_pars)])

