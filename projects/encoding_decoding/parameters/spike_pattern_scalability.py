__author__ = 'duarte'
import sys
from preset import *
import numpy as np

"""
spike_pattern_input
- main spike_input stimulus processing
"""

run = 'local'
data_label = 'ED_spikepatterninput_jitter'

# ######################################################################################################################
# PARAMETER RANGE declarations
# ======================================================================================================================
parameter_range = {
	'lexicon_size': [10],
	'T': [100] #np.arange(100, 1100, 100)
}


def build_parameters(lexicon_size, T):
	# ######################################################################################################################
	# System / Kernel Parameters
	# ######################################################################################################################
	system = dict(
		nodes=1,
		ppn=16,
		mem=48000,
		walltime='01-00:00:00',
		queue='defqueue',
		transient_time=1000.,
		sim_time=1000.)

	kernel_pars = set_kernel_defaults(run_type=run, data_label=data_label, **system)
	# override seeds (replication tests)
	# kernel_pars['grng_seed'] = 22118373677
	# kernel_pars['rng_seeds'] = [22118373678, 22118373679, 22118373680, 22118373681, 22118373682, 22118373683,
	#                             22118373684, 22118373685, 22118373686, 22118373687, 22118373688, 22118373689,
	#                             22118373690, 22118373691, 22118373692, 22118373693]
	# kernel_pars['np_seed'] = 235329953

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
	g = 11.
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

	net_pars['record_spikes'] = [False, False]
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

	# lexicon_size = 100
	n_distractors = 0  # (if applicable)
	# T = 10000
	T_discard = 10  # number of elements to discard (>=1, for some weird reasons..)

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
	inp_amplitude = 14.
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
	n_afferents = int(nE)  # number of stimulus-specific afferents (if necessary)
	n_stim = lexicon_size  # number of different input stimuli

	# wIn structure
	gamma_in = pEE
	r = 0.5
	w_in = 1.
	sig_w = 0.5 * w_in

	# jt = (jitter, True)
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
		gen_to_enc_W=None,
		jitter=None)

	encoding_pars = set_encoding_defaults(default_set=4, input_dimensions=n_stim, n_encoding_neurons=n_afferents,
	                                      **input_synapses)
	encoding_pars['encoder']['n_neurons'] = [n_afferents]
	add_parrots(encoding_pars, n_afferents, decode=False, **{}) # encoder parrots are necessary
	# #################################################################################################################
	# Decoding / Readout Parameters
	# ##################################################################################################################
	out_resolution = 1.
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

	decoding_pars = set_decoding_defaults(output_resolution=out_resolution, to_memory=True, **decoders)

	## Set decoders for input population (if applicable)
	# input_decoder = dict(
	# 	state_variable=['spikes'],
	# 	filter_time=filter_tau,
	# 	readouts=readout_labels,
	# 	readout_algorithms=readout_algorithms,
	# 	output_resolution=out_resolution,
	# 	sampling_times=state_sampling,
	# 	reset_states=[True],
	# 	average_states=[False],
	# 	standardize=[False]
	# )
	#
	# encoding_pars = add_input_decoders(encoding_pars, input_decoder, kernel_pars)
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


