__author__ = 'duarte'
import sys
from preset import *
import numpy as np

"""
two_pool_noisedriven
- 2 pool network setup (2 BRN), driven with noisy, Poissonian input, unidirectional connections P1->P2, background noise
- quantify and set population state
- run with noise_driven_dynamics in computations
- debug with noise_driven_dynamics script
"""

run = 'local'
data_label = 'ST_twopool_stimulusdriven_ud_bgnoise_025'


# ######################################################################################################################
# PARAMETER RANGE declarations
# ======================================================================================================================
parameter_range = {
	# 'nu_x': np.arange(2, 12.1, .5),
	# 'gamma': np.arange(9, 17.1, .5)
	'n_stim': np.array([5, 20, 50, 100, 500, 1000, 3000, 5000])
}


# def build_parameters(nu_x, gamma):
def build_parameters(n_stim):
	gamma 	= 16.
	nu_x 	= 5.
	noise_ratio = 0.25
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
		sim_time=2000.)

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
		            {'autapses': False, 'multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon * (1 - noise_ratio)},  # E2<-E1
		            {'autapses': False, 'multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon},  # I1<-E1
		            {'autapses': False, 'multapses': False, 'rule': 'pairwise_bernoulli', 'p': epsilon * (1 - noise_ratio)},  # I2<-E1

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
	multimeter.update({'record_from': ['V_m', 'g_in', 'g_ex'], 'record_n': 100})
	net_pars['analog_device_pars'] = [copy_dict(multimeter, {'label': 'mmE1'}),
	                                  copy_dict(multimeter, {'label': 'mmI1'}),
	                                  copy_dict(multimeter, {'label': 'mmE2'}),
	                                  copy_dict(multimeter, {'label': 'mmI2'})]
	# ##################################################################################################################
	# Stimulus Parameters
	# ##################################################################################################################
	trials_dict = {5: 100,
				   20: 400,
				   50: 1000,
				   100: 2000,
				   500: 5000,
				   1000: 10000,
				   3000: 15000,
				   5000: 20000}
	n_trials = trials_dict[n_stim]  # 100

	n_discard = 10

	# n_stim = 5

	stim_pars = dict(
		n_stim				= n_stim,
		elements			= np.arange(0, n_stim, 1).astype(int),
		grammar				= None,
		full_set_length		= int(n_trials + n_discard),
		transient_set_length= int(n_discard),
		train_set_length	= int(n_trials * 0.8),
		test_set_length		= int(n_trials * 0.2),
	)

	# ##################################################################################################################
	# Input Parameters
	# ##################################################################################################################
	inp_resolution = 0.1
	inp_amplitude = nu_x * nE
	inp_duration = 200.
	inter_stim_interval = 0.

	input_pars = {'signal': {
		'N': n_stim,
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

	# ##################################################################################################################
	# Encoding Parameters
	# ##################################################################################################################
	# Noise
	# ##################################################################################################################
	k_x = epsilon * nE

	encoding_pars = set_encoding_defaults(default_set=0)

	background_noise = dict(
		generator_label="BgNoise1",
		start=0.,
		stop=sys.float_info.max,
		origin=0.,
		rate=nu_x * k_x * noise_ratio,
		target_population_names=['E1', 'I1'],
		additional_parameters={
			'syn_specs': {},
			'models': 'static_synapse',
			'model_pars': {},
			'weight_dist': wE,
			'delay_dist': delay})
	add_background_noise(encoding_pars, background_noise)

	background_noise = dict(
		generator_label="BgNoise2",
		start=0.,
		stop=sys.float_info.max,
		origin=0.,
		rate=nu_x * k_x * noise_ratio,
		target_population_names=['E2', 'I2'],
		additional_parameters={
			'syn_specs': {},
			'models': 'static_synapse',
			'model_pars': {},
			'weight_dist': wE,
			'delay_dist': delay})
	add_background_noise(encoding_pars, background_noise)

	# ##################################################################################################################
	# Stimulus
	# ##################################################################################################################
	# n_afferents = N  # number of stimulus-specific afferents (if necessary)
	encoder_delay = 0.1
	w_in = wE

	# Input connectivity
	input_synapses = dict(
		target_population_names=['E1I1'],
		conn_specs	= [{'rule': 'pairwise_bernoulli', 'p': epsilon * (1 - noise_ratio)}],
		syn_specs	= [{}],
		models		= ['static_synapse'],
		model_pars	= [{}],
		weight_dist	= [w_in],
		delay_dist	= [encoder_delay],
		preset_W	= [None],
		gen_to_enc_W= None,
		jitter		= None) # jitter=None or jitter=(value[float], correct_borders[bool])

	encoding_pars = set_encoding_defaults(default_set=3, input_dimensions=n_stim, **input_synapses)
	# encoding_pars['encoder']['n_neurons'] = [n_afferents]

	# add_parrots(encoding_pars, n_afferents, decode=True, **{})

	# ##################################################################################################################
	# Decoding / Readout Parameters
	# ##################################################################################################################
	out_resolution 		= 0.1
	filter_tau 			= 20.  # time constant of exponential filter (applied to spike trains)
	state_sampling 		= None  # 1.(cannot start at 0)
	readout_labels 		= ['ridge_classifier']
	readout_algorithms 	= ['ridge']

	decoders = dict(
		decoded_population	= [['E1I1'], ['E1I1'], ['E2I2'], ['E2I2']],
		state_variable		= ['spikes', 'V_m', 'spikes', 'V_m'],
		filter_time			= filter_tau,
		readouts			= readout_labels,
		readout_algorithms	= readout_algorithms,
		sampling_times		= state_sampling,
		reset_states		= [True, False, True, False],
		average_states		= [False, False, False, False],
		standardize			= [False, False, False, False])

	decoding_pars = set_decoding_defaults(output_resolution=out_resolution, to_memory=True, **decoders)

	# ##################################################################################################################
	# Decoding / Readout Parameters
	# ##################################################################################################################

	analysis_pars = {
		# analysis depth
		'depth': 3,	# 1: save only summary of data, use only fastest measures
					# 2: save all data, use only fastest measures
					# 3: save only summary of data, use all available measures
					# 4: save all data, use all available measures

		'store_activity': False,	# [int] - store all population activity in the last n steps of the test
									# phase; if set True the entire test phase will be stored;

		'population_activity': {
			'time_bin': 	2.,  	# bin width for spike counts, fano factors and correlation coefficients
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
	             ('input_pars', input_pars),
	             ('neuron_pars', neuron_pars),
	             ('net_pars', net_pars),
	             ('encoding_pars', encoding_pars),
	             ('decoding_pars', decoding_pars),
	             ('stim_pars', stim_pars),
	             ('connection_pars', connection_pars),
				 ('analysis_pars', analysis_pars)])
