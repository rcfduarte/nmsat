__author__ = 'duarte'
import sys
from preset import *
import numpy as np

"""
spike_pattern_input
- main spike_input stimulus processing
"""

run = 'local'
data_label = 'example1'

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
		mem=32,
		walltime='01-00:00:00',
		queue='batch',
		transient_time=1000.,
		sim_time=2000.)

	kernel_pars = set_kernel_defaults(run_type=run, data_label=data_label, **system)

	# ##################################################################################################################
	# Neuron, Synapse and Network Parameters
	# ##################################################################################################################
	N = 1250
	delay = 1.
	epsilon = 0.1
	kE = int(epsilon * (N * 0.8))
	kI = int(epsilon * (N * 0.2))

	wE = 32.29
	gamma = 6.
	wI = -gamma * wE

	recurrent_synapses = dict(
		connected_populations=[('E', 'E'), ('E', 'I'), ('I', 'E'), ('I', 'I')],
		synapse_models=['static_synapse', 'static_synapse', 'static_synapse', 'static_synapse'],
		synapse_model_parameters=[{}, {}, {}, {}],
		pre_computedW=[None, None, None, None],
		weights=[wE, wI, wE, wI],
		delays=[delay, delay, delay, delay],
		conn_specs=[{'rule': 'fixed_indegree', 'indegree': kE, 'autapses': False, 'multapses': False},
		            {'rule': 'fixed_indegree', 'indegree': kI, 'autapses': False, 'multapses': False},
		            {'rule': 'fixed_indegree', 'indegree': kE, 'autapses': False, 'multapses': False},
		            {'rule': 'fixed_indegree', 'indegree': kI, 'autapses': False, 'multapses': False}],
		syn_specs=[{}, {}, {}, {}]
	)
	neuron_pars, net_pars, connection_pars = set_network_defaults(neuron_set=0, N=N, kernel_pars=kernel_pars,
	                                                              **recurrent_synapses)

	# ##################################################################################################################
	# Stimulus Parameters
	# ##################################################################################################################
	n_trials = 100
	n_discard = 10

	n_stim = 5

	stim_pars = dict(
		n_stim=n_stim,
		elements=np.arange(0, n_stim, 1).astype(int),
		grammar=None,
		full_set_length=int(n_trials + n_discard),
		transient_set_length=int(n_discard),
		train_set_length=int(n_trials * 0.8),
		test_set_length=int(n_trials * 0.2))

	# ######################################################################################################################
	# Input Parameters
	# ######################################################################################################################
	inp_resolution = 0.1
	inp_amplitude = 1200.
	inp_duration = 200.
	inter_stim_interval = 0.

	input_pars = {
		'signal': {
			'N': n_stim,
			'durations': [inp_duration],
			'i_stim_i': [inter_stim_interval],
			'kernel': ('box', {}),
			'start_time': 0.,
			'stop_time': sys.float_info.max,
			'max_amplitude': [inp_amplitude],
			'min_amplitude': 0.,
			'resolution': inp_resolution}}

	# ##################################################################################################################
	# Encoding Parameters
	# ##################################################################################################################
	n_afferents = 1250
	encoder_delay = 0.1
	w_in = 90.

	# Input connectivity
	input_synapses = dict(
		target_population_names=['EI'],
		conn_specs=[{'rule': 'one_to_one'}],
		syn_specs=[{}],
		models=['static_synapse'],
		model_pars=[{}],
		weight_dist=[w_in],
		delay_dist=[encoder_delay],
		preset_W=[None],
		gen_to_enc_W=None,
		jitter=None) # jitter=None or jitter=(value[float], correct_borders[bool])

	encoding_pars = set_encoding_defaults(default_set=4, input_dimensions=n_stim,
	                                      n_encoding_neurons=n_afferents, **input_synapses)
	encoding_pars['encoder']['n_neurons'] = [n_afferents]

	add_parrots(encoding_pars, n_afferents, decode=False, **{})

	# ##################################################################################################################
	# Decoding / Readout Parameters
	# ##################################################################################################################
	out_resolution = 0.1
	filter_tau = 20.  # time constant of exponential filter (applied to spike trains)
	state_sampling = None

	nSteps = 10
	readout_labels = []
	for nn in range(int(nSteps)):
		readout_labels.append('mem{0}'.format(nn + 1))
	readout_labels.append('class0')

	readout_algorithms = ['ridge' for _ in range(len(readout_labels))]

	decoders = dict(
		decoded_population=[['E', 'I']],
		state_variable=['V_m'],
		filter_time=filter_tau,
		readouts=readout_labels,
		readout_algorithms=readout_algorithms,
		sampling_times=state_sampling,
		reset_states=[False],
		average_states=[False],
		standardize=[False]
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
	# 	average_states=[True],
	# 	standardize=[False]
	# )
	#
	# encoding_pars = add_input_decoders(encoding_pars, input_decoder, kernel_pars)

	# ##################################################################################################################
	# RETURN dictionary of Parameters dictionaries
	# ==================================================================================================================
	return dict([('kernel_pars', 	kernel_pars),
				 ('input_pars', 	input_pars),
				 ('neuron_pars', 	neuron_pars),
				 ('net_pars',	 	net_pars),
				 ('encoding_pars', 	encoding_pars),
				 ('decoding_pars', 	decoding_pars),
				 ('stim_pars', 		stim_pars),
				 ('connection_pars',connection_pars)])


