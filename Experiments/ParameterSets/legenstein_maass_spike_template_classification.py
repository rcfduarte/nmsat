"""
[[X]] grid= 6*6*15
[[X]] eN = 0.8
[[X]] iN = 0.2
probability of synaptic connection= distrib.

? static vs stdp synapses

----
Neuron parameters:
 [[X]] membrane time constant 30ms,
 [[X]] absolute refractory period 3ms (excitatory neurons), 2ms (inhibitory neurons),
 [[X]] threshold 15mV (for a resting membrane potential assumed to be 0),
 [[X]] reset voltage 13.5mV,
 constant nonspecific background current I b independently chosen for each neuron from the interval [14.975nA, 15.025nA],
 input resistance 1 Mohm.

----

U (use), D (time constant for depression), F (time constant for facilitation): randomly chosen
from Gaussian distributions
.5, 1.1, .05 (EE)
.05, .125, 1.2 (EI)
.25, .7, .02 (IE)
.32, .144, .06 (II).

The scaling parameter A (in nA) was chosen  to be 1.2 (EE), 1.6 (EI), -3.0 (IE), -2.8 (II).
In the case of input synapses the parameter A had a value of 0.1 nA.
The SD of each parameter was chosen to be 50 % of its mean (with negative
values replaced by values chosen from an appropriate uniform distribution).

----
[[X]]
The transmission delays between liquid neurons were chosen uniformly
to be 1.5 ms (EE), and 0.8 for the other connections.
"""


import sys
from Preset import *
from Preset.Paths import paths
import numpy as np
import random

"""
encoding_decoding_tests
- to run with function .. in Computation
or with DiscreteInput_NStep_Analysis.py (debugging)
"""
__author__ = 'duarte'

run 			= 'local'
data_label 		= 'AD_StimulusDriven_SingleRunComparison'
project_label 	= 'Alzheimer'


def topology_random_grid_3D(size_x, size_y, size_z, layers=None):
	"""
	Returns the positions of neurons on a 3D grid structure, where each point lies within [-0.5, 0.5) on
	each axis.
	:param size_x: nr size on X coordinate
	:param size_y:
	:param size_z:
	:return:
	"""
	if layers is None:
		layers = [size_x * size_y * size_z]

	# calculate normalized distance between neurons on each axis
	d_x = 1. / (size_x + 1)
	d_y = 1. / (size_y + 1)
	d_z = 1. / (size_z + 1)
	# compute grid positions on the [-0.5 + d_axis, 0.5 - d_axis) interval with d_axis distance
	coord_x = np.arange(-0.5 + d_x / 2, 0.5, d_x)
	coord_y = np.arange(-0.5 + d_y / 2, 0.5, d_y)
	coord_z = np.arange(-0.5 + d_z / 2, 0.5, d_z)

	layer_positions	= []
	used_positions  = []
	for n in layers:
		tmp_pos = []
		for i in range(n):
			pos = [random.choice(coord_x), random.choice(coord_y), random.choice(coord_z)]
			while pos in used_positions:
				pos = [random.choice(coord_x), random.choice(coord_y), random.choice(coord_z)]

			tmp_pos.append(pos)  		#  add pos to current layer
			used_positions.append(pos) 	#  exclude current pos in future
		layer_positions.append(tmp_pos)
	return layer_positions


def build_parameters():
	# ######################################################################################################################
	# System / Kernel Parameters
	# ######################################################################################################################
	system = dict(
		nodes	= 1,
		ppn		= 16,
		mem		= 32,
		walltime= '01-00:00:00',
		queue	= 'batch',
		transient_time = 1000.,
		sim_time= 2000.)

	kernel_pars = set_kernel_defaults(default_set=3, run_type=run, data_label=data_label, project_label=project_label,
									  **system)
	# ######################################################################################################################
	# Neuron, Synapse and Network Parameters
	# ######################################################################################################################
	N 		= 540
	delayEE = 1.5
	delay  	= 0.8

	gamma = 6.

	wE = 32.29
	wI = -gamma * wE

	recurrent_synapses = dict(
		synapse_model_parameters = [{}, {}, {}, {}], # Question: what's this?
		connected_populations	 = [('E', 'E'), ('E', 'I'), ('I', 'E'), ('I', 'I')],
		# synapse_models 			 = ['static_synapse', 'static_synapse', 'static_synapse', 'static_synapse'],
		synapse_models 	= ['stdp_synapse', 'stdp_synapse', 'stdp_synapse', 'stdp_synapse'],
		pre_computedW 	= [None, None, None, None], # Question: needed right now?
		weights 		= [wE, wI, wE, wI],
		delays 			= [delayEE, delay, delay, delay],
		conn_specs 		= [   {'connection_type': 'divergent', 'allow_autapses': False, 'allow_multapses': True,
							   'kernel': {'gaussian': {'p_center': 1., 'sigma': 4., 'mean': 0., 'c': 0.}}},
							  {'connection_type': 'divergent', 'allow_autapses': False, 'allow_multapses': True,
							   'kernel': {'gaussian': {'p_center': 1., 'sigma': 4., 'mean': 0., 'c': 0.}}},
							  {'connection_type': 'divergent', 'allow_autapses': False, 'allow_multapses': True,
							   'kernel': {'gaussian': {'p_center': 1., 'sigma': 4., 'mean': 0., 'c': 0.}}},
							  {'connection_type': 'divergent', 'allow_autapses': False, 'allow_multapses': True,
							   'kernel': {'gaussian': {'p_center': 1., 'sigma': 4., 'mean': 0., 'c': 0.}}} ],
		syn_specs 		= [{}, {}, {}, {}]
	)
	neuron_pars, net_pars, connection_pars = set_network_defaults(default_set=4, neuron_set=6, connection_set=1.1, N=N,
																  kernel_pars=kernel_pars, **recurrent_synapses)

	net_pars['neuron_pars'][0].update({'V_reset': 13.5, 'tau_m': 30., 't_ref': 3.})
	net_pars['neuron_pars'][1].update({'V_reset': 13.5, 'tau_m': 30., 't_ref': 2.})

	connection_pars.topology_dependent = [True, True, True, True] # Question: True for connection within 1 layer also?
	grid_positions = topology_random_grid_3D(6, 6, 15, [int(6*6*15*.8), int(6*6*15*.2)])

	net_pars.update({ 'topology': 		[True, True],
					  'topology_dict': 	[{'positions': grid_positions[0]}, {'positions': grid_positions[1]}]})
	# ######################################################################################################################
	# Input Parameters
	# ######################################################################################################################
	n_trials 	= 2500
	n_discard 	= 10

	n_stim = 80

	stim_pars = dict(
		n_stim 		= n_stim,
		elements 	= np.arange(0, n_stim, 1).astype(int),
		grammar 	= None,
		full_set_length 	 = int(n_trials + n_discard),
		transient_set_length = int(n_discard),
		train_set_length 	 = 2000,
		test_set_length 	 = 500
	)

	inp_resolution 		= 1.
	inp_amplitude 		= 20.
	inp_duration 		= 200.
	inter_stim_interval = 0.

	input_pars = {'signal': {
					'N': n_stim,
						'durations'		: [inp_duration],
						'i_stim_i'		: [inter_stim_interval],
						'kernel'		: ('box', {}),
						'start_time'	: 0.,
						'stop_time'		: sys.float_info.max,
						'max_amplitude'	: [inp_amplitude],
						'min_amplitude'	: 0.,
						'resolution'	: inp_resolution},
					'noise': {
						'N'				: 0,
						'noise_source'	: ['GWN'],
						'noise_pars'	: {'amplitude': 5., 'mean': 1., 'std': 0.25},
						'rectify'		: True,
						'start_time'	: 0.,
						'stop_time'		: sys.float_info.max,
						'resolution'	: inp_resolution, } }

	# ######################################################################################################################
	# Encoding Parameters
	# ######################################################################################################################
	# Question: this one?
	filter_tau 	= 20.                              # time constant of exponential filter (applied to spike trains)
	n_afferents = 4		                           # number of stimulus-specific afferents (if necessary)
	n_stim 		= stim_pars['n_stim']              # number of different input stimuli

	w_in = 20. # Question: this one?

	# Input connectivity
	input_synapses = dict(
		target_population_names = ['EI'],
		conn_specs	= [{'rule': 'fixed_outdegree', 'outdegree': 25}],
		syn_specs	= [{}],
		models		= ['static_synapse'],
		model_pars	= [{}],
		weight_dist	= [w_in],
		delay_dist	= [1.],
		preset_W	= [None],
		gen_to_enc_W= None,
		jitter		= None) # Question: how to define gaussian jitter?

	encoding_pars = set_encoding_defaults(default_set=4, input_dimensions=n_stim,
										  n_encoding_neurons=n_afferents, **input_synapses)
	add_parrots(encoding_pars, n_afferents, decode=True, **{})

	# ######################################################################################################################
	# Decoding / Readout Parameters
	# ######################################################################################################################
	state_sampling = None #1.(cannot start at 0)
	readout_labels = ['class0']

	decoders = dict(
		decoded_population		= [['E', 'I']],
		state_variable			= ['spikes'],
		filter_time				= filter_tau,
		readouts				= readout_labels,
		readout_algorithms		= ['ridge'],
		global_sampling_times	= state_sampling)

	decoding_pars = set_decoding_defaults(default_set=1, output_resolution=1., to_memory=True, kernel_pars=kernel_pars,
										  **decoders)

	## Set decoders for input population (if applicable)
	input_decoder=dict(
		state_variable	= ['spikes'],
		filter_time		= filter_tau,
		readouts		= readout_labels,
		readout_algorithms		= ['ridge'],
		output_resolution		= inp_resolution,
		global_sampling_times	= state_sampling)

	encoding_pars = set_input_decoders(encoding_pars, input_decoder, kernel_pars)

	####################################################################################################################
	# RETURN dictionary of Parameters dictionaries
	#===================================================================================================================
	return dict([('kernel_pars', 	kernel_pars),
				 ('input_pars', 	input_pars),
				 ('neuron_pars', 	neuron_pars),
				 ('net_pars',	 	net_pars),
				 ('encoding_pars', 	encoding_pars),
				 ('decoding_pars', 	decoding_pars),
				 ('stim_pars', 		stim_pars),
				 ('connection_pars',connection_pars)])

# build_parameters()