"""
[[SOLVED]] grid= 6*6*15
[[SOLVED]] eN = 0.8
[[SOLVED]] iN = 0.2
[[SOLVED]] ++> probability of synaptic connection= distrib. There's a C in Maass, but not our paper. C = p_center

======================================================================
Neuron parameters:
[[SOLVED]] 	membrane time constant 30ms,
[[SOLVED]] 	absolute refractory period 3ms (excitatory neurons), 2ms (inhibitory neurons),
[[SOLVED]] 	threshold 15mV (for a resting membrane potential assumed to be 0),
[[SOLVED]] 	reset voltage 13.5mV,
[[CHECK]]   constant nonspecific background current I_b independently chosen for each neuron
			from the interval [14.975nA, 15.025nA],
			==> randomize initial value, careful at units
			+?> used np.uniform, individually for each neuron. Still need randomization?
[[CHECK]]   input resistance 1 Mohm.  ===> t=RC, solve equation
			t=RC, s= ohm * Faraday === ms = Mohm * pF * 10^-3

======================================================================
[[SOLVED]]
	The postsynaptic current was modeled as an exponential decay exp(-t/tau_s )
	with tau_s =3ms (tau_s =6ms) for excitatory (inhibitory) synapses

[[SOLVED]]
   "2.) The time constant of each tsodyks_synapse targeting a particular neuron
   must be chosen equal to that neuron's synaptic time constant. In particular that means
   that all synapses targeting a particular neuron have the same parameter tau_psc."

   tau_psc (tsodyks) == tau_syn_ex / tau_syn_in (iaf_psc_exp) ???

======================================================================
[[GAVE IT A SHOT]]
[[HALFWAY]]
	Ref. says: U (use), D (time constant for depression), F (time constant for facilitation):
	all randomly chosen from Gaussian distributions

	Questions:
		1) is U (use) === U (maximum probability of release) in the model?
	    	U         double - maximum probability of release [0,1] 	= U?
            tau_psc   double - time constant of synaptic current in ms  = time constant for decay? = tau_psc, one in the synapse, and also one in the iaf_psc_exp neuron?
            tau_fac   double - time constant for facilitation in ms     = F ?
            tau_rec   double - time constant for depression in ms       = D ?
	===> values are actually in seconds, in NEST ms though.. adjust!!!!
	Values from paper: U, D, F
		.5, 	1.1, 	.05 (EE)
		.05, 	.125, 	1.2 (EI)
		.25, 	.7, 	.02 (IE)
		.32, 	.144, 	.06 (II).

	===> trunc_normal instead of normal distribution for tau_fac, tau_rec, etc.

[[SOLVED]] === what's the A?
	The scaling parameter A (in nA) was chosen  to be 1.2 (EE), 1.6 (EI), -3.0 (IE), -2.8 (II).
	In the case of input synapses the parameter A had a value of 0.1 nA.
	The SD of each parameter was chosen to be 50 % of its mean (with negative
	values replaced by values chosen from an appropriate uniform distribution).

	===> A is the synaptic weight, in pA (paper has nA), it's also a distribution, means are defined!
	===> first try it fixed, maybe later turn to real distributions?!
	W_scale scales just A

======================================================================
[[SOLVED]] The transmission delays between liquid neurons were chosen uniformly
			to be 1.5 ms (EE), and 0.8 for the other connections.

======================================================================
[[SOLVED]] jitter with N(0, 10ms)

======================================================================
[[MISSING]]
I_background = "constant unspecific background current"

the reference says:
	The background current, I b , had a constant value for each neuron, randomly
	distributed across the network. We chose a uniform distribution centered
	at the threshold level with a range of 0.05 mV; this resulted in the basal
	firing rates of the excitatory neurons being between 1 and 20 Hz, with an
	average of 7 Hz.
"""


from Preset import *
import numpy as np
import random

"""

"""
__author__ = 'duarte'

run 			= 'local'
data_label 		= 'Legenstein_kernel_queality_lambda=6.W_scale=6'
project_label 	= 'Alzheimer'


def topology_random_grid_3D(size_x, size_y, size_z, layers=None):
	"""
	Returns the positions of neurons on a 3D grid structure, where each point lies within [-0.5, 0.5) on
	each axis.
	:param size_x: size on X coordinate
	:param size_y:
	:param size_z:
	:param layers: a list specifying how many neurons should be placed on each layer
	:return:
	"""
	if layers is None:
		layers = [size_x * size_y * size_z]

	# compute possible grid positions
	coord_x = np.arange(-size_x / 2., size_x / 2., 1)
	coord_y = np.arange(-size_y / 2., size_y / 2., 1)
	coord_z = np.arange(-size_z / 2., size_z / 2., 1)

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
	N 		= 540 # 6x6x15 grid
	N_e		= .8  # number of excitatory neurons
	N_i		= .2  # number of inhibitory neurons

	delayEE = 1.5  # delay for EE synapses
	delay  	= 0.8  # delay for all other synapses

	kernel_lambda = 6.  # definition corresponds exactly to the one in the paper
	W_scale 	 = 6.
	connect_C_EE = 0.3 # connectivity parameter C controlling exp. distribution (Maass 2002)
	connect_C_EI = 0.2 # connectivity parameter C controlling exp. distribution (Maass 2002)
	connect_C_IE = 0.4 # connectivity parameter C controlling exp. distribution (Maass 2002)
	connect_C_II = 0.1 # connectivity parameter C controlling exp. distribution (Maass 2002)
	kernel_sigma = kernel_lambda / np.sqrt(2)  #required because we use a Gaussian for the exponential distance

	recurrent_synapses = dict(
		synapse_model_parameters = [{}, {}, {}, {}],
		# T<-S
		connected_populations	 = [('E', 'E'), ('E', 'I'), ('I', 'E'), ('I', 'I')],
		synapse_names 	= ['EE_synapse', 'EI_synapse', 'IE_synapse', 'II_synapse'],
		synapse_models 	= ['tsodyks_synapse', 'tsodyks_synapse', 'tsodyks_synapse', 'tsodyks_synapse'],
		pre_computedW 	= [None, None, None, None],
		# TODO in the paper this is actually a normal distribution, for now fixed but later try changing it?
		# Maass 2002 says nA as unit, NEST uses pA.. 1.2 nA == 1200 pA
		weights 		= [1200 * W_scale, -3000 * W_scale, 1600 * W_scale, -2800 * W_scale],
		delays 			= [delayEE, delay, delay, delay],
		conn_specs 		= [   {'connection_type': 'divergent', 'allow_autapses': False, 'allow_multapses': True,
							   'kernel': {'gaussian': {'p_center': connect_C_EE, 'sigma': kernel_sigma, 'mean': 0., 'c': 0.}}},
							  {'connection_type': 'divergent', 'allow_autapses': False, 'allow_multapses': True,
							   'kernel': {'gaussian': {'p_center': connect_C_EI, 'sigma': kernel_sigma, 'mean': 0., 'c': 0.}}},
							  {'connection_type': 'divergent', 'allow_autapses': False, 'allow_multapses': True,
							   'kernel': {'gaussian': {'p_center': connect_C_IE, 'sigma': kernel_sigma, 'mean': 0., 'c': 0.}}},
							  {'connection_type': 'divergent', 'allow_autapses': False, 'allow_multapses': True,
							   'kernel': {'gaussian': {'p_center': connect_C_II, 'sigma': kernel_sigma, 'mean': 0., 'c': 0.}}} ],
		syn_specs		= [
							# E<-E
							{
								'model': 	'EE_synapse',
								'tau_psc': 	3., # excitatory synapses
								'tau_fac': 	{'distribution': 'normal_clipped', 'mu': 0.05, 'sigma': 0.025},
								'tau_rec': 	{'distribution': 'normal_clipped', 'mu': 1.1, 'sigma': 0.55},
								'U':		{'distribution': 'normal_clipped', 'mu': 0.5, 'sigma': 0.25}},

							# E<-I
							{
								'model': 	'EI_synapse',
								'tau_psc': 	6., # inhibitory synapses
								'tau_fac': 	{'distribution': 'normal_clipped', 'mu': 0.02, 'sigma': 0.01},
								'tau_rec': 	{'distribution': 'normal_clipped', 'mu': 0.7, 'sigma': 0.35},
								'U': 		{'distribution': 'normal_clipped', 'mu': 0.25, 'sigma': 0.125}},

							# IE
							{
								'model': 	'IE_synapse',
								'tau_psc': 	3.,  # excitatory synapses
								'tau_fac': 	{'distribution': 'normal_clipped', 'mu': 1.2, 'sigma': 0.6},
								'tau_rec': 	{'distribution': 'normal_clipped', 'mu': 0.125, 'sigma': 0.0625},
								'U':  		{'distribution': 'normal_clipped', 'mu': 0.05, 'sigma': 0.}},

							# II
							{
								'model': 	'II_synapse',
								'tau_psc': 	6., # inhibitory synapses
								'tau_fac': 	{'distribution': 'normal_clipped', 'mu': 0.06, 'sigma': 0.03},
								'tau_rec': 	{'distribution': 'normal_clipped', 'mu': 0.144, 'sigma': 0.072},
								'U': 		{'distribution': 'normal_clipped', 'mu': 0.32, 'sigma': 0.16} } ]
	)

	neuron_pars, net_pars, connection_pars = set_network_defaults(default_set=4, neuron_set=6, connection_set=1.1, N=N,
																  kernel_pars=kernel_pars, **recurrent_synapses)

	# for C_m:  30ms = 1Mohm * 30000 pF
	net_pars['neuron_pars'][0].update({'V_reset': 13.5,
									   'tau_m': 30., # in ms
									   'tau_syn_ex': 3.,  # tau for excitatory synapses
									   'tau_syn_in': 6.,  # tau for inhibitory synapses
									   't_ref': 3.,
									   'I_e': float(np.random.uniform(14975, 15025)),
									   'C_m': 30. * 1000 }) # C_m is in pF as required by the neuron model

	net_pars['neuron_pars'][1].update({'V_reset': 13.5,
									   'tau_m': 30.,
									   'tau_syn_ex': 3.,  # tau for excitatory synapses
									   'tau_syn_in': 6.,  # tau for inhibitory synapses
									   't_ref': 2.,
									   'I_e': float(np.random.uniform(14975, 15025)),
									   'C_m': 30. * 1000})  # C_m is in pF as required by the neuron model

	connection_pars.topology_dependent = [True, True, True, True]
	grid_positions = topology_random_grid_3D(6, 6, 15, [int(6*6*15*N_e), int(6*6*15*N_i)])

	net_pars.update({ 'topology': 		[True, True],
					  'topology_dict': 	[{'positions': grid_positions[0], 'extent':	[6., 6., 15.]},
										 {'positions': grid_positions[1], 'extent': [6., 6., 15.]}]})

	# ######################################################################################################################
	# Input Parameters
	# ######################################################################################################################
	n_trials 	= 100 # 2500
	n_discard 	= 10

	n_stim = 500

	stim_pars = dict(
		n_stim 		= n_stim,
		elements 	= np.arange(0, n_stim, 1).astype(int),
		grammar 	= None,
		full_set_length 	 = int(n_trials + n_discard),
		transient_set_length = int(n_discard),
		train_set_length 	 = 50, #2000,
		test_set_length 	 = 50 #500
	)

	inp_resolution 		= 1.
	inp_amplitude 		= 20. # 20.
	inp_duration 		= 200.
	inter_stim_interval = 0.

	input_pars = {'signal': {
					'N' 			: n_stim,
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
	filter_tau 	= 30.                              # time constant of exponential filter (applied to spike trains)
	n_afferents = 4		                           # number of stimulus-specific afferents (if necessary)
	n_stim 		= stim_pars['n_stim']              # number of different input stimuli

	w_in = 100. # [pA] In the case of input synapses the parameter A had a value of 0.1 nA.

	# Input connectivity
	input_synapses = dict(
		target_population_names = ['EI'],
		conn_specs	= [{'rule': 'fixed_outdegree', 'outdegree': int(N * 0.3)}],
		syn_specs	= [{}],
		models		= ['static_synapse'],
		model_pars	= [{}],
		weight_dist	= [w_in],
		delay_dist	= [1.],
		preset_W	= [None],
		gen_to_enc_W= None,
		jitter		= 10.) # Gaussian jitter with 10ms SD, 0 mean

	encoding_pars = set_encoding_defaults(default_set=4, input_dimensions=n_stim,
										  n_encoding_neurons=n_afferents, **input_synapses)
	add_parrots(encoding_pars, n_afferents, decode=True, **{})

	# ######################################################################################################################
	# Decoding / Readout Parameters
	# ######################################################################################################################
	state_sampling 	= None #1.(cannot start at 0)
	readout_labels 	= ['class0']

	decoders = dict(
		decoded_population		= [['E', 'I']],
		state_variable			= ['spikes'],
		filter_time				= filter_tau,
		readouts				= readout_labels,
		readout_algorithms		= ['pinv'],
		global_sampling_times	= state_sampling)

	decoding_pars = set_decoding_defaults(default_set=1, output_resolution=1., to_memory=True, kernel_pars=kernel_pars,
										  **decoders)

	## Set decoders for input population (if applicable)
	input_decoder=dict(
		state_variable	= ['spikes'],
		filter_time		= filter_tau,
		readouts		= readout_labels,
		readout_algorithms		= ['pinv'],
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