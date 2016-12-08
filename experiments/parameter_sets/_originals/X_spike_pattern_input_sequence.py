import sys
from Preset import *
from Preset.Paths import paths
import numpy as np

"""
encoding_decoding_tests
- to run with function .. in Computation
or with DiscreteInput_NStep_Analysis.py (debugging)
"""
__author__ = 'duarte'

run 			= 'local'
data_label 		= 'AD_StimulusDriven_SingleRunComparison'
project_label 	= 'Alzheimer'
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
N 		= 1250
delay 	= 1.
epsilon = 0.1
kE 		= int(epsilon * (N * 0.8))
kI 		= int(epsilon * (N * 0.2))

gamma = 6.

wE = 32.29
wI = -gamma * wE


recurrent_synapses = dict(
	synapse_model_parameters = [{}, {}, {}, {}],
	connected_populations	 = [('E', 'E'), ('E', 'I'), ('I', 'E'), ('I', 'I')],
	synapse_models 			 = ['static_synapse', 'static_synapse', 'static_synapse', 'static_synapse'],
	pre_computedW 	= [None, None, None, None],
	weights 		= [wE, wI, wE, wI],
	delays 			= [delay, delay, delay, delay],
	conn_specs 		= [ {'rule': 'fixed_indegree', 'indegree': kE, 'autapses': False, 'multapses': True},
						{'rule': 'fixed_indegree', 'indegree': kI, 'autapses': False, 'multapses': True},
						{'rule': 'fixed_indegree', 'indegree': kE, 'autapses': False, 'multapses': True},
						{'rule': 'fixed_indegree', 'indegree': kI, 'autapses': False, 'multapses': True} ],
	syn_specs 		= [{}, {}, {}, {}]
)
neuron_pars, net_pars, connection_pars = set_network_defaults(default_set=4, neuron_set=6, connection_set=1.1, N=N,
                                                              kernel_pars=kernel_pars, **recurrent_synapses)
# ######################################################################################################################
# Input Parameters
# ######################################################################################################################
n_trials 	= 100
n_discard 	= 10

n_stim = 5

stim_pars = dict(
	n_stim 		= n_stim,
	elements 	= np.arange(0, n_stim, 1).astype(int),
	grammar 	= None,
	full_set_length 	 = int(n_trials + n_discard),
	transient_set_length = int(n_discard),
	train_set_length 	 = int(n_trials * 0.8),
	test_set_length 	 = int(n_trials * 0.2),
)

inp_resolution 		= 1.
inp_amplitude 		= 1200.
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
filter_tau 	= 20.                              # time constant of exponential filter (applied to spike trains)
n_afferents = 1250                             # number of stimulus-specific afferents (if necessary)
n_stim 		= stim_pars['n_stim']              # number of different input stimuli

w_in = 20.

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
	jitter		= None)

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

# ######################################################################################################################
# Report Parameters
# ######################################################################################################################
report_pars = set_report_defaults(3, run, paths, kernel_pars, neuron_pars, net_pars, connection_pars, encoding_pars,
                                   decoding_pars={})