__author__ = 'duarte'
import sys
sys.path.insert(0, "../")
# from preset import *
from preset.old_presets import *
import numpy as np
from auxiliary_fcns import determine_lognormal_parameters

"""
synaptic_responses@rest
- run with synaptic_responses in computations
- debug with script synaptic_response.py
"""

run = 'local'
data_label = 'E_Glus_Heterogeneous3'
project_label = 'Timescales'

# ######################################################################################################################
# PARAMETER RANGE declarations
# ======================================================================================================================
parameter_range = {

	}


########################################################################################################################
def build_parameters():
	# ampa = 1.6
	# nmda = 0.003
	# gabaa = 1.0
	# gabab = 0.022

	n_neurons = 1
	window = [-10., 500.]
	ln_pars = determine_lognormal_parameters(0.45, 4. * 0.45, median=0.45)
	w = 1.
	# {'distribution': 'lognormal_clipped', 'mu': ln_pars[0], 'sigma': ln_pars[1], 'low': 0.0001,
	# 'high': 100.*0.45}
	ln_pars = determine_lognormal_parameters(1.8, 2. * 1.8, median=1.8)
	d = 0.8  # {'distribution': 'lognormal_clipped', 'mu': ln_pars[0], 'sigma': ln_pars[1], 'low': 0.1,
	# 'high': 100.*1.8}
	neuron_type = 'E'
	heterogeneous = False

	# ######################################################################################################################
	# System / Kernel Parameters
	# ######################################################################################################################
	system = dict(
		nodes=1,
		ppn=8,
		mem=32,
		walltime='00:20:00:00',
		queue='singlenode',
		transient_time=500.,
		sim_time=1000.)
	kernel_pars = set_kernel_defaults(default_set=1, run_type=run, data_label=data_label, project_label=project_label,
	                                  **system)
	np.random.seed(kernel_pars['np_seed'])
	kernel_pars.update({'window': window})

	# ######################################################################################################################
	# Neuron/Network Parameters
	# ######################################################################################################################
	randomized_pars = randomized_neuron_parameters(heterogeneous=heterogeneous)
	if heterogeneous:
		neuron_pars = set_neuron_defaults(default_set=1.2)
	else:
		neuron_pars = set_neuron_defaults(default_set=1.1)
		randomized_pars.update({neuron_type: {}})

	neuron_pars[neuron_type]['E_L'] = -55.

	# neuron_pars[neuron_type]['rec_cond'][neuron_pars[neuron_type]['rec_names'].index('AMPA')] = ampa
	# neuron_pars[neuron_type]['rec_cond'][neuron_pars[neuron_type]['rec_names'].index('NMDA')] = nmda
	# neuron_pars[neuron_type]['rec_cond'][neuron_pars[neuron_type]['rec_names'].index('GABA_{A}')] = gabaa
	# neuron_pars[neuron_type]['rec_cond'][neuron_pars[neuron_type]['rec_names'].index('GABA_{B}')] = gabab

	multimeter = rec_device_defaults(device_type='multimeter')
	multimeter.update({'record_from': ['V_m', 'C1', 'C2', 'C3', 'C4', 'I_ex', 'I_in', 'G_syn_tot'], 'record_n': None})
	net_pars = ParameterSet({
		'n_populations': 1,
		'pop_names': [neuron_type],
		'n_neurons': [n_neurons],
		'neuron_pars': [neuron_pars[neuron_type]],
		'randomize_neuron_pars': [randomized_pars[neuron_type]],
		'topology': [False],
		'topology_dict': [None],
		'record_spikes': [True],
		'spike_device_pars': [rec_device_defaults(device_type='spike_detector', label='single_neuron_spikes')],
		'record_analogs': [True],
		'analog_device_pars': [copy_dict(multimeter, {'label': '{0}_analogs'.format(neuron_type)})],
		'description': {'topology': 'None'}
	})
	# connection_pars = {'description': {
	# 				'connectivity': 'None',
	# 				'plasticity': 'None'}}
	# neuron_pars['description']['synapses'] += ' Non-adapting'
	connection_pars = {}
	# ######################################################################################################################
	# Encoding Parameters
	# ######################################################################################################################
	encoding_pars = set_encoding_defaults(default_set=0)
	analysis_interval = [kernel_pars.transient_t - 100., kernel_pars.transient_t + kernel_pars.sim_time]
	E_spike_times = []  # [540., 689., 900.] #np.arange(analysis_interval[0], analysis_interval[1]-500., 500.)
	I_spike_times = [540., 689., 900.]  # np.arange(analysis_interval[0], analysis_interval[1]-500., 500.)
	encoding_pars.generator.update({
		'N': 2,
		'labels': ['E_input', 'I_input'],
		'models': ['spike_generator', 'spike_generator'],
		'model_pars': [{'spike_times': E_spike_times}, {'spike_times': I_spike_times}],
		'topology': [False, False],
		'topology_pars': [None, None],
	})
	encoding_pars.encoder.update({
		'N': 2,
		'labels': ['E_parrot', 'I_parrot'],
		'n_neurons': [1, 1],
		'neuron_pars': [{'model': 'parrot_neuron'}, {'model': 'parrot_neuron'}],
		'record_spikes': [False, False],
		'spike_device_pars': [{}, {}],
		'record_analogs': [False, False],
		'analog_device_pars': [{}, {}],
		'models': ['parrot_neuron', 'parrot_neuron'],
		'model_pars': [{}, {}],
		'topology': [False, False],
		'topology_dict': [None, None],
	})

	encoding_pars.connectivity.update({
		'connections': [('E_parrot', 'E_input'), ('I_parrot', 'I_input'),
		                (neuron_type, 'E_parrot'), (neuron_type, 'I_parrot'), ],
		'synapse_name': ['Eparr', 'Iparr', 'E_in', 'I_in'],
		'topology_dependent': [False, False, False, False],
		'conn_specs': [{'rule': 'all_to_all'}, {'rule': 'all_to_all'}, {'rule': 'all_to_all'}, {'rule': 'all_to_all'}],
		'syn_specs': [{}, {}, {'receptor_type': 1}, {'receptor_type': 2}],
		'models': ['static_synapse', 'static_synapse', 'multiport_synapse', 'multiport_synapse', ],
		'model_pars': [{}, {}, {'receptor_types': [1., 3.]}, {'receptor_types': [2., 4.]}, ],
		'weight_dist': [1., 1., w, w],
		'delay_dist': [0.1, 0.1, d, d],
		'preset_W': [None, None, None, None]})

	# ##################################################################################################################
	# RETURN dictionary of Parameters dictionaries
	# ==================================================================================================================
	return dict([('kernel_pars', kernel_pars),
	             ('neuron_pars', neuron_pars),
	             ('net_pars', net_pars),
	             ('connection_pars', connection_pars),
	             ('encoding_pars', encoding_pars)])