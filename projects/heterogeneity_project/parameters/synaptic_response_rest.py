__author__ = 'duarte'
import sys
sys.path.insert(0, "../")
from preset import *
import numpy as np
from modules.input_architect import StochasticGenerator
from auxiliary_fcns import determine_lognormal_parameters

"""
synaptic_responses@rest
- run with synaptic_responses in computations
- debug with script synaptic_response.py
"""

run = 'local'
data_label = 'EE_heterogeneous'
neuron_type = 'E'
connection_type = 'EE'  # 'target source'

heterogeneity = {'synaptic': True, 'neuronal': False, 'structural': False}
n_neurons = 1000
window = [-10., 500.]

# ######################################################################################################################
# PARAMETER RANGE declarations
# ======================================================================================================================
parameter_range = {
	}


########################################################################################################################
def build_parameters():

	# retrieve connectivity data
	_, weights, delays = connection_parameters(heterogeneous=heterogeneity['synaptic'])
	w = weights[connection_type]
	d = delays[connection_type]
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
		sim_time=5000.)
	kernel_pars = set_kernel_defaults(run_type=run, data_label=data_label, **system)
	np.random.seed(kernel_pars['np_seed'])
	kernel_pars.update({'window': window})

	# ######################################################################################################################
	# Neuron/Network Parameters
	# ######################################################################################################################
	randomized_pars = randomize_neuron_parameters(heterogeneous=heterogeneity['neuronal'])
	if heterogeneity['neuronal']:
		neuron_pars = set_neuron_defaults(default_set=1.2)
	else:
		neuron_pars = set_neuron_defaults(default_set=1.1)
		randomized_pars.update({neuron_type: {}})

	# neuron_pars[neuron_type]['E_L'] = -55.

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
	})
	connection_pars = {}
	# ######################################################################################################################
	# Encoding Parameters
	# ######################################################################################################################
	encoding_pars = set_encoding_defaults(default_set=0)
	analysis_interval = [kernel_pars.transient_t - 100., kernel_pars.transient_t + kernel_pars.sim_time]

	# for illustration:
	# gen = StochasticGenerator()
	# spk_times = np.round(gen.poisson_generator(5., t_start=500.1, t_stop=1000., array=True), 1)

	if connection_type[-1] == 'E':
		E_spike_times = np.arange(analysis_interval[0], analysis_interval[1]-500., 500.)
		I_spike_times = []
	else:
		E_spike_times = []
		I_spike_times = np.arange(analysis_interval[0], analysis_interval[1]-500., 500.)

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