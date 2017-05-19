__author__ = 'duarte'
import sys
sys.path.insert(0, "../")
from preset import *
import numpy as np

"""
single_neuron_dc_input
- run with single_neuron_dcinput in computations
- debug with script single_neuron_dcinput
"""

run = 'local'
data_label = 'SingleNeuron_DC_homogeneous'
neuron_type = 'E'
heterogeneous = True
total_time = 100000.
analysis_interval = 1000.
min_current = 0.
max_current = 800.
trial = 0

# ######################################################################################################################
# PARAMETER RANGE declarations
# ======================================================================================================================
parameter_range = {

	}


########################################################################################################################
def build_parameters():
	# ######################################################################################################################
	# System / Kernel Parameters
	# ######################################################################################################################
	system = dict(
		nodes=1,
		ppn=8,
		mem=32,
		walltime='00:20:00:00',
		queue='singlenode',
		transient_time=0.,
		sim_time=total_time)

	kernel_pars = set_kernel_defaults(run_type=run, data_label=data_label, **system)

	# ######################################################################################################################
	# Neuron, Synapse and Network Parameters
	# ######################################################################################################################
	if heterogeneous:
		neuron_set = 1.2
	else:
		neuron_set = 1.1
	neuron_pars, net_pars, connection_pars = set_network_defaults(default_set=1, neuron_set=neuron_set,
	                                                              kernel_pars=kernel_pars,
	                                                              random_parameters=randomize_neuron_parameters(
		                                                              heterogeneous))

	# ######################################################################################################################
	# Input Parameters
	# ######################################################################################################################
	times = list(np.arange(0., total_time, analysis_interval))
	amplitudes = list(np.linspace(min_current, max_current, len(times)))

	connections = [('{0}'.format(str(n)), 'DC_Input') for n in net_pars.pop_names]
	n_connections = len(connections)
	encoding_pars = ParameterSet({
		'generator': {
			'N': 1,
			'labels': ['DC_Input'],
			'models': ['step_current_generator'],
			'model_pars': [
				{'start': 0.,
				 'stop': kernel_pars['sim_time'],
				 'origin': 0.,
				 'amplitude_times': times,
				 'amplitude_values': amplitudes}],
			'topology': [False],
			'topology_pars': [None]},

		'connectivity': {
			'connections': connections,
			'topology_dependent': [False for _ in range(n_connections)],
			'conn_specs': [{'rule': 'all_to_all'} for _ in range(n_connections)],
			'syn_specs': [{} for _ in range(n_connections)],
			'models': ['static_synapse' for _ in range(n_connections)],
			'model_pars': [{} for _ in range(n_connections)],
			'weight_dist': [1. for _ in range(n_connections)],
			'delay_dist': [1. for _ in range(n_connections)],
			'preset_W': [None for _ in range(n_connections)]},
	})
	# ##################################################################################################################
	# RETURN dictionary of Parameters dictionaries
	# ==================================================================================================================
	return dict([('kernel_pars', kernel_pars),
	             ('neuron_pars', neuron_pars),
	             ('net_pars', net_pars),
	             ('connection_pars', connection_pars),
	             ('encoding_pars', encoding_pars)])