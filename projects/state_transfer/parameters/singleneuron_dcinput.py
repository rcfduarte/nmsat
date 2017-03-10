__author__ = 'duarte'
from preset import *
from defaults.paths import paths
from modules.parameters import ParameterSet
import numpy as np

"""
single_neuron_dcinput
- simulate a single, point neuron, driven by direct current input (e.g. to determine the neuron's fI curve)
- run with run_single_neuron_dcinput in computations
- debug with run_single_neuron_dcinput script
"""

run = 'local'
data_label = 'amat2_test'


def build_parameters():
	# ##################################################################################################################
	# DC input parameters
	# ==================================================================================================================
	total_time = 1000000.
	analysis_interval = 1000.
	min_current = 200.
	max_current = 800.

	times = list(np.arange(0., total_time, analysis_interval))
	amplitudes = list(np.linspace(min_current, max_current, len(times)))

	# ##################################################################################################################
	# System / Kernel Parameters
	# ##################################################################################################################
	system = dict(
		nodes=1,
		ppn=8,
		mem=32,
		walltime='00:20:00:00',
		queue='singlenode',
		transient_time=0.,
		sim_time=total_time)

	kernel_pars = set_kernel_defaults(run_type=run, data_label=data_label, **system)

	# ##################################################################################################################
	# Neuron, Synapse and Network Parameters
	# ##################################################################################################################
	neuron_pars = {
		'E': {
			# 'model': 'amat2_psc_exp',
			# 'alpha_1': -0.5,
			# 'alpha_2': 0.35,
			# 'beta': -0.3,
			# 'omega': -65.,
			# 'I_e': 91.,
			# 'tau_m': 10.,
			# 'tau_1': 10.,
			# 'tau_2': 200.,
			# 't_ref': 2.,
			# 'C_m': 200.,}}
			'model': 'amat2_psc_exp',
			'alpha_1': 180.,
			'alpha_2': 3.,
			'beta': 0.2,
			'omega': -55.,
			'I_e': 0.,
			'tau_m': 10.,
			'tau_1': 10.,
			'tau_2': 200.,
			't_ref': 2.,
			'C_m': 200., }}

	n_pars = {k: v for k, v in neuron_pars.items() if k != 'description'}
	multimeter = rec_device_defaults(device_type='multimeter')
	multimeter.update({'record_from': ['V_m', 'V_th'], 'record_n': 1})
	pop_names = ['{0}'.format(str(n)) for n in n_pars.keys()]
	n_neurons = [1 for _ in n_pars.keys()]

	if len(neuron_pars.keys()) > 1:
		neuron_params = [n_pars[n] for n in n_pars.keys()]
	else:
		neuron_params = [n_pars[n_pars.keys()[0]]]

	net_pars = ParameterSet({
		'n_populations': len(n_pars.keys()),
		'pop_names': pop_names,
		'n_neurons': n_neurons,
		'neuron_pars': neuron_params,
		# 'randomize_neuron_pars': [{'V_m': (np.random.uniform, {'low': -70., 'high': -50.})}],
		'topology': [False for _ in n_pars.keys()],
		'topology_dict': [None for _ in n_pars.keys()],
		'record_spikes': [True for _ in n_pars.keys()],
		'spike_device_pars': [rec_device_defaults(device_type='spike_detector', label='single_neuron_spikes') for _ in
		                      n_pars.keys()],
		'record_analogs': [True for _ in n_pars.keys()],
		'analog_device_pars': [multimeter for _ in n_pars.keys()],
	})
	neuron_pars = ParameterSet(neuron_pars)

	# ##################################################################################################################
	# Input/Encoding Parameters
	# ##################################################################################################################
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
			'delay_dist': [0.1 for _ in range(n_connections)],
			'preset_W': [None for _ in range(n_connections)]},
	})

	# ##################################################################################################################
	# RETURN dictionary of Parameters dictionaries
	# ==================================================================================================================
	return dict([('kernel_pars', kernel_pars),
	             ('neuron_pars', neuron_pars),
	             ('net_pars', net_pars),
	             ('encoding_pars', encoding_pars)])

# ######################################################################################################################
# PARAMETER RANGE declarations
# ======================================================================================================================
parameter_range = {
	# 'max_current': [200., 500.]
}