from defaults.paths import paths
from modules.parameters import ParameterSet, copy_dict
import numpy as np
import sys

"""
single_neuron_fI
- simulate a single neuron, driven by direct current input and determine the neuron's fI curve
- run with single_neuron_dcinput in computations
- debug with run_single_neuron_dcinput script
"""

system_name = 'local'
data_label = 'example1_singleneuron_fI'

# ######################################################################################################################
# PARAMETER RANGE declarations
# ======================================================================================================================
parameter_range = {
	'max_current': [800.]
}


def build_parameters(max_current):
	# ##################################################################################################################
	# DC input parameters
	# ==================================================================================================================
	total_time = 10000.                    # total simulation time [ms]
	analysis_interval = 1000.               # duration of each current step [ms]
	min_current = 0.                        # initial current amplitude [pA]
	# max_current = 600.                      # final current amplitude [pA]

	# specify input times and input amplitudes
	times = list(np.arange(0., total_time, analysis_interval))
	amplitudes = list(np.linspace(min_current, max_current, len(times)))

	# ##################################################################################################################
	# System / Kernel Parameters
	# ##################################################################################################################
	# system-specific parameters (resource allocation, simulation times)
	system_pars = dict(
		nodes=1,
		ppn=8,
		mem=32,
		walltime='00:20:00:00',
		queue='singlenode',
		transient_time=0.,
		sim_time=total_time)

	# seeds for rngs
	N_vp = system_pars['nodes'] * system_pars['ppn']
	np_seed = np.random.randint(1000000000) + 1
	np.random.seed(np_seed)
	msd = np.random.randint(100000000000)

	# main kernel parameter set
	kernel_pars = ParameterSet({
		'resolution': 0.1,
		'sim_time': total_time,
		'transient_t': 0.,
		'data_prefix': data_label,
		'data_path': paths[system_name]['data_path'],
		'mpl_path': paths[system_name]['matplotlib_rc'],
		'overwrite_files': True,
		'print_time': (system_name == 'local'),
		'rng_seeds': range(msd + N_vp + 1, msd + 2 * N_vp + 1),
		'grng_seed': msd + N_vp,
		'total_num_virtual_procs': N_vp,
		'local_num_threads': 16,
		'np_seed': np_seed,

		'system': {
			'local': (system_name == 'local'),
			'system_label': system_name,
			'queueing_system': paths[system_name]['queueing_system'],
			'jdf_template': paths[system_name]['jdf_template'],
			'remote_directory': paths[system_name]['remote_directory'],
			'jdf_fields': {'{{ script_folder }}': '',
			               '{{ nodes }}': str(system_pars['nodes']),
			               '{{ ppn }}': str(system_pars['ppn']),
			               '{{ mem }}': str(system_pars['mem']),
			               '{{ walltime }}': system_pars['walltime'],
			               '{{ queue }}': system_pars['queue'],
			               '{{ computation_script }}': ''}
		}
	})
	# ##################################################################################################################
	# Recording devices
	# ##################################################################################################################
	multimeter = {
		'start': 0.,
		'stop': sys.float_info.max,
		'origin': 0.,
		'interval': 0.1,
		'record_to': ['memory'],
		'label': '',
		'model': 'multimeter',
		'close_after_simulate': False,
		'flush_after_simulate': False,
		'flush_records': False,
		'close_on_reset': True,
		'withtime': True,
		'withgid': True,
		'withweight': False,
		'time_in_steps': False,
		'scientific': False,
		'precision': 3,
		'binary': False,
	}

	# ##################################################################################################################
	# Neuron, Synapse and Network Parameters
	# ##################################################################################################################
	neuron_pars = {
		'AdEx': {
				'model': 'aeif_cond_exp',
				'C_m': 250.0,
				'Delta_T': 2.0,
				'E_L': -70.,
				'E_ex': 0.0,
				'E_in': -75.0,
				'I_e': 0.,
				'V_m': -70.,
				'V_th': -50.,
				'V_reset': -60.0,
				'V_peak': 0.0,
				'a': 4.0,
				'b': 80.5,
				'g_L': 16.7,
				'g_ex': 1.0,
				'g_in': 1.0,
				't_ref': 2.0,
				'tau_minus': 20.,
				'tau_minus_triplet': 200.,
				'tau_w': 144.0,
				'tau_syn_ex': 2.,
				'tau_syn_in': 6.0,
			}}

	multimeter.update({'record_from': ['V_m'], 'record_n': 1})
	pop_names = ['{0}'.format(str(n)) for n in neuron_pars.keys()]
	n_neurons = [1 for _ in neuron_pars.keys()]
	if len(neuron_pars.keys()) > 1:
		neuron_params = [neuron_pars[n] for n in neuron_pars.keys()]
	else:
		neuron_params = [neuron_pars[neuron_pars.keys()[0]]]

	net_pars = ParameterSet({
		'n_populations': len(neuron_pars.keys()),
		'pop_names': pop_names,
		'n_neurons': n_neurons,
		'neuron_pars': neuron_params,
		'randomize_neuron_pars': [{'V_m': (np.random.uniform, {'low': -70., 'high': -50.})}],
		'topology': [False for _ in neuron_pars.keys()],
		'topology_dict': [None for _ in neuron_pars.keys()],
		'record_spikes': [True for _ in neuron_pars.keys()],
		'spike_device_pars': [copy_dict(multimeter, {'model': 'spike_detector'}) for _ in neuron_pars.keys()],
		'record_analogs': [True for _ in neuron_pars.keys()],
		'analog_device_pars': [copy_dict(multimeter, {'record_from': ['V_m'], 'record_n': 1}) for _ in
	                        neuron_pars.keys()],
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

