import numpy as np
import sys
import inspect

sys.path.append('../../')
from Defaults.Paths import paths

run = 'local'
md = paths[run]['modules_path']
sys.path.append(md)
from modules.parameters import *
import itertools


# def build_parameters():
def build_parameters(total_time, nodes):
	# total_time 			= 100000.
	analysis_interval 	= 1000.
	min_current 		= 0.
	max_current 		= 1000.

	# nodes		= 1
	ppn 		= 8
	mem 		= 32
	N_vp 		= nodes * ppn
	walltime 	= '00:20:00:00'
	queue 	 	= 'singlenode'

	# np_seed = np.random.randint(1000000000)+1
	np_seed = 100
	np.random.seed(np_seed)
	# msd = np.random.randint(100000000000)
	msd = 500

	kernel_pars = {
			'resolution': 0.1,
			'sim_time': total_time,
			'transient_t': 0.,

			'data_prefix': 'DC1',
			'data_path': paths[run]['data_path'],
			'mpl_path': paths[run]['matplotlib_rc'],
			'overwrite_files': True,
			'print_time': True,
			'rng_seeds': range(msd+N_vp+1, msd+2*N_vp+1),
			'grng_seed': msd+N_vp,
			'total_num_virtual_procs': N_vp,
			'local_num_threads': ppn,
			'np_seed': np_seed,

			'system': {
				'local': True,
				'jdf_template': paths[run]['jdf_template'],
				'remote_directory': paths[run]['remote_directory'],
				'jdf_fields': {'{{ script_folder }}': '',
							  '{{ nodes }}': str(nodes),
							  '{{ ppn }}': str(ppn),
							  '{{ mem }}': str(mem),
							  '{{ walltime }}': walltime,
							  '{{ queue }}': queue,
							  '{{ computation_script }}': ''}
			}
	}

	if run == 'cluster':
		assert(not kernel_pars['system']['local']), "Set local to False in kernel_pars!!"

	rec_devices = {
		'start': 0.,
		'stop': sys.float_info.max,
		'origin': 0.,
		'interval': 0.1,
		'record_to': ['memory'],
		'label': '',
		'model': 'spike_detector',
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

	neuron = {
		'model': 'iaf_cond_mtime',
		'C_m': 250.0,
		'E_L': -70.0,
		'I_e': 0.,
		'V_m': -70.0,
		'V_th': -50.0,
		'V_reset': -70.0,
		'a': 0.,
		'b': 0.,
		'g_L': 16.7,
		'rec_bal': [1., 1.],
		'rec_cond': [0.7, 0.8],
		'rec_type': [1., -1.],
		'rec_reversal': [0., -70.],
		'rec_names': ['AMPA', 'GABA_{A}'],
		't_ref': 2.,
		'tau_minus': 20.,
		'tau_minus_triplet': 200.,
		'tau_w': 100.,
		'tau_syn_d1': [2., 6.],
		'tau_syn_d2': [5., 6.],
		'tau_syn_rise': [0.5, 0.3],
		}

	#############################################################################################################
	# NETWORK Parameters
	#============================================================================================================
	net_pars = {
		'n_populations': 1,
		'pop_names': ['singleneuron'],
		'n_neurons': [1],
		'neuron_pars': [copy_dict(neuron)],
		'topology': [False],
		'topology_dict': [None],
		'record_spikes': [True],
		'spike_device_pars': [copy_dict(rec_devices,
										{'model': 'spike_detector',
										 'record_to': ['memory'],
										 'interval': 0.1,
										 'label': 'single_neuron_spikes'})],
		'record_analogs': [True],
		'analog_device_pars': [copy_dict(rec_devices,
										 {'model': 'multimeter',
										  'record_from': ['V_m'],
										  'record_n': 1,
										  'interval': 0.1,
										  'record_to': ['memory']})],
	}

	####################################################################################################################
	# INPUT and ENCODING Parameters
	#============================================================================================================
	times = list(np.arange(0., total_time, analysis_interval))
	amplitudes = list(np.linspace(min_current, max_current, len(times)))

	encoding_pars = {
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
			'connections': [('singleneuron', 'DC_Input')],
			'topology_dependent': [False],
			'conn_specs': [{'rule': 'all_to_all'}],
			'syn_specs': [{}],
			'models': ['static_synapse'],
			'model_pars': [{}],
			'weight_dist': [1.],
			'delay_dist': [1.],
			'preset_W': [None]}
	}

	####################################################################################################################
	# RETURN dictionary of Parameters dictionaries
	#============================================================================================================
	return dict([('kernel_pars', 	kernel_pars),
				 ('rec_devices', 	rec_devices),
				 ('neuron', 		neuron),
				 ('net_pars',	 	net_pars),
				 ('encoding_pars', 	encoding_pars)])

########################################################################################################################
# PARAMETER RANGE declarations

parameter_range = {
	'nodes': [1, 2],
	'total_time': [100000., 1.]
}
