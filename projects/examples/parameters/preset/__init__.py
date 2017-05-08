import numpy as np
import sys
from modules.parameters import ParameterSet, copy_dict
from defaults.paths import paths


def set_kernel_defaults(run_type='local', data_label='', **system_pars):
	"""
	Return pre-defined kernel parameters dictionary
	:param default_set:
	:return:
	"""
	keys = ['nodes', 'ppn', 'mem', 'walltime', 'queue', 'sim_time', 'transient_time']
	if not np.mean(np.sort(system_pars.keys()) == np.sort(keys)).astype(bool):
		raise TypeError("system parameters dictionary must contain the following keys {0}".format(str(keys)))

	N_vp = system_pars['nodes'] * system_pars['ppn']
	np_seed = np.random.randint(1000000000) + 1
	np.random.seed(np_seed)
	msd = np.random.randint(100000000000)

	kernel_pars = {
		'resolution': 0.1,
		'sim_time': system_pars['sim_time'],
		'transient_t': system_pars['transient_time'],
		'data_prefix': data_label,
		'data_path': paths[run_type]['data_path'],
		'mpl_path': paths[run_type]['matplotlib_rc'],
		'overwrite_files': True,
		'print_time': (run_type == 'local'),
		'rng_seeds': range(msd + N_vp + 1, msd + 2 * N_vp + 1),
		'grng_seed': msd + N_vp,
		'total_num_virtual_procs': N_vp,
		'local_num_threads': system_pars['ppn'],
		'np_seed': np_seed,

		'system': {
			'local': (run_type == 'local'),
			'system_label': run_type,
			'queueing_system': paths[run_type]['queueing_system'],
			'jdf_template': paths[run_type]['jdf_template'],
			'remote_directory': paths[run_type]['remote_directory'],
			'jdf_fields': {'{{ script_folder }}': '',
			               '{{ nodes }}': str(system_pars['nodes']),
			               '{{ ppn }}': str(system_pars['ppn']),
			               '{{ mem }}': str(system_pars['mem']),
			               '{{ walltime }}': system_pars['walltime'],
			               '{{ queue }}': system_pars['queue'],
			               '{{ computation_script }}': ''}
		}
	}
	return ParameterSet(kernel_pars)


def rec_device_defaults(start=0., stop=sys.float_info.max, resolution=0.1, record_to='memory',
                        device_type='spike_detector', label=''):
	"""
	Standard device parameters
	:param default_set:
	:return:
	"""
	rec_devices = {
		'start': start,
		'stop': stop,
		'origin': 0.,
		'interval': resolution,
		'record_to': [record_to],
		'label': label,
		'model': device_type,
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
	return ParameterSet(rec_devices)