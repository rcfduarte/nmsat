__author__ = 'duarte'
import numpy as np
import sys
from Modules.parameters import *
import scipy.stats as st


def set_kernel_defaults(default_set=0, run_type='local', data_label='', project_label=None, **system_pars):
	"""
	Return pre-defined kernel parameters dictionary
	:param default_set:
	:return:
	"""
	from Experiments.ParameterSets.Preset.Paths import paths
	if run_type == 'local':
		run = True
	else:
		run = False
	if project_label is not None:
		paths[run_type]['report_templates_path'] += project_label + '/'

	if default_set == 0:
		keys = ['nodes', 'ppn', 'mem', 'walltime', 'queue']
		if not np.mean(np.sort(system_pars.keys()) == np.sort(keys)).astype(bool):
			raise TypeError("system parameters dictionary must contain the following keys {0}".format(str(keys)))

		N_vp = system_pars['nodes'] * system_pars['ppn']
		np_seed = np.random.randint(1000000000) + 1
		np.random.seed(np_seed)
		msd = np.random.randint(100000000000)

		kernel_pars = {
			'resolution': 0.1,
			'sim_time': 0.,
			'transient_t': 0.,
			'data_prefix': data_label,
			'data_path': paths[run_type]['data_path'],
			'mpl_path': paths[run_type]['matplotlib_rc'],
			'overwrite_files': True,
			'print_time': run,
			'rng_seeds': range(msd + N_vp + 1, msd + 2 * N_vp + 1),
			'grng_seed': msd + N_vp,
			'total_num_virtual_procs': N_vp,
			'local_num_threads': system_pars['ppn'],
			'np_seed': np_seed,

			'system': {
				'local': run,
				'system_label': run_type,
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
	elif default_set == 1:
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
			'print_time': run,
			'rng_seeds': range(msd + N_vp + 1, msd + 2 * N_vp + 1),
			'grng_seed': msd + N_vp,
			'total_num_virtual_procs': N_vp,
			'local_num_threads': system_pars['ppn'],
			'np_seed': np_seed,

			'system': {
				'local': run,
				'system_label': run_type,
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
	elif default_set == 3:
		keys = ['nodes', 'ppn', 'mem', 'walltime', 'queue', 'sim_time', 'transient_time']
		if not np.mean(np.sort(system_pars.keys()) == np.sort(keys)).astype(bool):
			raise TypeError("system parameters dictionary must contain the following keys {0}".format(str(keys)))

		N_vp = system_pars['nodes'] * system_pars['ppn']
		np_seed = np.random.randint(1000000000) + 1
		np.random.seed(np_seed)
		msd = np.random.randint(100000000000)

		kernel_pars = {
			'resolution': 0.01,
			'sim_time': system_pars['sim_time'],
			'transient_t': system_pars['transient_time'],
			'data_prefix': data_label,
			'data_path': paths[run_type]['data_path'],
			'mpl_path': paths[run_type]['matplotlib_rc'],
			'overwrite_files': True,
			'print_time': run,
			'rng_seeds': range(msd + N_vp + 1, msd + 2 * N_vp + 1),
			'grng_seed': msd + N_vp,
			'total_num_virtual_procs': N_vp,
			'local_num_threads': system_pars['ppn'],
			'np_seed': np_seed,

			'system': {
				'local': run,
				'system_label': run_type,
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


def rec_device_defaults(default_set=0, start=0., stop=sys.float_info.max, resolution=0.1, record_to='memory',
                        device_type='spike_detector', label=''):
	"""

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


def set_neuron_defaults(default_set=1):
	# TODO extend the commentary on this?
	"""
	Collections of single neuron ParameterSets

	:param default_set: (int) Load default set:
	------------------
	1) iaf_cond_mtime, AMPA+GABAa (heterogeneous syn pars for E/I neurons), fixed neuron pars
	2) iaf_cond_mtime, AMPA+GABAa (hetereogeneous syn pars for E/I neurons), fixed neuron pars
	3) iaf_cond_mtime, AMPA+NMDA+GABAa+GABAb (homogeneous E/I pars), fixed neuron pars
	4) iaf_cond_exp, AMPA+GABAa (homogeneous syn pars for E/I neurons), fixed neuron pars
	:return: ParameterSet
	"""
	if default_set == 1.1:
		print ("\nLoading Default Neuron Set 1.1 - iaf_cond_mtime, fixed voltage threshold, fixed absolute refractory \
		        time, Fast synapses (AMPA, GABAa), heterogeneous parameters")
		neuron_pars = {
			'description': {
				'neurons': 'Leaky integrate-and-fire, fixed voltage threshold, fixed absolute refractory time',
				'synapses': r'Fast synapses ($\mathrm{AMPA}$, $\mathrm{GABA}_{A}$), heterogeneous parameters'},
			'E': {
				'model': 'iaf_cond_mtime',
				'C_m': 250.0,
				'E_L': -70.0,
				'I_e': 0.,
				'V_m': -70.0,
				'V_th': -50.0,
				'V_reset': -60.0,
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
			},
			'I': {
				'model': 'iaf_cond_mtime',
				'C_m': 250.0,
				'E_L': -70.0,
				'I_e': 0.,
				'V_m': -70.0,
				'V_th': -50.0,
				'V_reset': -60.0,
				'a': 0.,
				'b': 0.,
				'g_L': 16.7,
				'rec_bal': [1., 1.],
				'rec_cond': [1.0, 0.8],
				'rec_type': [1., -1.],
				'rec_reversal': [0., -70.],
				'rec_names': ['AMPA', 'GABA_{A}'],
				't_ref': 2.,
				'tau_minus': 20.,
				'tau_minus_triplet': 200.,
				'tau_w': 100.,
				'tau_syn_d1': [1., 3.],
				'tau_syn_d2': [5., 6.],
				'tau_syn_rise': [0.25, 0.3],
			}
		}
	elif default_set == 1.2:
		print ("\nLoading Default Neuron Set 1.2 - iaf_cond_mtime, fixed voltage threshold, fixed absolute refractory \
		        time, Fast synapses (AMPA, GABAa), homogeneous parameters")
		neuron_pars = {
			'description': {
				'neurons': 'Leaky integrate-and-fire, fixed voltage threshold, fixed absolute refractory time',
				'synapses': r'Fast synapses ($\mathrm{AMPA}$, $\mathrm{GABA}_{A}$), homogeneous parameters'},
			'E': {
				'model': 'iaf_cond_mtime',
				'C_m': 250.0,
				'E_L': -70.0,
				'I_e': 0.,
				'V_m': -70.0,
				'V_th': -50.0,
				'V_reset': -60.0,
				'a': 1.8,
				'b': 16.,
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
			},
			'I1': {
				'model': 'iaf_cond_mtime',
				'C_m': 250.0,
				'E_L': -70.0,
				'I_e': 0.,
				'V_m': -70.0,
				'V_th': -50.0,
				'V_reset': -60.0,
				'a': 0.5,
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
			},
			'I2': {
				'model': 'iaf_cond_mtime',
				'C_m': 250.0,
				'E_L': -70.0,
				'I_e': 0.,
				'V_m': -70.0,
				'V_th': -50.0,
				'V_reset': -60.0,
				'a': 0.,
				'b': 6.0,
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
		}

	elif default_set == 1.3:
		print ("\nLoading Default Neuron Set 1.3 - iaf_cond_mtime, fixed voltage threshold, fixed absolute refractory "
			   "time, "
			   "Fast and slow synapses (AMPA, GABAa, NMDA, GABAb),"
			   " homogeneous parameters")
		neuron_pars = {
			'description': {
				'neurons': 'Leaky integrate-and-fire, fixed voltage threshold, fixed absolute refractory time',
				'synapses': r'Fast and slow synapses ($\mathrm{AMPA}$, $\mathrm{GABA}_{A}$, $\mathrm{NMDA}$, '
				            '$\mathrm{GABA}_{B}$), homogeneous parameters'},
			'E': {
				'model': 'iaf_cond_mtime',
				'C_m': 250.0,
				'E_L': -70.0,
				'I_e': 0.,
				'V_m': -70.0,
				'V_th': -50.0,
				'V_reset': -60.0,
				'a': 1.8,
				'b': 16.,
				'g_L': 16.7,
				'rec_bal': [1., 1., 0., 0.8],
				'rec_cond': [0.7, 0.8, 1.5, 1.],
				'rec_type': [1., -1., 2., -1.],
				'rec_reversal': [0., -70., 0., -95.],
				'rec_names': ['AMPA', 'GABA_{A}', 'NMDA', 'GABA_{B}'],
				't_ref': 2.,
				'tau_minus': 20.,
				'tau_minus_triplet': 200.,
				'tau_w': 100.,
				'tau_syn_d1': [2., 6., 150., 700.],
				'tau_syn_d2': [5., 6., 150., 200.],
				'tau_syn_rise': [0.5, 0.3, 1.5, 30.],
			},
			'I1': {
				'model': 'iaf_cond_mtime',
				'C_m': 250.0,
				'E_L': -70.0,
				'I_e': 0.,
				'V_m': -70.0,
				'V_th': -50.0,
				'V_reset': -60.0,
				'a': 0.5,
				'b': 0.,
				'g_L': 16.7,
				'rec_bal': [1., 1., 0., 0.8],
				'rec_cond': [0.7, 0.8, 1.5, 1.],
				'rec_type': [1., -1., 2., -1.],
				'rec_reversal': [0., -70., 0., -95.],
				'rec_names': ['AMPA', 'GABA_{A}', 'NMDA', 'GABA_{B}'],
				't_ref': 2.,
				'tau_minus': 20.,
				'tau_minus_triplet': 200.,
				'tau_w': 100.,
				'tau_syn_d1': [2., 6., 150., 700.],
				'tau_syn_d2': [5., 6., 150., 200.],
				'tau_syn_rise': [0.5, 0.3, 1.5, 30.],
			},
			'I2': {
				'model': 'iaf_cond_mtime',
				'C_m': 250.0,
				'E_L': -70.0,
				'I_e': 0.,
				'V_m': -70.0,
				'V_th': -50.0,
				'V_reset': -60.0,
				'a': 0.,
				'b': 6.0,
				'g_L': 16.7,
				'rec_bal': [1., 1., 0., 0.8],
				'rec_cond': [0.7, 0.8, 1.5, 1.],
				'rec_type': [1., -1., 2., -1.],
				'rec_reversal': [0., -70., 0., -95.],
				'rec_names': ['AMPA', 'GABA_{A}', 'NMDA', 'GABA_{B}'],
				't_ref': 2.,
				'tau_minus': 20.,
				'tau_minus_triplet': 200.,
				'tau_w': 100.,
				'tau_syn_d1': [2., 6., 150., 700.],
				'tau_syn_d2': [5., 6., 150., 200.],
				'tau_syn_rise': [0.5, 0.3, 1.5, 30.],
			}
		}

	elif default_set == 1.4:
		print ("\nLoading Default Neuron Set 1.4 - iaf_cond_mtime, Fast and slow synapses (AMPA, GABAa, NMDA, GABAb), "
		       "heterogeneous neurons parameters")
		neuron_pars = {
			'description': {
				'neurons': 'Leaky integrate-and-fire, random voltage threshold, absolute refractory time',
				'synapses': r'Fast and slow synapses ($\mathrm{AMPA}$, $\mathrm{GABA}_{A}$, $\mathrm{NMDA}$, '
				            '$\mathrm{GABA}_{B}$), homogeneous parameters'},
			'E': {
				'model': 'iaf_cond_mtime',
				'C_m': 250.0,
				'E_L': -70.0,
				'I_e': 0.,
				'V_m': -70.0,
				'V_th': -50.0,
				'V_reset': -60.0,
				'a': 4.,
				'b': 30.,
				'g_L': 16.7,
				'rec_bal': [1., 1., 0., 0.8],
				'rec_cond': [0.5, 0.8, 0.4, 0.2],
				'rec_type': [1., -1., 2., -1.],
				'rec_reversal': [0., -75., 0., -90.],
				'rec_names': ['AMPA', 'GABA_{A}', 'NMDA', 'GABA_{B}'],
				't_ref': 2.,
				'tau_minus': 20.,
				'tau_minus_triplet': 200.,
				'tau_w': 100.,
				'tau_syn_d1': [2., 6., 0.1, 200.],  # fast
				'tau_syn_d2': [0.1, 0.1, 100., 600.],  # slow
				'tau_syn_rise': [0.3, 0.25, 1., 30.],
			},
			'I1': {
				'model': 'iaf_cond_mtime',
				'C_m': 250.0,
				'E_L': -70.0,
				'I_e': 0.,
				'V_m': -70.0,
				'V_th': -50.0,
				'V_reset': -60.0,
				'a': 0.,
				'b': 0.,
				'g_L': 16.7,
				'rec_bal': [1., 1., 0., 0.8],
				'rec_cond': [1.0, 0.9, 0.1, 0.05],
				'rec_type': [1., -1., 2., -1.],
				'rec_reversal': [0., -75., 0., -90.],
				'rec_names': ['AMPA', 'GABA_{A}', 'NMDA', 'GABA_{B}'],
				't_ref': 2.,
				'tau_minus': 20.,
				'tau_minus_triplet': 200.,
				'tau_w': 100.,
				'tau_syn_d1': [0.7, 2.5, 0.1, 50.],
				'tau_syn_d2': [0.1, 0.1, 100., 400.],
				'tau_syn_rise': [0.1, 0.1, 1., 25.],
			},
			'I2': {
				'model': 'iaf_cond_mtime',
				'C_m': 250.0,
				'E_L': -70.0,
				'I_e': 0.,
				'V_m': -70.0,
				'V_th': -50.0,
				'V_reset': -60.0,
				'a': 2.,
				'b': 10.,
				'g_L': 16.7,
				'rec_bal': [1., 1., 0., 0.8],
				'rec_cond': [0.7, 0.8, 0.1, 0.2],
				'rec_type': [1., -1., 2., -1.],
				'rec_reversal': [0., -75., 0., -90.],
				'rec_names': ['AMPA', 'GABA_{A}', 'NMDA', 'GABA_{B}'],
				't_ref': 2.,
				'tau_minus': 20.,
				'tau_minus_triplet': 200.,
				'tau_w': 100.,
				'tau_syn_d1': [1.8, 5., 0.1, 150.],
				'tau_syn_d2': [0.1, 0.1, 100., 500.],
				'tau_syn_rise': [0.2, 0.2, 1., 25.],
			}
		}

	elif default_set == 1.5:
		print ("\nLoading Default Neuron Set 1.5 - iaf_cond_mtime, fixed voltage threshold, fixed absolute refractory "
		       "time, Fast synapses (AMPA, GABAa), randomized parameters")
		neuron_pars = {
			'description': {
				'neurons': 'Leaky integrate-and-fire, fixed voltage threshold, fixed absolute refractory time',
				'synapses': r'Fast synapses ($\mathrm{AMPA}$, $\mathrm{GABA}_{A}$), homogeneous parameters'},
			'E': {
				'model': 'iaf_cond_mtime',
				'C_m': 250.0,
				'E_L': -70.0,
				'I_e': 0.,
				'V_m': -70.0,
				'V_th': -50.0,
				'V_reset': -60.0,
				'a': 4.,
				'b': 30.,
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
			},
			'I1': {
				'model': 'iaf_cond_mtime',
				'C_m': 250.0,
				'E_L': -70.0,
				'I_e': 0.,
				'V_m': -70.0,
				'V_th': -50.0,
				'V_reset': -60.0,
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
			},
			'I2': {
				'model': 'iaf_cond_mtime',
				'C_m': 250.0,
				'E_L': -70.0,
				'I_e': 0.,
				'V_m': -70.0,
				'V_th': -50.0,
				'V_reset': -60.0,
				'a': 2.,
				'b': 10.,
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
		}

	elif default_set == 2.1:
		print ("\nLoading Default Neuron Set 2.1 - iaf_cond_exp, fixed voltage threshold, fixed absolute refractory "
		       "time, Fast, exponential synapses (AMPA, GABAa), homogeneous parameters")
		neuron_pars = {
			'description': {
				'neurons': 'Leaky integrate-and-fire, fixed voltage threshold, fixed absolute refractory time',
				'synapses': r'Fast, exponential synapses ($\mathrm{AMPA}$, $\mathrm{GABA}_{A}$, homogeneous parameters'},
			'E': {
				'model': 'iaf_cond_exp',
				'C_m': 289.5,
				'E_L': -70.0,
				'I_e': 0.,
				'V_m': -70.0,
				'V_th': -57.0,
				'V_reset': -60.0,
				'g_L': 28.95,
				't_ref': 2.,
				'tau_minus': 20.,
				'tau_minus_triplet': 200.,
				'E_ex': 0.,
				'g_ex': 1.,
				'tau_syn_ex': 5.,
				'E_in': -75.,
				'g_in': 1.,
				'tau_syn_in': 10.
			},
			'I': {
				'model': 'iaf_cond_exp',
				'C_m': 289.5,
				'E_L': -70.0,
				'I_e': 0.,
				'V_m': -70.0,
				'V_th': -57.0,
				'V_reset': -60.0,
				'g_L': 28.95,
				't_ref': 2.,
				'tau_minus': 20.,
				'tau_minus_triplet': 200.,
				'E_ex': 0.,
				'g_ex': 1.,
				'tau_syn_ex': 5.,
				'E_in': -75.,
				'g_in': 1.,
				'tau_syn_in': 10.
			}
		}
	elif default_set == 2.2:
		print ("\nLoading Default Neuron Set 2.2 - iaf_cond_exp_sfa_rr, fixed voltage threshold, dynamic refractory "
		       "time spike frequency adaptation, Fast, conductance-based exponential synapses, homogeneous parameters")
		neuron_pars = {
			'description': {
				'neurons': 'Leaky integrate-and-fire, sfa, dynamic refractory time',
				'synapses': r'Fast, exponential, conductance-based synapses (homogeneous parameters'},
			'E': {
				'model': 'iaf_cond_exp_sfa_rr',
				'C_m': 289.5,
				'E_L': -70.0,
				'V_reset': -60.0,
				'V_th': -57.,
				't_ref': 0.5,
				'E_ex': 0.,
				'g_ex': 1.,
				'g_in': 1.,
				'E_in': -75.,
				'g_L': 28.95,
				'tau_syn_ex': 5.,
				'tau_syn_in': 10.,
				'q_sfa': 14.48,
				'q_rr': 3214.,
				'tau_sfa': 110.,
				'tau_rr': 1.97,
				'E_sfa': -70.,
				'E_rr': -70.,
				'I_e': 0.
			},
			'I': {
				'model': 'iaf_cond_exp_sfa_rr',
				'C_m': 289.5,
				'E_L': -70.0,
				'V_reset': -60.0,
				'V_th': -57.,
				't_ref': 0.5,
				'E_ex': 0.,
				'g_ex': 1.,
				'g_in': 1.,
				'E_in': -75.,
				'g_L': 28.95,
				'tau_syn_ex': 5.,
				'tau_syn_in': 10.,
				'q_sfa': 14.48,
				'q_rr': 3214.,
				'tau_sfa': 110.,
				'tau_rr': 1.97,
				'E_sfa': -70.,
				'E_rr': -70.,
				'I_e': 0.
			},
		}

	elif default_set == 2.3:
		print ("\nLoading Default Neuron Set 1.3 - iaf_cond_mtime, fixed voltage threshold, fixed absolute refractory "
		       "time, Fast and slow synapses (AMPA, GABAa, NMDA, GABAb), homogeneous parameters")
		neuron_pars = {
			'description': {
				'neurons': 'Leaky integrate-and-fire, fixed voltage threshold, fixed absolute refractory time',
				'synapses': r'Fast and slow synapses ($\mathrm{AMPA}$, $\mathrm{GABA}_{A}$, $\mathrm{NMDA}$, '
				            '$\mathrm{GABA}_{B}$), homogeneous parameters'},
			'E': {
				'model': 'iaf_cond_mtime',
				'C_m': 289.5,
				'E_L': -70.0,
				'I_e': 0.,
				'V_m': -70.0,
				'V_th': -57.0,
				'V_reset': -60.0,
				'a': 0.,
				'b': 0.,
				'g_L': 28.95,
				'rec_bal': [1., 1., 0., 0.8],
				'rec_cond': [0.7, 0.8, 1.5, 1.],
				'rec_type': [1., -1., 2., -1.],
				'rec_reversal': [0., -70., 0., -95.],
				'rec_names': ['AMPA', 'GABA_{A}', 'NMDA', 'GABA_{B}'],
				't_ref': 2.,
				'tau_minus': 20.,
				'tau_minus_triplet': 200.,
				'tau_w': 100.,
				'tau_syn_d1': [2., 6., 150., 700.],
				'tau_syn_d2': [5., 6., 150., 200.],
				'tau_syn_rise': [0.5, 0.3, 1.5, 30.],
			},
			'I': {
				'model': 'iaf_cond_mtime',
				'C_m': 289.5,
				'E_L': -70.0,
				'I_e': 0.,
				'V_m': -70.0,
				'V_th': -57.0,
				'V_reset': -60.0,
				'a': 0.,
				'b': 0.,
				'g_L': 28.95,
				'rec_bal': [1., 1., 0., 0.8],
				'rec_cond': [0.7, 0.8, 1.5, 1.],
				'rec_type': [1., -1., 2., -1.],
				'rec_reversal': [0., -70., 0., -95.],
				'rec_names': ['AMPA', 'GABA_{A}', 'NMDA', 'GABA_{B}'],
				't_ref': 2.,
				'tau_minus': 20.,
				'tau_minus_triplet': 200.,
				'tau_w': 100.,
				'tau_syn_d1': [2., 6., 150., 700.],
				'tau_syn_d2': [5., 6., 150., 200.],
				'tau_syn_rise': [0.5, 0.3, 1.5, 30.],
			},
		}

	elif default_set == 4:
		print ("\nLoading Default Neuron Set 4 - iaf_cond_exp, fixed voltage threshold, fixed absolute refractory "
		       "time, Fast, exponential synapses (AMPA, GABAa), homogeneous parameters")
		neuron_pars = {
			'description': {
				'neurons': 'Leaky integrate-and-fire, fixed voltage threshold, fixed absolute refractory time',
				'synapses': r'Fast, exponential synapses ($\mathrm{AMPA}$, $\mathrm{GABA}_{A}$, homogeneous parameters'},
			'E': {
				'model': 'iaf_cond_exp',
				'C_m': 250.0,
				'E_L': -70.0,
				'I_e': 0.,
				'V_m': -70.0,
				'V_th': -50.0,
				'V_reset': -60.0,
				'g_L': 16.7,
				't_ref': 5.,
				'tau_minus': 20.,
				'tau_minus_triplet': 200.,
				'E_ex': 0.,
				'tau_syn_ex': 5.,
				'E_in': -80.,
				'tau_syn_in': 10.
			},
			'I': {
				'model': 'iaf_cond_exp',
				'C_m': 250.0,
				'E_L': -70.0,
				'I_e': 0.,
				'V_m': -70.0,
				'V_th': -50.0,
				'V_reset': -60.0,
				'g_L': 16.7,
				't_ref': 5.,
				'tau_minus': 20.,
				'tau_minus_triplet': 200.,
				'E_ex': 0.,
				'tau_syn_ex': 5.,
				'E_in': -80.,
				'tau_syn_in': 10.
			}
		}
	elif default_set == 5:
		print ("\nLoading Default Neuron Set 5 - amat2_psc_exp, fixed voltage threshold, fixed absolute refractory "
		       "time, Fast, exponential synapses, homogeneous parameters")
		neuron_pars = {
			'description': {
				'neurons': 'Leaky integrate-and-fire, Multi-Timescale Adaptive threshold, fixed absolute '
				           'refractory time',
				'synapses': r'Fast, exponential, current-based synapses (homogeneous parameters'},
			'E': {
				'model': 'amat2_psc_exp',
				'C_m': 200.0,
				'E_L': -70.0,
				'I_e': 0.,
				'V_m': -70.0,
				'V_th_v': 0.0,
				'alpha_1': 180.0,
				'alpha_2': 3.0,
				'beta': 0.2,
				'omega': -55.0,
				't_ref': 2.,
				'tau_1': 10.0,
				'tau_2': 200.0,
				'tau_m': 10.0,
				'tau_syn_ex': 3.0,
				'tau_syn_in': 7.0,
				'tau_v': 5.0
			},
			'I': {
				'model': 'amat2_psc_exp',
				'C_m': 200.0,
				'E_L': -70.0,
				'I_e': 0.,
				'V_m': -70.0,
				'V_th_v': 0.0,
				'alpha_1': 180.0,
				'alpha_2': 3.0,
				'beta': 0.2,
				'omega': -55.0,
				't_ref': 2.,
				'tau_1': 10.0,
				'tau_2': 200.0,
				'tau_m': 10.0,
				'tau_syn_ex': 3.0,
				'tau_syn_in': 7.0,
				'tau_v': 5.0
			},
		}
	elif default_set == 5.1:
		print ("\nLoading Default Neuron Set 5.1 - amat2_psc_exp, fixed voltage threshold, fixed absolute refractory "
		       "time, Fast, exponential synapses, homogeneous parameters")
		neuron_pars = {
			'description': {
				'neurons': 'Leaky integrate-and-fire, Multi-Timescale Adaptive threshold, fixed absolute '
				           'refractory time',
				'synapses': r'Fast, exponential, current-based synapses (homogeneous parameters'},
			'E1': {
				'model': 'amat2_psc_exp',
				'C_m': 200.0,
				'E_L': -70.0,
				'I_e': 0.,
				'V_m': -70.0,
				'V_th_v': 0.0,
				'alpha_1': 180.0,
				'alpha_2': 3.0,
				'beta': 0.2,
				'omega': -55.0,
				't_ref': 2.,
				'tau_1': 10.0,
				'tau_2': 200.0,
				'tau_m': 10.0,
				'tau_syn_ex': 3.0,
				'tau_syn_in': 7.0,
				'tau_v': 5.0
			},
			'E2': {
				'model': 'amat2_psc_exp',
				'C_m': 200.0,
				'E_L': -70.0,
				'I_e': 0.,
				'V_m': -70.0,
				'V_th_v': 0.0,
				'alpha_1': 180.0,
				'alpha_2': 3.0,
				'beta': 0.2,
				'omega': -55.0,
				't_ref': 2.,
				'tau_1': 10.0,
				'tau_2': 200.0,
				'tau_m': 10.0,
				'tau_syn_ex': 3.0,
				'tau_syn_in': 7.0,
				'tau_v': 5.0
			},
			'I1': {
				'model': 'amat2_psc_exp',
				'C_m': 200.0,
				'E_L': -70.0,
				'I_e': 0.,
				'V_m': -70.0,
				'V_th_v': 0.0,
				'alpha_1': 180.0,
				'alpha_2': 3.0,
				'beta': 0.2,
				'omega': -55.0,
				't_ref': 2.,
				'tau_1': 10.0,
				'tau_2': 200.0,
				'tau_m': 10.0,
				'tau_syn_ex': 3.0,
				'tau_syn_in': 7.0,
				'tau_v': 5.0
			},
			'I2': {
				'model': 'amat2_psc_exp',
				'C_m': 200.0,
				'E_L': -70.0,
				'I_e': 0.,
				'V_m': -70.0,
				'V_th_v': 0.0,
				'alpha_1': 180.0,
				'alpha_2': 3.0,
				'beta': 0.2,
				'omega': -55.0,
				't_ref': 2.,
				'tau_1': 10.0,
				'tau_2': 200.0,
				'tau_m': 10.0,
				'tau_syn_ex': 3.0,
				'tau_syn_in': 7.0,
				'tau_v': 5.0
			},
		}

	elif default_set == 6:
		print ("\nLoading Default Neuron Set 6 - iaf_psc_exp_ps, fixed voltage threshold, fixed absolute refractory "
		       "time, Fast, current-based exponential synapses, homogeneous parameters")
		neuron_pars = {
			'description': {
				'neurons': 'Leaky integrate-and-fire, fixed firing threshold, fixed absolute refractory time',
				'synapses': r'Fast, exponential, current-based synapses (homogeneous parameters'},
			'E': {
				'model': 'iaf_psc_exp_ps',
				'C_m': 250.0,
				'E_L': 0.0,
				'V_reset': 0.0,
				'V_th': 15.,
				't_ref': 2.,
				'tau_syn_ex': 2.,
				'tau_syn_in': 2.,
				'tau_m': 20.
			},
			'I': {
				'model': 'iaf_psc_exp_ps',
				'C_m': 250.0,
				'E_L': 0.0,
				'V_reset': 0.0,
				'V_th': 15.,
				't_ref': 2.,
				'tau_syn_ex': 2.,
				'tau_syn_in': 2.,
				'tau_m': 20.
			},
		}
	elif default_set == 7.1:
		print ("\nLoading Default Neuron Set 7.1 - aeif_cond_exp, fixed voltage threshold, fixed absolute refractory "
		       "time, Fast, conductance-based exponential synapses")
		neuron_pars = {
			'description': {
				'neurons': 'Adaptive Leaky integrate-and-fire, fixed firing threshold, fixed absolute refractory time',
				'synapses': r'Fast, exponential, conductance-based synapses'},
			'E': {
				'model': 'aeif_cond_exp',
				'C_m': 450.0,
				'Delta_T': 2.0,
				'E_L': -70.6,
				'E_ex': 0.0,
				'E_in': -75.0,
				'I_e': 0.,
				'V_m': -70.6,
				'V_th': -50.4,
				'V_reset': -70.6,
				'V_peak': 20.0,
				'a': 4.0,
				'b': 80.5,
				'g_L': 25.,
				'g_ex': 1.0,
				'g_in': 1.2,
				't_ref': 2.0,
				'tau_minus': 20.,
				'tau_minus_triplet': 200.,
				'tau_w': 144.0,
				'tau_syn_ex': 2.,
				'tau_syn_in': 6.0,
			},
			'I':  {
				'model': 'aeif_cond_exp',
				'C_m': 450.0,
				'Delta_T': 2.0,
				'E_L': -70.6,
				'E_ex': 0.0,
				'E_in': -75.0,
				'I_e': 0.,
				'V_m': -70.6,
				'V_th': -50.4,
				'V_reset': -70.6,
				'V_peak': 20.0,
				'a': 4.0,
				'b': 80.5,
				'g_L': 25.,
				'g_ex': 1.0,
				'g_in': 1.2,
				't_ref': 2.0,
				'tau_minus': 20.,
				'tau_minus_triplet': 200.,
				'tau_w': 144.0,
				'tau_syn_ex': 2.,
				'tau_syn_in': 6.0,
			},
		}
	elif default_set == 7.2:
		print ("\nLoading Default Neuron Set 7.2 - aeif_cond_exp, fixed voltage threshold, fixed absolute refractory "
		       "time, Fast, conductance-based exponential synapses")
		neuron_pars = {
			'description': {
				'neurons': 'Adaptive Leaky integrate-and-fire, fixed firing threshold, fixed absolute refractory time',
				'synapses': r'Fast, exponential, conductance-based synapses'},
			'E': {
				'model': 'aeif_cond_exp',
				'C_m': 281.0,
				'Delta_T': 2.0,
				'E_L': -70.6,
				'E_ex': 0.0,
				'E_in': -75.0,
				'I_e': 0.,
				'V_m': -70.6,
				'V_th': -50.4,
				'V_reset': -60.0,
				'V_peak': 0.0,
				'a': 4.0,
				'b': 80.5,
				'g_L': 30.,
				'g_ex': 1.0,
				'g_in': 1.2,
				't_ref': 2.0,
				'tau_minus': 20.,
				'tau_minus_triplet': 200.,
				'tau_w': 144.0,
				'tau_syn_ex': 2.,
				'tau_syn_in': 6.0,
			},
			'I':  {
				'model': 'aeif_cond_exp',
				'C_m': 281.0,
				'Delta_T': 2.0,
				'E_L': -70.6,
				'E_ex': 0.0,
				'E_in': -75.0,
				'I_e': 0.,
				'V_m': -70.6,
				'V_th': -50.4,
				'V_reset': -60.0,
				'V_peak': 0.0,
				'a': 4.0,
				'b': 80.5,
				'g_L': 30.,
				'g_ex': 1.0,
				'g_in': 1.2,
				't_ref': 2.0,
				'tau_minus': 20.,
				'tau_minus_triplet': 200.,
				'tau_w': 144.0,
				'tau_syn_ex': 2.,
				'tau_syn_in': 6.0,
			},
		}
	elif default_set == 7.3:
		print ("\nLoading Default Neuron Set 7.3 - aeif_cond_exp, fixed voltage threshold, fixed absolute refractory "
		       "time, Fast, conductance-based exponential synapses")
		neuron_pars = {
			'description': {
				'neurons': 'Adaptive Leaky integrate-and-fire, fixed firing threshold, fixed absolute refractory time',
				'synapses': r'Fast, exponential, conductance-based synapses'},
			'E': {
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
			},
			'I':  {
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
			},
		}
	elif default_set == 8.1:
		print ("\nLoading Default Neuron Set 8.1 - iaf_cond_exp_sfa_rr, fixed voltage threshold, dynamic refractory "
			   "time "
			   "spike frequency adaptation, Fast, conductance-based exponential synapses, homogeneous parameters")
		neuron_pars = {
			'description': {
				'neurons': 'Leaky integrate-and-fire, sfa, dynamic refractory time',
				'synapses': r'Fast, exponential, conductance-based synapses (homogeneous parameters'},
			'E': {
				'model': 'iaf_cond_exp_sfa_rr',
				'C_m': 600.0,
				'E_L': -70.0,
				'V_reset': -70.0,
				'V_th': -54.,
				't_ref': 0.,
				'E_in': -75.,
				'E_ex': 0.,
				'g_L': 100.,
				'tau_syn_ex': 2.,
				'tau_syn_in': 6.,
				'q_sfa': 5.,
				'q_rr': 200.,
				'tau_sfa': 200.,
				'tau_rr': 2.,
				'E_sfa': -80.,
				'E_rr': -80.,
				'I_e': 0.
			},
			'I': {
				'model': 'iaf_cond_exp_sfa_rr',
				'C_m': 600.0,
				'E_L': -70.0,
				'V_reset': -70.0,
				'V_th': -54.,
				't_ref': 0.,
				'E_in': -75.,
				'E_ex': 0.,
				'g_L': 100.,
				'tau_syn_ex': 2.,
				'tau_syn_in': 6.,
				'q_sfa': 5.,
				'q_rr': 200.,
				'tau_sfa': 200.,
				'tau_rr': 2.,
				'E_sfa': -80.,
				'E_rr': -80.,
				'I_e': 0.
			},
		}
	elif default_set == 8.2:
		print ("\nLoading Default Neuron Set 8.2 - iaf_cond_exp_sfa_rr, fixed voltage threshold, dynamic refractory "
			   "time "
			   "spike frequency adaptation, Fast, conductance-based exponential synapses, homogeneous parameters")
		neuron_pars = {
			'description': {
				'neurons': 'Leaky integrate-and-fire, sfa, dynamic refractory time',
				'synapses': r'Fast, exponential, conductance-based synapses (homogeneous parameters'},
			'E': {
				'model': 'iaf_cond_exp_sfa_rr',
				'C_m': 250.0,
				'E_L': -70.0,
				'V_reset': -60.0,
				'V_th': -50.,
				't_ref': 0.5,
				'E_ex': 0.,
				'E_in': -75.,
				'g_L': 16.7,
				'tau_syn_ex': 2.,
				'tau_syn_in': 6.,
				'q_sfa': 10.,
				'q_rr': 1000.,
				'tau_sfa': 200.,
				'tau_rr': 2.,
				'E_sfa': -70.,
				'E_rr': -70.,
				'I_e': 0.
			},
			'I': {
				'model': 'iaf_cond_exp_sfa_rr',
				'C_m': 250.0,
				'E_L': -70.0,
				'V_reset': -60.0,
				'V_th': -50.,
				't_ref': 0.5,
				'E_ex': 0.,
				'E_in': -75.,
				'g_L': 16.7,
				'tau_syn_ex': 2.,
				'tau_syn_in': 6.,
				'q_sfa': 10.,
				'q_rr': 1000.,
				'tau_sfa': 200.,
				'tau_rr': 2.,
				'E_sfa': -70.,
				'E_rr': -70.,
				'I_e': 0.
			},
		}
	elif default_set == 8.3:
		print ("\nLoading Default Neuron Set 8.3 - iaf_cond_exp_sfa_rr, fixed voltage threshold, dynamic refractory "
			   "time "
			   "spike frequency adaptation, Fast, conductance-based exponential synapses, homogeneous parameters")
		neuron_pars = {
			'description': {
				'neurons': 'Leaky integrate-and-fire, sfa, dynamic refractory time',
				'synapses': r'Fast, exponential, conductance-based synapses (homogeneous parameters'},
			'E': {
				'model': 'iaf_cond_exp_sfa_rr',
				'C_m': 289.5,
				'E_L': -70.0,
				'V_reset': -70.0,
				'V_th': -57.,
				't_ref': 0.5,
				'E_ex': 0.,
				'g_ex': 2.,
				'g_in': 2.,
				'E_in': -75.,
				'g_L': 28.95,
				'tau_syn_ex': 1.5,
				'tau_syn_in': 10.,
				'q_sfa': 14.48,
				'q_rr': 3214.,
				'tau_sfa': 110.,
				'tau_rr': 1.97,
				'E_sfa': -70.,
				'E_rr': -70.,
				'I_e': 0.
			},
			'I': {
				'model': 'iaf_cond_exp_sfa_rr',
				'C_m': 289.5,
				'E_L': -70.0,
				'V_reset': -70.0,
				'V_th': -57.,
				't_ref': 0.5,
				'E_ex': 0.,
				'g_ex': 2.,
				'g_in': 2.,
				'E_in': -75.,
				'g_L': 28.95,
				'tau_syn_ex': 1.5,
				'tau_syn_in': 10.,
				'q_sfa': 14.48,
				'q_rr': 3214.,
				'tau_sfa': 110.,
				'tau_rr': 1.97,
				'E_sfa': -70.,
				'E_rr': -70.,
				'I_e': 0.
			},
		}
	elif default_set == 8.4:
		print ("\nLoading Default Neuron Set 8.4 - iaf_cond_exp_sfa_rr, fixed voltage threshold, dynamic refractory "
			   "time "
			   "spike frequency adaptation, Fast, conductance-based exponential synapses, homogeneous parameters")
		neuron_pars = {
			'description': {
				'neurons': 'Leaky integrate-and-fire, sfa, dynamic refractory time',
				'synapses': r'Fast, exponential, conductance-based synapses (homogeneous parameters'},
			'E': {
				'model': 'iaf_cond_exp_sfa_rr',
				'C_m': 289.5,
				'E_L': -70.0,
				'V_reset': -70.0,
				'V_th': -57.,
				't_ref': 0.5,
				'E_ex': 0.,
				'E_in': -75.,
				'g_L': 28.95,
				'tau_syn_ex': 2.,
				'tau_syn_in': 6.,
				'q_sfa': 14.48,
				'q_rr': 3214.,
				'tau_sfa': 110.,
				'tau_rr': 1.97,
				'E_sfa': -70.,
				'E_rr': -70.,
				'I_e': 0.
			},
			'I': {
				'model': 'iaf_cond_exp_sfa_rr',
				'C_m': 289.5,
				'E_L': -70.0,
				'V_reset': -70.0,
				'V_th': -57.,
				't_ref': 0.5,
				'E_ex': 0.,
				'E_in': -75.,
				'g_L': 28.95,
				'tau_syn_ex': 2.,
				'tau_syn_in': 6.,
				'q_sfa': 14.48,
				'q_rr': 3214.,
				'tau_sfa': 110.,
				'tau_rr': 1.97,
				'E_sfa': -70.,
				'E_rr': -70.,
				'I_e': 0.
			},
		}

	else:
		raise IOError("default_set not defined")

	return neuron_pars


def set_network_defaults(default_set=1, neuron_set=1, connection_set=0, N=1250, kernel_pars=None, **synapse_pars):
	# IDEA make a class for all the possible sets, along with their description, so one can also print-help it? overview
	"""

	:param default_set:
	:param neuron_set:
	:param connection_set:
	:param N:
	:param kernel_pars:
	:param synapse_pars:
	:return:
	"""
	if default_set == 1.1:
		print ("\nLoading Default Network Set 1 - Short time constants")
		syn_pars = ParameterSet(synapse_pars)
		nE = 0.8 * N
		nI = 0.2 * N

		rec_devices = rec_device_defaults(start=0.)
		neuron_pars = set_neuron_defaults(default_set=neuron_set)

		#############################################################################################################
		# NETWORK Parameters
		# ===========================================================================================================
		net_pars = {
			'n_populations': 2,
			'pop_names': 	 ['E', 'I'],
			'n_neurons': 	 [int(nE), int(nI)],
			'neuron_pars': 	 [neuron_pars['E'], neuron_pars['I']],
			'topology': 	 [False, False],
			'topology_dict': [None, None],
			'record_spikes': [True, True],
			'randomize_neuron_pars': [{'V_m': (np.random.uniform, {'low': -70., 'high': -50.})},
									  {'V_m': (np.random.uniform, {'low': -70., 'high': -50.})}],
			'spike_device_pars': [copy_dict(rec_devices,
			                                {'model': 		'spike_detector',
			                                 'record_to': 	['memory'],
			                                 'interval': 	0.1,
			                                 'label':		'E_Spikes'}),
			                      copy_dict(rec_devices,
			                                {'model': 		'spike_detector',
			                                 'record_to': 	['memory'],
			                                 'interval': 	0.1,
			                                 'label': 		'I_Spikes'})],
			'record_analogs': 		[False, False],
			'analog_device_pars': 	[None, None],
			'description': 			{'topology': 'None'}
		}
		#############################################################################################################
		# SYNAPSE Parameters
		# ============================================================================================================
		connection_pars, neuron_pars = set_connection_defaults(default_set=connection_set, syn_pars=syn_pars,
															   neuron_pars=neuron_pars)

	elif default_set == 1.2:
		print ("\nLoading Default Network Set 2 - Long time constants")
		syn_pars = ParameterSet(synapse_pars)
		nE = 0.8 * N
		nI = 0.2 * N

		rec_devices = rec_device_defaults(start=0.)
		neuron_pars = set_neuron_defaults(default_set=neuron_set)

		rec_keys = [len(n) for n in neuron_pars.keys() if n.find('rec')]
		assert(np.mean(np.diff(rec_keys)) == 0.), "Inconsistent neuron receptor parameters"

		#############################################################################################################
		# NETWORK Parameters
		# ===========================================================================================================
		net_pars = {
			'n_populations': 2,
			'pop_names': ['E', 'I'],
			'n_neurons': [int(nE), int(nI)],
			'neuron_pars': [neuron_pars['E'], neuron_pars['I']],
			'topology': [False, False],
			'topology_dict': [None, None],
			'record_spikes': [True, True],
			'spike_device_pars': [copy_dict(rec_devices,
			                                {'model': 'spike_detector',
			                                 'record_to': ['memory'],
			                                 'interval': 0.1,
			                                 'label': 'E_Spikes'}),
			                      copy_dict(rec_devices,
			                                {'model': 'spike_detector',
			                                 'record_to': ['memory'],
			                                 'interval': 0.1,
			                                 'label': 'I_Spikes'})],
			'record_analogs': [False, False],
			'analog_device_pars': [None, None],
			'description': {'topology': 'None'}
		}
		#############################################################################################################
		# SYNAPSE Parameters
		# ============================================================================================================
		connection_pars, neuron_pars = set_connection_defaults(default_set=connection_set, syn_pars=syn_pars,
															   neuron_pars=neuron_pars)

	elif default_set == 3:
		print ("\nLoading Default Network Set 3 - AMAT")
		syn_pars = ParameterSet(synapse_pars)
		nE = 0.8 * N
		nI = 0.2 * N

		rec_devices = rec_device_defaults(start=0.)
		neuron_pars = set_neuron_defaults(default_set=neuron_set)
		keys = neuron_pars.keys()
		keys.remove('description')
		if len(keys) > 1:
			neuron_params = [neuron_pars[n] for n in keys]
		else:
			neuron_params = [neuron_pars[keys[0]], neuron_pars[keys[0]]]
		#############################################################################################################
		# NETWORK Parameters
		# ===========================================================================================================
		net_pars = {
			'n_populations': 2,
			'pop_names': ['E', 'I'],
			'n_neurons': [int(nE), int(nI)],
			'neuron_pars': neuron_params,
			'topology': [False, False],
			'topology_dict': [None, None],
			'record_spikes': [True, True],
			'spike_device_pars': [copy_dict(rec_devices,
			                                {'model': 'spike_detector',
			                                 'record_to': ['memory'],
			                                 'interval': 0.1,
			                                 'label': 'E_Spikes'}),
			                      copy_dict(rec_devices,
			                                {'model': 'spike_detector',
			                                 'record_to': ['memory'],
			                                 'interval': 0.1,
			                                 'label': 'I_Spikes'})],
			'record_analogs': [False, False],
			'analog_device_pars': [None, None],
			'description': {'topology': 'None'}
		}
		#############################################################################################################
		# SYNAPSE Parameters
		# ============================================================================================================
		connection_pars, neuron_pars = set_connection_defaults(default_set=connection_set, syn_pars=syn_pars,
															   neuron_pars=neuron_pars)

	elif default_set == 3.1:
		print ("\nLoading Default Network Set 3.1 - Two-pool system")
		syn_pars = ParameterSet(synapse_pars)
		nE = 0.8 * N
		nI = 0.2 * N

		rec_devices = rec_device_defaults(start=0.)
		neuron_pars = set_neuron_defaults(default_set=neuron_set)
		keys = neuron_pars.keys()
		keys.remove('description')
		if len(keys) > 1:
			neuron_params = [neuron_pars[n] for n in keys]
		else:
			neuron_params = [neuron_pars[keys[0]], neuron_pars[keys[0]]]
		#############################################################################################################
		# NETWORK Parameters
		# ===========================================================================================================
		net_pars = {
			'n_populations': 4,
			'pop_names': ['E1', 'I1', 'E2', 'I2'],
			'n_neurons': [int(nE), int(nI), int(nE), int(nI)],
			'neuron_pars': [neuron_pars['E1'], neuron_pars['I1'], neuron_pars['E2'], neuron_pars['I2']],
			'topology': [False, False, False, False],
			'topology_dict': [None, None, None, None],
			'record_spikes': [True, True, True, True],
			'spike_device_pars': [copy_dict(rec_devices,
			                                {'model': 'spike_detector',
			                                 'record_to': ['memory'],
			                                 'interval': 0.1,
			                                 'label': 'E1_Spikes'}),
			                      copy_dict(rec_devices,
			                                 {'model': 'spike_detector',
			                                  'record_to': ['memory'],
			                                  'interval': 0.1,
			                                  'label': 'I1_Spikes'}),
			                      copy_dict(rec_devices,
			                                {'model': 'spike_detector',
			                                 'record_to': ['memory'],
			                                 'interval': 0.1,
			                                 'label': 'E2_Spikes'}),
			                      copy_dict(rec_devices,
			                                 {'model': 'spike_detector',
			                                  'record_to': ['memory'],
			                                  'interval': 0.1,
			                                  'label': 'I2_Spikes'}),],
			'record_analogs': [False, False, False, False],
			'analog_device_pars': [None, None, None, None],
			'description': {'topology': 'None'}
		}
		#############################################################################################################
		# SYNAPSE Parameters
		# ============================================================================================================
		synapses = syn_pars.connected_populations
		synapse_names = ['{0}'.format(syn[1]+'_'+syn[0]) for syn in synapses]
		assert (
			np.mean([n in synapses for n in syn_pars.connected_populations]).astype(bool)), "Inconsistent Parameters"

		connection_pars = {
			'n_synapse_types': len(synapses),
			'synapse_types': synapses,
			'synapse_names': synapse_names,
			'topology_dependent': [False for _ in range(len(synapses))],
			'models': syn_pars.synapse_models,
			'model_pars': syn_pars.synapse_model_parameters,
			'pre_computedW': syn_pars.pre_computedW,
			'weight_dist': syn_pars.weights,
			'delay_dist': syn_pars.delays,
			'conn_specs': syn_pars.conn_specs,
			'syn_specs': syn_pars.syn_specs,
			'description': {
				'connectivity': 'Sparse, Random with density 0.1 (all connections)',
				'plasticity': 'None'}
		}
		neuron_pars['description']['synapses'] += ' Non-adapting'

	elif default_set == 4.1:
		print ("\nLoading Default Network Set 4.1 - Standard BRN, grid topology")
		syn_pars = ParameterSet(synapse_pars)
		nE = 0.8 * N
		nI = 0.2 * N

		rec_devices = rec_device_defaults(start=0.)
		neuron_pars = set_neuron_defaults(default_set=neuron_set)
		keys = neuron_pars.keys()
		keys.remove('description')
		if len(keys) > 1:
			neuron_params = [neuron_pars[n] for n in keys]
		else:
			neuron_params = [neuron_pars[keys[0]], neuron_pars[keys[0]]]
		#############################################################################################################
		# NETWORK Parameters
		# ===========================================================================================================
		pos = (np.sqrt(N) * np.random.random_sample((int(N), 2))).tolist()
		tp_dict = {
			'elements': neuron_pars['E']['model'],
			'extent': [np.round(np.sqrt(N)), np.round(np.sqrt(N))],
			'center': [np.round(np.sqrt(N)) / 2., np.round(np.sqrt(N)) / 2.],
			'positions': 0,
			'edge_wrap': True
		}
		net_pars = {
			'n_populations': 2,
			'pop_names': ['E', 'I'],
			'n_neurons': [int(nE), int(nI)],
			'neuron_pars': neuron_params,
			'topology': [True, True],
			'topology_dict': [copy_dict(tp_dict, {'positions': pos[:int(nE)]}),
			                  copy_dict(tp_dict, {'positions': pos[int(nE):]})],
			'record_spikes': [True, True],
			'spike_device_pars': [copy_dict(rec_devices,
			                                {'model': 'spike_detector',
			                                 'record_to': ['memory'],
			                                 'interval': 0.1,
			                                 'label': 'E_Spikes'}),
			                      copy_dict(rec_devices,
			                                {'model': 'spike_detector',
			                                 'record_to': ['memory'],
			                                 'interval': 0.1,
			                                 'label': 'I_Spikes'})],
			'record_analogs': [False, False],
			'analog_device_pars': [None, None],
			'description': {'topology': 'None'}
		}
		#############################################################################################################
		# SYNAPSE Parameters
		# ============================================================================================================
		connection_pars, neuron_pars = set_connection_defaults(default_set=connection_set, syn_pars=syn_pars,
															   neuron_pars=neuron_pars)
	elif default_set == 4:
		print ("\nLoading Default Network Set 4 - Standard BRN")
		syn_pars = ParameterSet(synapse_pars)
		nE = 0.8 * N
		nI = 0.2 * N

		rec_devices = rec_device_defaults(start=0.)
		neuron_pars = set_neuron_defaults(default_set=neuron_set)
		keys = neuron_pars.keys()
		keys.remove('description')
		if len(keys) > 1:
			neuron_params = [neuron_pars[n] for n in keys]
		else:
			neuron_params = [neuron_pars[keys[0]], neuron_pars[keys[0]]]
		#############################################################################################################
		# NETWORK Parameters
		# ===========================================================================================================
		net_pars = {
			'n_populations': 2,
			'pop_names': ['E', 'I'],
			'n_neurons': [int(nE), int(nI)],
			'neuron_pars': neuron_params,
			'topology': [False, False],
			'topology_dict': [None, None],
			'record_spikes': [True, True],
			'spike_device_pars': [copy_dict(rec_devices,
			                                {'model': 'spike_detector',
			                                 'record_to': ['memory'],
			                                 'interval': 0.1,
			                                 'label': 'E_Spikes'}),
			                      copy_dict(rec_devices,
			                                {'model': 'spike_detector',
			                                 'record_to': ['memory'],
			                                 'interval': 0.1,
			                                 'label': 'I_Spikes'})],
			'record_analogs': [False, False],
			'analog_device_pars': [None, None],
			'description': {'topology': 'None'}
		}
		#############################################################################################################
		# SYNAPSE Parameters
		# ============================================================================================================
		connection_pars, neuron_pars = set_connection_defaults(default_set=connection_set, syn_pars=syn_pars,
		                                          neuron_pars=neuron_pars)

	elif default_set == 5:
		print ("\nLoading Default Network Set 5 - Single Neuron DC input")
		neuron_pars = set_neuron_defaults(default_set=neuron_set)
		n_pars = {k: v for k, v in neuron_pars.items() if k != 'description'}
		multimeter = rec_device_defaults(device_type='multimeter')
		multimeter.update({'record_from': ['V_m'], 'record_n': 1})
		pop_names = ['{0}'.format(str(n)) for n in n_pars.keys()]
		n_neurons = [1 for _ in n_pars.keys()]
		if len(neuron_pars.keys()) > 1:
			neuron_params = [n_pars[n] for n in n_pars.keys()]
		else:
			neuron_params = [n_pars[n_pars.keys()[0]], n_pars[n_pars.keys()[1]]]
		net_pars = ParameterSet({
			'n_populations': len(n_pars.keys()),
			'pop_names': pop_names,
			'n_neurons': n_neurons,
			'neuron_pars': neuron_params,
			'randomize_neuron_pars': [{} for _ in pop_names],
			'topology': [False for _ in n_pars.keys()],
			'topology_dict': [None for _ in n_pars.keys()],
			'record_spikes': [True for _ in n_pars.keys()],
			'spike_device_pars': [rec_device_defaults(device_type='spike_detector', label='single_neuron_spikes') for _
			                      in n_pars.keys()],
			'record_analogs': [True for _ in n_pars.keys()],
			'analog_device_pars': [multimeter for _ in n_pars.keys()],
			'description': {'topology': 'None'}
		})
		connection_pars = {}

	elif default_set == 5.1:
		print ("\nLoading Default Network Set 5.1 - Single Neuron DC input Variable parameters")
		neuron_pars = set_neuron_defaults(default_set=neuron_set)
		n_pars = {k: v for k, v in neuron_pars.items() if k != 'description'}
		multimeter = rec_device_defaults(device_type='multimeter')
		multimeter.update({'record_from': ['V_m'], 'record_n': 1})
		pop_names = ['{0}'.format(str(n)) for n in n_pars.keys()]
		n_neurons = [1 for n in n_pars.keys()]
		if len(neuron_pars.keys()) > 1:
			neuron_params = [n_pars[n] for n in n_pars.keys()]
		else:
			neuron_params = [n_pars[n_pars.keys()[0]], n_pars[n_pars.keys()[1]]]

		randomized_pars = {
			'E': {
				'C_m': (st.norm.rvs, {'loc': 114., 'scale': 8.7}),
				'E_L': (st.norm.rvs, {'loc': -73., 'scale': 4.}),
				'V_m': (np.random.uniform, {'low': -70., 'high': -50.}),
				'V_th': (st.norm.rvs, {'loc': -42., 'scale': 4.}),
				'V_reset': (st.norm.rvs, {'loc': -52., 'scale': 5.}),
				'g_L': (st.norm.rvs, {'loc': 4.73, 'scale': 0.38}),
				't_ref': (st.lognorm.rvs, {'s': 0.3, 'loc': 1.8, 'scale': 0.25})
			},
			'I1': {
				'C_m': (st.lognorm.rvs, {'s': 0.18, 'loc': 68.9, 'scale': 35.6}),
				'E_L': (st.norm.rvs, {'loc': -67.5, 'scale': 2.}),
				'V_m': (np.random.uniform, {'low': -70., 'high': -50.}),
				'V_th': (st.norm.rvs, {'loc': -40., 'scale': 4.}),
				'V_reset': (st.norm.rvs, {'loc': -58., 'scale': 6.4}),
				'g_L': (st.norm.rvs, {'loc': 9.09, 'scale': 0.75}),
				't_ref': (st.lognorm.rvs, {'s': 0.3, 'loc': 0.5, 'scale': 0.01})
			},
			'I2': {
				'C_m': (st.lognorm.rvs, {'s': 0.18, 'loc': 82.24, 'scale': 17.7}),
				'E_L': (st.norm.rvs, {'loc': -62.6, 'scale': 2.}),
				'V_m': (np.random.uniform, {'low': -70., 'high': -50.}),
				'V_th': (st.norm.rvs, {'loc': -36., 'scale': 2.}),
				'V_reset': (st.norm.rvs, {'loc': -54., 'scale': 5.4}),
				'g_L': (st.norm.rvs, {'loc': 4.5, 'scale': 0.2}),
				't_ref': (st.lognorm.rvs, {'s': 0.3, 'loc': 1.3, 'scale': 0.05})
			}}

		net_pars = ParameterSet({
			'n_populations': len(n_pars.keys()),
			'pop_names': pop_names,
			'n_neurons': n_neurons,
			'neuron_pars': neuron_params,
			'randomize_neuron_pars': [randomized_pars[n] for n in pop_names],
			'topology': [False for _ in n_pars.keys()],
			'topology_dict': [None for _ in n_pars.keys()],
			'record_spikes': [True for _ in n_pars.keys()],
			'spike_device_pars': [rec_device_defaults(device_type='spike_detector', label='single_neuron_spikes') for _
			                      in n_pars.keys()],
			'record_analogs': [True for _ in n_pars.keys()],
			'analog_device_pars': [multimeter for _ in n_pars.keys()],
			'description': {'topology': 'None'}
		})
		connection_pars = {}

	elif default_set == 6:
		print "\nLoading Default Network Set 6 - Single Neuron Synaptic input"
		syn_pars = ParameterSet(synapse_pars)
		nE = 0.8 * N
		nI = 0.2 * N
		rec_devices = rec_device_defaults(start=kernel_pars['transient_t'])
		neuron_pars = set_neuron_defaults(default_set=neuron_set)
		parrots = {'model': 'parrot_neuron'}

		n_pars = {k: v for k, v in neuron_pars.items() if k != 'description'}
		pop_names = ['{0}'.format(str(n)) for n in n_pars.keys()]

		n_neurons = [1 for n in n_pars.keys()]
		if len(neuron_pars.keys()) > 1:
			neuron_params = [n_pars[n] for n in n_pars.keys()]
		else:
			neuron_params = [n_pars[n_pars.keys()[0]], n_pars[n_pars.keys()[1]]]
		neuron_params.extend([parrots for _ in n_pars.keys()])

		n_neurons = [1 for _ in n_pars.keys()]
		n_neurons.extend([eval("int(n{0})".format(str(n))) for n in n_pars.keys()])

		devices_flags = [True for _ in range(len(pop_names))]
		devices_flags.extend([False for _ in range(len(pop_names))])

		sds = [syn_pars.devices[1] for _ in range(len(pop_names))]
		sds.extend([None for _ in range(len(pop_names))])

		mms = [syn_pars.devices[0] for _ in range(len(pop_names))]
		mms.extend([None for _ in range(len(pop_names))])

		pop_names.extend(['{0}_inputs'.format(str(n)) for n in n_pars.keys()])

		net_pars = {
			'n_populations': len(pop_names),
			'pop_names': pop_names,
			'n_neurons': n_neurons,
			'neuron_pars': neuron_params,
			'topology': [False for _ in range(len(pop_names))],
			'topology_dict': [None for _ in range(len(pop_names))],
			'record_spikes': devices_flags,
			'spike_device_pars': sds,
			'record_analogs': devices_flags,
			'analog_device_pars': mms,
			'description': {'topology': 'None'}
		}
		##############################################################
		connection_pars, neuron_pars = set_connection_defaults(default_set=connection_set, syn_pars=syn_pars,
															   neuron_pars=neuron_pars)

	elif default_set == 6.1:
		print ("\nLoading Default Network Set 5.2 - Single Neuron Synaptic input, heterogeneous")
		syn_pars = ParameterSet(synapse_pars)
		nE = 0.8 * N
		nI = 0.2 * N
		nI1 = 0.35 * nI
		nI2 = 0.7 * nI
		rec_devices = rec_device_defaults(start=kernel_pars['transient_t'])
		neuron_pars = set_neuron_defaults(default_set=neuron_set)
		parrots = {'model': 'parrot_neuron'}

		n_pars = {k: v for k, v in neuron_pars.items() if k != 'description'}
		pop_names = ['{0}'.format(str(n)) for n in n_pars.keys()]

		n_neurons = [1 for n in n_pars.keys()]
		if len(neuron_pars.keys()) > 1:
			neuron_params = [n_pars[n] for n in n_pars.keys()]
		else:
			neuron_params = [n_pars[n_pars.keys()[0]], n_pars[n_pars.keys()[1]]]
		neuron_params.extend([parrots for _ in n_pars.keys()])

		n_neurons = [1 for _ in n_pars.keys()]
		n_neurons.extend([eval("int(n{0})".format(str(n))) for n in n_pars.keys()])

		devices_flags = [True for _ in range(len(pop_names))]
		devices_flags.extend([False for _ in range(len(pop_names))])

		sds = [syn_pars.devices[1] for _ in range(len(pop_names))]
		sds.extend([None for _ in range(len(pop_names))])

		mms = [syn_pars.devices[0] for _ in range(len(pop_names))]
		mms.extend([None for _ in range(len(pop_names))])

		pop_names.extend(['{0}_inputs'.format(str(n)) for n in n_pars.keys()])
		randomized_pars = {
			'E': {
				'C_m': (st.norm.rvs, {'loc': 114., 'scale': 8.7}),
				'E_L': (st.norm.rvs, {'loc': -73., 'scale': 4.}),
				'V_m': (np.random.uniform, {'low': -70., 'high': -50.}),
				'V_th': (st.norm.rvs, {'loc': -42., 'scale': 4.}),
				'V_reset': (st.norm.rvs, {'loc': -52., 'scale': 5.}),
				'g_L': (st.norm.rvs, {'loc': 4.73, 'scale': 0.38}),
				't_ref': (st.lognorm.rvs, {'s': 0.3, 'loc': 1.8, 'scale': 0.25})
			},
			'I1': {
				'C_m': (st.lognorm.rvs, {'s': 0.18, 'loc': 68.9, 'scale': 35.6}),
				'E_L': (st.norm.rvs, {'loc': -67.5, 'scale': 2.}),
				'V_m': (np.random.uniform, {'low': -70., 'high': -50.}),
				'V_th': (st.norm.rvs, {'loc': -40., 'scale': 4.}),
				'V_reset': (st.norm.rvs, {'loc': -58., 'scale': 6.4}),
				'g_L': (st.norm.rvs, {'loc': 9.09, 'scale': 0.75}),
				't_ref': (st.lognorm.rvs, {'s': 0.3, 'loc': 0.5, 'scale': 0.01})
			},
			'I2': {
				'C_m': (st.lognorm.rvs, {'s': 0.18, 'loc': 82.24, 'scale': 17.7}),
				'E_L': (st.norm.rvs, {'loc': -62.6, 'scale': 2.}),
				'V_m': (np.random.uniform, {'low': -70., 'high': -50.}),
				'V_th': (st.norm.rvs, {'loc': -36., 'scale': 2.}),
				'V_reset': (st.norm.rvs, {'loc': -54., 'scale': 5.4}),
				'g_L': (st.norm.rvs, {'loc': 4.5, 'scale': 0.2}),
				't_ref': (st.lognorm.rvs, {'s': 0.3, 'loc': 1.3, 'scale': 0.05})
			}}
		net_pars = {
			'n_populations': len(pop_names),
			'pop_names': pop_names,
			'n_neurons': n_neurons,
			'neuron_pars': neuron_params,
			'randomize_neuron_pars': [randomized_pars[n] for n in pop_names if n[-6:] == 'inputs'],
			'topology': [False for _ in range(len(pop_names))],
			'topology_dict': [None for _ in range(len(pop_names))],
			'record_spikes': devices_flags,
			'spike_device_pars': sds,
			'record_analogs': devices_flags,
			'analog_device_pars': mms,
			'description': {'topology': 'None'}
		}
		##############################################################
		connection_pars, neuron_pars = set_connection_defaults(default_set=connection_set, syn_pars=syn_pars,
		                                          neuron_pars=neuron_pars)

	elif default_set == 6.2:
		print ("\nLoading Default Network Set 2 - Long time constants, heterogeneous")
		syn_pars = ParameterSet(synapse_pars)
		nE = 0.8 * N
		nI = 0.2 * N
		nI1 = 0.35 * nI
		nI2 = 0.65 * nI

		rec_devices = rec_device_defaults(start=0.)
		neuron_pars = set_neuron_defaults(default_set=neuron_set)

		keys = neuron_pars.keys()
		keys.remove('description')
		#############################################################################################################
		# NETWORK Parameters
		# =======================================================================================c====================
		pos = (np.sqrt(N) * np.random.random_sample((int(N), 2))).tolist()
		tp_dict = {
			'elements': neuron_pars['E']['model'],
			'extent': [np.round(np.sqrt(N)), np.round(np.sqrt(N))],
			'center': [np.round(np.sqrt(N)) / 2., np.round(np.sqrt(N)) / 2.],
			'positions': 0,
			'edge_wrap': True}

		randomized_pars = {
			'E': {
				'C_m': (st.norm.rvs, {'loc': 114., 'scale': 8.7}),
				'E_L': (st.norm.rvs, {'loc': -73., 'scale': 4.}),
				'V_m': (np.random.uniform, {'low': -70., 'high': -50.}),
				'V_th': (st.norm.rvs, {'loc': -42., 'scale': 4.}),
				'V_reset': (st.norm.rvs, {'loc': -52., 'scale': 5.}),
				'g_L': (st.norm.rvs, {'loc': 4.73, 'scale': 0.38}),
				't_ref': (st.lognorm.rvs, {'s': 0.3, 'loc': 1.8, 'scale': 0.25})
			},
			'I1': {
				'C_m': (st.lognorm.rvs, {'s': 0.18, 'loc': 68.9, 'scale': 35.6}),
				'E_L': (st.norm.rvs, {'loc': -67.5, 'scale': 2.}),
				'V_m': (np.random.uniform, {'low': -70., 'high': -50.}),
				'V_th': (st.norm.rvs, {'loc': -40., 'scale': 4.}),
				'V_reset': (st.norm.rvs, {'loc': -58., 'scale': 6.4}),
				'g_L': (st.norm.rvs, {'loc': 9.09, 'scale': 0.75}),
				't_ref': (st.lognorm.rvs, {'s': 0.3, 'loc': 0.5, 'scale': 0.01})
			},
			'I2': {
				'C_m': (st.lognorm.rvs, {'s': 0.18, 'loc': 82.24, 'scale': 17.7}),
				'E_L': (st.norm.rvs, {'loc': -62.6, 'scale': 2.}),
				'V_m': (np.random.uniform, {'low': -70., 'high': -50.}),
				'V_th': (st.norm.rvs, {'loc': -36., 'scale': 2.}),
				'V_reset': (st.norm.rvs, {'loc': -54., 'scale': 5.4}),
				'g_L': (st.norm.rvs, {'loc': 4.5, 'scale': 0.2}),
				't_ref': (st.lognorm.rvs, {'s': 0.3, 'loc': 1.3, 'scale': 0.05})
			}}
		net_pars = {
			'n_populations': 3,
			'pop_names': ['E', 'I1', 'I2'],
			'n_neurons': [int(nE), int(nI1), int(nI2)],
			'neuron_pars': [neuron_pars['E'], neuron_pars['I1'], neuron_pars['I2']],
			'randomize_neuron_pars': [randomized_pars['E'], randomized_pars['I1'], randomized_pars['I2']],
			'topology': [False, False, False],
			'topology_dict': [None, None, None],#copy_dict(tp_dict, {'positions': pos[:int(nE)]}),
			                  #copy_dict(tp_dict, {'positions': pos[int(nE):int(nI1)]}),
			                  #copy_dict(tp_dict, {'positions': pos[int(nE)+int(nI1):]})],
			'record_spikes': [True, True, True],
			'spike_device_pars': [copy_dict(rec_devices,
			                                {'model': 'spike_detector',
			                                 'record_to': ['memory'],
			                                 'interval': 0.1,
			                                 'label': 'E_Spikes'}),
			                      copy_dict(rec_devices,
			                                {'model': 'spike_detector',
			                                 'record_to': ['memory'],
			                                 'interval': 0.1,
			                                 'label': 'I1_Spikes'}),
			                      copy_dict(rec_devices,
			                                {'model': 'spike_detector',
			                                 'record_to': ['memory'],
			                                 'interval': 0.1,
			                                 'label': 'I2_Spikes'})],
			'record_analogs': [False, False, False],
			'analog_device_pars': [None, None, None],
			'description': {'topology': 'Random'}
		}
		##############################################################
		connection_pars, neuron_pars = set_connection_defaults(default_set=connection_set, syn_pars=syn_pars,
															   neuron_pars=neuron_pars)
	elif default_set == 6.3:
		print ("\nLoading Default Network Set 2 - Long time constants, homogeneous")
		syn_pars = ParameterSet(synapse_pars)
		nE = 0.8 * N
		nI = 0.2 * N
		nI1 = 0.35 * nI
		nI2 = 0.7 * nI

		rec_devices = rec_device_defaults(start=0.)
		neuron_pars = set_neuron_defaults(default_set=neuron_set)

		keys = neuron_pars.keys()
		keys.remove('description')
		#############################################################################################################
		# NETWORK Parameters
		# ===========================================================================================================
		pos = (np.sqrt(N) * np.random.random_sample((int(N), 2))).tolist()
		tp_dict = {
			'elements': neuron_pars['E']['model'],
			'extent': [np.round(np.sqrt(N)), np.round(np.sqrt(N))],
			'center': [np.round(np.sqrt(N)) / 2., np.round(np.sqrt(N)) / 2.],
			'positions': 0,
			'edge_wrap': True}

		net_pars = {
			'n_populations': 3,
			'pop_names': ['E', 'I1', 'I2'],
			'n_neurons': [int(nE), int(nI1), int(nI2)],
			'neuron_pars': [neuron_pars['E'], neuron_pars['I1'], neuron_pars['I2']],
			#'randomize_neuron_pars': [randomized_pars['E'], randomized_pars['I1'], randomized_pars['I2']],
			'topology': [False, False, False],
			'topology_dict': [None, None, None],  # copy_dict(tp_dict, {'positions': pos[:int(nE)]}),
			# copy_dict(tp_dict, {'positions': pos[int(nE):int(nI1)]}),
			# copy_dict(tp_dict, {'positions': pos[int(nE)+int(nI1):]})],
			'record_spikes': [True, True, True],
			'spike_device_pars': [copy_dict(rec_devices,
			                                {'model': 'spike_detector',
			                                 'record_to': ['memory'],
			                                 'interval': 0.1,
			                                 'label': 'E_Spikes'}),
			                      copy_dict(rec_devices,
			                                {'model': 'spike_detector',
			                                 'record_to': ['memory'],
			                                 'interval': 0.1,
			                                 'label': 'I1_Spikes'}),
			                      copy_dict(rec_devices,
			                                {'model': 'spike_detector',
			                                 'record_to': ['memory'],
			                                 'interval': 0.1,
			                                 'label': 'I2_Spikes'})],
			'record_analogs': [False, False, False],
			'analog_device_pars': [None, None, None],
			'description': {'topology': 'Random'}
		}
		##############################################################
		connection_pars, neuron_pars = set_connection_defaults(default_set=connection_set, syn_pars=syn_pars,
		                                                       neuron_pars=neuron_pars)
	else:
		raise IOError("default_set not defined")

	return ParameterSet(neuron_pars), ParameterSet(net_pars), ParameterSet(connection_pars)


def set_connection_defaults(default_set=0, syn_pars=None, neuron_pars=None):
	"""

	:param default_set:
	:param syn_pars:
	:param neuron_pars:
	:return:
	"""
	connection_pars = {}
	if default_set == 1.1:
		print "\nLoading Default Connection Set 1.1 - Multiple receptors, E/I neurons, Short time constants, " \
		      "Topology not considered"
		#############################################################################################################
		# SYNAPSE Parameters
		# ============================================================================================================
		synapses = syn_pars.connected_populations #[('E', 'E'), ('E', 'I'), ('I', 'E'), ('I', 'I')]
		synapse_names = [n[1]+n[0] for n in synapses]
		synapse_models = syn_pars.synapse_models
		model_pars = syn_pars.synapse_model_parameters

		assert (
		np.mean([n in synapses for n in syn_pars.connected_populations]).astype(bool)), "Inconsistent Parameters"
		connection_pars = {
			'n_synapse_types': len(synapses),
			'synapse_types': synapses,
			'synapse_names': synapse_names,
			'topology_dependent': [False for _ in range(len(synapses))],
			'models': synapse_models,
			'model_pars': model_pars,

			'pre_computedW': syn_pars.pre_computedW,
			'weight_dist': syn_pars.weights,
			'delay_dist': syn_pars.delays,
			'conn_specs': syn_pars.conn_specs,
			'syn_specs': syn_pars.syn_specs,
			'description': {
				'connectivity': 'Sparse, Random with density 0.1 (all connections)',
				'plasticity': 'None'}
		}
		neuron_pars['description']['synapses'] += ' Non-adapting'

	elif default_set == 1.2:
		print "\nLoading Default Connection Set 1.2 - Multiple receptors, E/I neurons, Short and long time constants, " \
		      "Topology not considered"
		#############################################################################################################
		# SYNAPSE Parameters
		# ============================================================================================================
		assert(isinstance(syn_pars.connected_populations, list) and isinstance(syn_pars.connected_populations[0],
		                                                                       tuple)), "Incorrect type"
		synapses = syn_pars.connected_populations * 2
		synapse_names = []
		model_pars = []
		synapse_models = []
		for idx, n in enumerate(synapses):
			if idx >= len(synapses)/2:
				synapse_names.append(str(n[0]+n[1]+'_copy'))
				model_pars.append(syn_pars.synapse_model_parameters[idx - len(synapses)/2])
				synapse_models.append(syn_pars.synapse_model_parameters[idx - len(synapses)/2])
			else:
				synapse_names.append(str(n[0]+n[1]))
				model_pars.append(syn_pars.synapse_model_parameters[idx])
				synapse_models.append(syn_pars.synapse_model_parameters[idx])

		assert (
		np.mean([n in synapses for n in syn_pars.connected_populations]).astype(bool)), "Inconsistent Parameters"
		connection_pars = {
			'n_synapse_types': len(synapses),
			'synapse_types': synapses,
			'synapse_names': synapse_names,
			'topology_dependent': [False for _ in range(len(synapses))],
			'models': synapse_models,
			'model_pars': model_pars,

			'pre_computedW': syn_pars.pre_computedW,
			'weight_dist': syn_pars.weights,
			'delay_dist': syn_pars.delays,
			'conn_specs': syn_pars.conn_specs,
			'syn_specs': syn_pars.syn_specs,  # TODO: Check
			'description': {
				'connectivity': 'Sparse, Random with density 0.1 (all connections)',
				'plasticity': 'None'}
		}
		neuron_pars['description']['synapses'] += ' Non-adapting'

	elif default_set == 1.3:
		print "\nLoading Default Connection Set 1.3 - Multiple receptors, E/I1/I2 neurons, Short and long time " \
		      "constants, " \
		      "Topology not considered"
		#############################################################################################################
		# SYNAPSE Parameters
		# ============================================================================================================
		assert(isinstance(syn_pars.connected_populations, list) and isinstance(syn_pars.connected_populations[0],
		                                                                       tuple)), "Incorrect type"
		synapses = syn_pars.connected_populations
		assert (
		np.mean([n in synapses for n in syn_pars.connected_populations]).astype(bool)), "Inconsistent Parameters"
		connection_pars = {
			'n_synapse_types': len(synapses),
			'synapse_types': synapses,
			'synapse_names': syn_pars.synapse_names,
			'topology_dependent': [False for _ in range(len(synapses))],
			'models': syn_pars.synapse_models,
			'model_pars': syn_pars.synapse_model_parameters,
			'pre_computedW': syn_pars.pre_computedW,
			'weight_dist': syn_pars.weights,
			'delay_dist': syn_pars.delays,
			'conn_specs': syn_pars.conn_specs,
			'syn_specs': syn_pars.syn_specs,
			'description': {
				'connectivity': 'Sparse, Random with density 0.1 (all connections)',
				'plasticity': 'None'}
		}
		neuron_pars['description']['synapses'] += ' Non-adapting'

	return connection_pars, neuron_pars


def set_encoding_defaults(default_set=1, input_dimensions=1, n_encoding_neurons=0, encoder_neuron_pars=None,
                          **synapse_pars):
	"""

	:param default_set:
	:return:
	"""
	if default_set == 0:
		print "\nLoading Default Encoding Set 0 - Empty Settings (add background noise)"
		encoding_pars = {
			'description': {'general': r'',
			                'specific': r'',
			                'parameters': r''},
			'encoder': {
				'N': 0,
				'labels': [],
				'models': [],
				'model_pars': [],
				'n_neurons': [],
				'neuron_pars': [],
				'topology': [],
				'topology_dict': [],
				'record_spikes': [],
				'spike_device_pars': [],
				'record_analogs': [],
				'analog_device_pars': []},
			'generator': {
				'N': 0,
				'labels': [],
				'models': [],
				'model_pars': [],
				'topology': [],
				'topology_pars': []},
			'connectivity': {
				'synapse_name': [],
				'connections': [],
				'topology_dependent': [],
				'conn_specs': [],
				'syn_specs': [],
				'models': [],
				'model_pars': [],
				'weight_dist': [],
				'delay_dist': [],
				'preset_W': []},
			'input_decoder': None}

	elif default_set == 1:
		# ###################################################################
		# Encoding Type 1 - DC injection to target populations
		# ###################################################################
		gen_label = 'DC_input'
		keys = ['target_population_names', 'conn_specs', 'syn_specs', 'models', 'model_pars',
		        'weight_dist', 'delay_dist', 'preset_W']
		if not np.mean([n in synapse_pars.keys() for n in keys]).astype(bool):
			raise TypeError("Incorrect Synapse Parameters")
		syn_pars = ParameterSet(synapse_pars)
		# n_connections = len(syn_pars.target_population_names)
		connections = [(n, gen_label) for n in syn_pars.target_population_names]
		synapse_names = [gen_label+'syn' for _ in syn_pars.target_population_names]
		print "\nLoading Default Encoding Set 1 - DC input to {0}".format(str(syn_pars.target_population_names))
		encoding_pars = {
			'description': {'general': 'Direct encoding (DC injection to network)',
			                'specific': r'Variable current trace ($u(t)$)',
			                'parameters': r'$N_{\mathrm{stim}}$ & $%s$ & Number of input stimuli \tabularnewline '
			                              r' \hline ' % str(input_dimensions) +
			                              r'$\gamma_{\mathrm{in}}$ & $1$ & Fraction of receiving neurons '
			                              r'\tabularnewline \hline '},
			'encoder': {
				'N': 0,
				'labels': [],
				'models': [],
				'model_pars': [],
				'n_neurons': [],
				'neuron_pars': [],
				'topology': [],
				'topology_dict': [],
				'record_spikes': [],
				'spike_device_pars': [],
				'record_analogs': [],
				'analog_device_pars': []},
			'generator': {
				'N': input_dimensions,
				'labels': [gen_label],
				'models': ['step_current_generator'],
				'model_pars': [{'start': 0., 'stop': sys.float_info.max, 'origin': 0.}],
				'topology': [False for _ in range(input_dimensions)],
				'topology_pars': [None for _ in range(input_dimensions)]},
			'connectivity': {
				'synapse_name': synapse_names,
				'connections': connections,
				'topology_dependent': [False, False],
				'conn_specs': syn_pars.conn_specs,
				'syn_specs': syn_pars.syn_specs,
				'models': syn_pars.models,
				'model_pars': syn_pars.model_pars,
				'weight_dist': syn_pars.weight_dist,
				'delay_dist': syn_pars.delay_dist,
				'preset_W': syn_pars.preset_W},
			'input_decoder': None
		}

	elif default_set == 2:
		# ###################################################################
		# Encoding Type 2 - Deterministic spike encoding layer
		# ###################################################################
		rec_devices = rec_device_defaults()
		enc_label = 'NEF'
		keys = ['target_population_names', 'conn_specs', 'syn_specs', 'models', 'model_pars',
		        'weight_dist', 'delay_dist', 'preset_W']
		if not np.mean([n in synapse_pars.keys() for n in keys]).astype(bool):
			raise TypeError("Incorrect Synapse Parameters")
		syn_pars = ParameterSet(synapse_pars)
		# n_connections = len(syn_pars.target_population_names) + 1
		labels = [enc_label + '{0}'.format(str(n)) for n in range(input_dimensions)]
		connections = [(n, enc_label) for n in syn_pars.target_population_names]
		conn_specs = syn_pars.conn_specs
		conn_specs.insert(0, {'rule': 'all_to_all'})#None)
		synapse_names = [enc_label+'_'+n for n in syn_pars.target_population_names]
		connections.insert(0, ('NEF', 'StepGen'))
		synapse_names.insert(0, 'Gen_Enc')
		spike_device_pars = [copy_dict(rec_devices, {'model': 'spike_detector',
		                                             'label': 'input_Spikes'}) for _ in range(input_dimensions)]
		models = syn_pars.models
		models.insert(0, 'static_synapse')
		model_pars = syn_pars.model_pars
		model_pars.insert(0, {})
		weight_dist = syn_pars.weight_dist
		weight_dist.insert(0, 1.)
		delay_dist = syn_pars.delay_dist
		delay_dist.insert(0, 0.1)
		syn_specs = syn_pars.syn_specs
		syn_specs.insert(0, {})
		if hasattr(syn_pars, 'gen_to_enc_W'):
			preset_W = syn_pars.preset_W
			preset_W.insert(0, syn_pars.gen_to_enc_W)
		else:
			preset_W = syn_pars.preset_W

		print "\nLoading Default Encoding Set 2 - Deterministic spike encoding, {0} input populations of {1} [{2} " \
		      "neurons] connected to {3}".format(
				str(input_dimensions), str(n_encoding_neurons), str(encoder_neuron_pars['model']), str(
				syn_pars.target_population_names))

		encoding_pars = {
			'description': {'general': 'Deterministic spike encoding, DC injection to input population (NEF)',
			                'specific': r'Variable current trace ($u(t)$) delivered to layer of {0} encoder '
			                            r'neurons'.format(str(n_encoding_neurons)),
			                'parameters': r'$N_{\mathrm{stim}}$ & $%s$ & Number of input stimuli \tabularnewline '
			                              r' \hline '% str(input_dimensions) + r'$N_{\mathrm{aff}}$ & $%s$ & '
			                              r'Number of afferent neurons \tabularnewline \hline ' % str(
				                n_encoding_neurons) + r'$\gamma_{\mathrm{in}}$ & $1$ & Fraction of receiving neurons \tabularnewline \hline '},
			'encoder': {
				'N': input_dimensions,
				'labels': [enc_label for _ in range(input_dimensions)],
				'models': [enc_label for _ in range(input_dimensions)],
				'model_pars': [None for _ in range(input_dimensions)],
				'n_neurons': [n_encoding_neurons for _ in range(input_dimensions)],
				'neuron_pars': [encoder_neuron_pars for _ in range(input_dimensions)],
				'topology': [False for _ in range(input_dimensions)],
				'topology_dict': [None for _ in range(input_dimensions)],
				'record_spikes': [True for _ in range(input_dimensions)],
				'spike_device_pars': spike_device_pars,
				'record_analogs': [False for _ in range(input_dimensions)],
				'analog_device_pars': [None for _ in range(input_dimensions)]},
			'generator': {
				'N': 1,
				'labels': ['StepGen'],
				'models': ['step_current_generator'],
				'model_pars': [{'start': 0., 'stop': sys.float_info.max, 'origin': 0.}],
				'topology': [False],
				'topology_pars': [None]},
			'connectivity': {
				'connections': connections,
				'synapse_name': synapse_names,
				'topology_dependent': [False for _ in range(len(synapse_names))],
				'conn_specs': conn_specs,
				'models': models,
				'model_pars': model_pars,
				'weight_dist': weight_dist,
				'delay_dist': delay_dist,
				'preset_W': preset_W,
				'syn_specs': syn_specs},
			'input_decoder': {'encoder_label': enc_label}
		}
	elif default_set == 3:
		# ###################################################################
		# Encoding Type 3 - Stochastic spike encoding layer
		# ###################################################################
		gen_label = 'inh_poisson'
		keys = ['target_population_names', 'conn_specs', 'syn_specs', 'models', 'model_pars',
		        'weight_dist', 'delay_dist', 'preset_W']
		if not np.mean([n in synapse_pars.keys() for n in keys]).astype(bool):
			raise TypeError("Incorrect Synapse Parameters")
		syn_pars = ParameterSet(synapse_pars)
		# n_connections = len(syn_pars.target_population_names)
		connections = [(n, gen_label) for n in syn_pars.target_population_names]
		synapse_names = [gen_label+'syn' for _ in syn_pars.target_population_names]
		print "\nLoading Default Encoding Set 3 - Stochastic spike encoding, independent realizations of " \
		      "inhomogeneous Poisson processes connected to {0}".format(str(syn_pars.target_population_names))

		encoding_pars = {
			'description': {'general': 'Stochastic spike encoding, no input population (inhomogeneous Poisson input '
			                           'to network)',
			                'specific': r'Variable rate ($u(t)$)',
			                'parameters': r'$N_{\mathrm{stim}}$ & $%s$ & Number of input stimuli \tabularnewline '
			                              r' \hline' % str(input_dimensions) + r'$N_{\mathrm{aff}}$ & $0$ & '
			                                                                         r'Number of afferent ' \
		                                                        r'neurons '
			                                    r'\tabularnewline '
			                              r'\hline ' + r'$\gamma_{\mathrm{in}}$ & $0.$ & Fraction of receiving neurons '
			                              r'\tabularnewline \hline ' + r'$\sigma_{\mathrm{in}}$ & $0$ & peak rate of '
			                              r'input process \tabularnewline \hline '},
			'encoder': {
				'N': 0,
				'labels': [],
				'models': [],
				'model_pars': [],
				'n_neurons': [],
				'neuron_pars': [],
				'topology': [],
				'topology_dict': [],
				'record_spikes': [],
				'spike_device_pars': [],
				'record_analogs': [],
				'analog_device_pars': []},
			'generator': {
				'N': input_dimensions,
				'labels': ['inh_poisson'],
				'models': ['inh_poisson_generator'],
				'model_pars': [{'start': 0., 'stop': sys.float_info.max, 'origin': 0.}],
				'topology': [False],
				'topology_pars': [None]},
			'connectivity': {
				'synapse_name': synapse_names,
				'connections': connections,
				'topology_dependent': [False, False],
				'conn_specs': syn_pars.conn_specs,
				'syn_specs': syn_pars.syn_specs,
				'models': syn_pars.models,
				'model_pars': syn_pars.model_pars,
				'weight_dist': syn_pars.weight_dist,
				'delay_dist': syn_pars.delay_dist,
				'preset_W': syn_pars.preset_W}
		}
	elif default_set == 4:
		# ###################################################################
		# Encoding Type 3 - Precise Spatiotemporal spike encoding (Frozen noise)
		# ###################################################################
		gen_label = 'spike_pattern'
		keys = ['target_population_names', 'conn_specs', 'syn_specs', 'models', 'model_pars',
		        'weight_dist', 'delay_dist', 'preset_W']
		if not np.mean([n in synapse_pars.keys() for n in keys]).astype(bool):
			raise TypeError("Incorrect Synapse Parameters")
		syn_pars = ParameterSet(synapse_pars)
		connections = [(n, gen_label) for n in syn_pars.target_population_names]
		synapse_names = [gen_label+'syn' for _ in syn_pars.target_population_names]
		conn_tp = [False for _ in range(len(connections))]
		if hasattr(syn_pars, 'jitter'):
			jitter = syn_pars.jitter
		else:
			jitter = None

		print "\nLoading Default Encoding Set 4 - Stochastic spike encoding, {0} fixed spike pattern templates " \
		      "composed of {1} independent spike trains connected to {2}".format(
				str(input_dimensions), str(n_encoding_neurons), str(syn_pars.target_population_names))

		encoding_pars = {
			'description': {'general': 'Precise spatiotemporal spike pattern encoding (each input symbol is converted to '
			                           'a fixed instance of frozen Poisson noise)',
			                'specific': r'Fixed spatiotemporal spike pattern',
			                'parameters': r'$N_{\mathrm{stim}}$ & $%s$ & Number of unique input patterns '
			                              r'\tabularnewline \hline ' % str(input_dimensions) + r'$N^{\mathrm{'
			                              r'aff}}$ & $%s$ & Number of afferent '
			                             r'neurons \tabularnewline \hline ' % str(n_encoding_neurons) +
			                             r'$d^{\mathrm{patt}}$ & $0$ & Pattern duration \tabularnewline \hline ' +
			                             r'$r^{\mathrm{patt}}$ & $0$ & Pattern amplitude \tabularnewline \hline ' +
			                             r'$d^{\mathrm{ipi}}$ & $0$ & Inter-stimulus interval \tabularnewline \hline '
			                },
			'encoder': {
				'N': 0,
				'labels': [],
				'models': [],
				'model_pars': [],
				'n_neurons': [],
				'neuron_pars': [],
				'topology': [],
				'topology_dict': [],
				'record_spikes': [],
				'spike_device_pars': [],
				'record_analogs': [],
				'analog_device_pars': []},
			'generator': {
				'N': 1,
				'labels': ['spike_pattern'],
				'models': ['spike_generator'],
				'model_pars': [{'start': 0., 'stop': sys.float_info.max, 'origin': 0.}],
				'jitter': jitter,
				'topology': [False],
				'topology_pars': [None],
				'gen_to_enc_W': syn_pars.gen_to_enc_W},
			'connectivity': {
				'synapse_name': synapse_names,
				'connections': connections,
				'topology_dependent': conn_tp,
				'conn_specs': syn_pars.conn_specs,
				'syn_specs': syn_pars.syn_specs,
				'models': syn_pars.models,
				'model_pars': syn_pars.model_pars,
				'weight_dist': syn_pars.weight_dist,
				'delay_dist': syn_pars.delay_dist,
				'preset_W': syn_pars.preset_W},
			'input_decoder': {}
		}

	else:
		raise IOError("default_set not defined")

	return ParameterSet(encoding_pars)


def add_background_noise(encoding_pars, noise_pars):
	"""
	Adds a source of Poisson input to the specified populations (by modifying the encoding parameters)
	:param encoding_pars: original encoding parameters
	:param noise_pars: parameters of the noise process
	:return: encoding ParameterSet
	"""
	extra_description = ', Background Poissonian input at {0} spikes/s'.format(str(noise_pars['rate']))
	extra_parameters = r'$\nu$ & ${0}$ & Background Noise Rate'.format(str(noise_pars['rate']))
	encoding_pars.description.general += extra_description
	encoding_pars.description.parameters += extra_parameters
	extra_pars = noise_pars.pop('additional_parameters')
	target_populations = noise_pars.pop('target_population_names')
	if 'generator_label' in noise_pars.keys():
		label = noise_pars['generator_label']
	else:
		label = 'X_noise'
	encoding_pars.generator.N += 1
	encoding_pars.generator.labels.append(label)
	encoding_pars.generator.models.append('poisson_generator')
	encoding_pars.generator.model_pars.append(noise_pars)
	encoding_pars.generator.topology.append(False)
	encoding_pars.generator.topology_pars.append(None)

	connections = [(x, label) for x in target_populations]
	encoding_pars.connectivity.synapse_name.extend([label for _ in range(len(connections))])
	encoding_pars.connectivity.connections.extend(connections)
	encoding_pars.connectivity.topology_dependent.extend([False for _ in range(len(connections))])
	if 'conn_specs' in extra_pars.keys():
		encoding_pars.connectivity.conn_specs.extend([extra_pars['conn_specs'] for _ in range(len(connections))])
	else:
		encoding_pars.connectivity.conn_specs.extend([{'rule': 'all_to_all'} for _ in range(len(connections))])
	if 'syn_specs' in extra_pars.keys():
		encoding_pars.connectivity.syn_specs.extend([extra_pars['syn_specs'] for _ in range(len(connections))])
	else:
		encoding_pars.connectivity.syn_specs.extend([{} for _ in range(len(connections))])
	if 'models' in extra_pars.keys():
		encoding_pars.connectivity.models.extend([extra_pars['models'] for _ in range(len(connections))])
	else:
		encoding_pars.connectivity.models.extend(['static_synapse' for _ in range(len(connections))])
	if 'model_pars' in extra_pars.keys():
		encoding_pars.connectivity.model_pars.extend([extra_pars['model_pars'] for _ in range(len(connections))])
	else:
		encoding_pars.connectivity.model_pars.extend([{} for _ in range(len(connections))])
	if 'weight_dist' in extra_pars.keys():
		encoding_pars.connectivity.weight_dist.extend([extra_pars['weight_dist'] for _ in range(len(connections))])
	else:
		encoding_pars.connectivity.weight_dist.extend([1.0 for _ in range(len(connections))])
	if 'delay_dist' in extra_pars.keys():
		encoding_pars.connectivity.delay_dist.extend([extra_pars['delay_dist'] for _ in range(len(connections))])
	else:
		encoding_pars.connectivity.delay_dist.extend([{} for _ in range(len(connections))])
	encoding_pars.connectivity.preset_W.extend([None for _ in range(len(connections))])


def set_input_decoders(encoding_pars, input_decoder_pars, kernel_pars):
	"""
	Updates encoding parameters, adding decoders to the encoding layer
	:param encoding_pars: original encoding parameters to update
	:param encoder_label: label of encoder to readout
	:param input_decoder_pars: parameters of decoder to attach
	:param kernel_pars: main system parameters
	:return: updates encoding ParameterSet
	"""
	if not isinstance(input_decoder_pars, ParameterSet):
		input_decoder_pars = ParameterSet(input_decoder_pars)
	if isinstance(encoding_pars, ParameterSet):
		encoding_pars = encoding_pars.as_dict()
	enc_label = encoding_pars['input_decoder'].pop('encoder_label')
	encoding_pars['input_decoder'] = {}
	decoder_dict = copy_dict(input_decoder_pars.as_dict(), {'decoded_population': [enc_label for _ in range(len(
		input_decoder_pars.state_variable))]})
	resolution = decoder_dict.pop('output_resolution')

	input_decoder = set_decoding_defaults(default_set=1, output_resolution=resolution, kernel_pars=kernel_pars, **decoder_dict)
	encoding_pars.update({'input_decoder': input_decoder.as_dict()})

	return ParameterSet(encoding_pars)


def add_parrots(encoding_pars, n_parrots, decode=True, **extra_pars):
	"""
	Attaches a layer of parrot neurons to the encoder (for cases when the generator is a spike-emmitting device)
	:param encoding_pars: original encoding parameters
	:param n_parrots: number of parrot neurons to attach (should be the same as the number of unique generators)
	:param decode: attach decoders and readouts to parrot neurons.. (only spikes can be read!)
	"""
	if extra_pars.items():
		conn_specs = extra_pars['conn_specs']
		presetW = extra_pars['preset_W']
	else:
		conn_specs = {'rule': 'one_to_one'}
		presetW = None
	rec_devices = rec_device_defaults()
	encoding_pars.encoder.N += 1
	encoding_pars.encoder.labels.extend(['parrots'])
	encoding_pars.encoder.models.extend(['parrot_neuron'])
	encoding_pars.encoder.model_pars.extend([None])
	encoding_pars.encoder.n_neurons.extend([n_parrots])
	encoding_pars.encoder.neuron_pars.extend([{'model': 'parrot_neuron'}])
	encoding_pars.encoder.topology.extend([False])
	encoding_pars.encoder.topology_dict.extend([None])
	encoding_pars.encoder.record_spikes.extend([True])
	encoding_pars.encoder.spike_device_pars.extend([copy_dict(rec_devices, {'model': 'spike_detector',
	                                             'label': 'input_Spikes'})])
	encoding_pars.encoder.record_analogs.extend([False])
	encoding_pars.encoder.analog_device_pars.extend([None])
	syn_name = encoding_pars.connectivity.synapse_name[0]  # all synapses from a device must have the same name!!
	encoding_pars.connectivity.synapse_name.extend([syn_name])
	encoding_pars.connectivity.connections.extend([('parrots', encoding_pars.generator.labels[0])])
	encoding_pars.connectivity.topology_dependent.extend([False])
	encoding_pars.connectivity.conn_specs.extend([conn_specs])
	encoding_pars.connectivity.syn_specs.extend([{}])
	encoding_pars.connectivity.models.extend(['static_synapse'])
	encoding_pars.connectivity.model_pars.extend([{}])
	encoding_pars.connectivity.weight_dist.extend([1.])
	encoding_pars.connectivity.delay_dist.extend([0.1])
	encoding_pars.connectivity.preset_W.extend([presetW])
	if decode:
		encoding_pars.input_decoder = {'encoder_label': 'parrots'}


def set_decoding_defaults(default_set=1, output_resolution=1., to_memory=True, kernel_pars={}, **decoder_pars):
	"""

	:return:
	"""
	if default_set == 1:
		keys = ['decoded_population', 'state_variable', 'filter_time', 'readouts', 'global_sampling_times']
		if not np.mean([n in decoder_pars.keys() for n in keys]).astype(bool) or len(decoder_pars[
			                                                                             'decoded_population']) != \
				len(decoder_pars['state_variable']):
			raise TypeError("Incorrect Decoder Parameters")

		dec_pars = ParameterSet(decoder_pars)
		n_decoders = len(dec_pars.decoded_population)
		if to_memory:
			rec_device = rec_device_defaults(start=0., #kernel_pars['transient_t'] - output_resolution,
			                                 resolution=output_resolution)
		else:
			rec_device = rec_device_defaults(start=0., #kernel_pars['transient_t'] - output_resolution,
			                                 resolution=output_resolution, record_to='file')
		state_specs = []
		for state_var in dec_pars.state_variable:
			if state_var == 'V_m':
				state_specs.append(copy_dict(rec_device, {'model': 'multimeter',
				                                          'record_n': None,
				                                          'record_from': ['V_m'],
				                                          }))
			elif state_var == 'spikes':
				state_specs.append({'tau_m': dec_pars.filter_time, 'interval': output_resolution})
		if 'N' in decoder_pars.keys():
			N = decoder_pars['N']
		else:
			N = len(dec_pars.readouts)
		if len(dec_pars.readout_algorithms) == N:
			readouts = [{'N': N, 'labels': dec_pars.readouts, 'algorithm': dec_pars.readout_algorithms} for n in
			            range(n_decoders)]
		else:
			readouts = [{'N': N, 'labels': dec_pars.readouts, 'algorithm': [
				dec_pars.readout_algorithms[n]]} for n in range(n_decoders)]

		decoding_pars = {
			'description': {'measurements': '{0} state decoders attached to {1}'.format(str(n_decoders),
			                                                        str(dec_pars.decoded_population))},
			'state_extractor': {
				'N': n_decoders,
				'filter_tau': dec_pars.filter_time,
				'source_population': dec_pars.decoded_population,
				'state_variable': dec_pars.state_variable,
				'state_specs': state_specs},
			'readout': readouts,
			'global_sampling_times': dec_pars.global_sampling_times,
		}
	else:
		raise IOError("default_set not defined")

	return ParameterSet(decoding_pars)


def set_report_defaults(default_set, run_type, paths, kernel_pars, neuron_pars, net_pars, connection_pars,
                        encoding_pars, decoding_pars):
	"""

	:param default_set:
	:return:
	"""
	# ==============================================================================================================
	if net_pars.items():
		topology_description = net_pars.description.topology
	else:
		topology_description = ''
	if connection_pars.items():
		connectivity_description = connection_pars.description.connectivity
		plasticity_description = connection_pars.description.plasticity
	else:
		connectivity_description = ''
		plasticity_description = ''
	if neuron_pars.items():
		neurons_description = neuron_pars.description.neurons
		synapse_description = neuron_pars.description.synapses
	else:
		neurons_description = ''
		synapse_description = ''

	input_full_description = ''
	connectivity_full_description = ''
	if encoding_pars.items():
		input_description = encoding_pars.description.general
		input_specific_description = encoding_pars.description.specific
		input_parameters = encoding_pars.description.parameters
		input_generators = list(itertools.chain(encoding_pars.generator.models))
		for n_inp, inp in enumerate(input_generators):
			targets = [xx[0] for xx in encoding_pars.connectivity.connections if xx[1] ==
			           encoding_pars.generator.labels[n_inp]]
			input_full_description += '{0} & {1} & {2}'.format(r'\verb+' + inp + '+', str(targets),
			                                                   input_specific_description + r'\tabularnewline' + ' \hline ')
	else:
		input_description = ''
		input_specific_description = ''
		input_parameters = ''
		input_full_description = ''
	if decoding_pars.items():
		measurements_description = decoding_pars.description.measurements
	else:
		measurements_description = ''

	if isinstance(net_pars['neuron_pars'][0], list):
		neuron_models = [x['model'] for x in list(itertools.chain(*net_pars['neuron_pars'])) if x != 'global' and
		x.has_key('model')]
		N_neurons = list(itertools.chain(net_pars['n_neurons']))
		pops = list(itertools.chain(net_pars['pop_names']))
	else:
		neuron_models = [x['model'] for x in list(itertools.chain(net_pars['neuron_pars'])) if x != 'global' and
		                 x.has_key('model')]
		N_neurons = list(itertools.chain(net_pars['n_neurons']))
		pops = list(itertools.chain(net_pars['pop_names']))

	#### network description (populations) ##############################
	population_full_description = ''
	population_parameters = ''
	for idxxx, nnn in enumerate(pops):
		population_full_description += ('{0} & {1} & {2}'.format(str(nnn), r'\verb+' + str(neuron_models[idxxx]) + '+',
		                                                         str(N_neurons[
			                                                             idxxx])) + r'\tabularnewline' + ' \hline ')
		population_parameters += ('$N^{0}$ & {1} & {2}'.format('{' + str(nnn) + '}', str(N_neurons[idxxx]),
		                                                       str(
			                                                       nnn) + ' Population Size') + r'\tabularnewline' + ' \hline ')
	if connection_pars.items():
		#### Connectivity description ########################################
		connectivity_parameters = r'\epsilon & $0.1$ & Connection Probability (for all synapses)' + r'\tabularnewline' + ' \hline '
		for idx in range(connection_pars['n_synapse_types']):
			connectivity_full_description += ('{0} & {1} & {2} & {3}'.format(str(connection_pars['synapse_names'][idx]),
		                                                                 str(connection_pars['synapse_types'][idx][1]),
		                                                                 str(connection_pars['synapse_types'][idx][0]),
		                                                                 'Random, ') + r'\tabularnewline' + ' \hline ')
			connectivity_parameters += '$w^{0}$ & {1} & Connection strength'.format('{' + str(connection_pars[
				                                                                                  'synapse_names'][
				                                                                                  idx]) + '}', str(
																				connection_pars['weight_dist'][idx])) + \
		                                                                         r'\tabularnewline' + ' \hline '
	else:
		connectivity_parameters = r'' + r'\tabularnewline' + ' \hline '

	if default_set == 1:
		# ### Extract single neuron parameters from dictionaries ###############
		keys = neuron_pars.keys()
		keys.remove('description')
		keys.remove('label')
		neuron_pars_description = {}
		for neuron_name in keys:
			rec_time_constants = dict(rise={}, d1={}, d2={}, rec_bal={}, rec_cond={})
			if hasattr(neuron_pars[neuron_name], 'rec_names'):
				for n_rec in range(len(neuron_pars[neuron_name]['rec_names'])):
					k = "{0}".format(neuron_pars[neuron_name]['rec_names'][n_rec])
					v = neuron_pars[neuron_name]['tau_syn_rise'][n_rec]
					rec_time_constants['rise'].update({k: v})

					v = neuron_pars[neuron_name]['tau_syn_d1'][n_rec]
					rec_time_constants['d1'].update({k: v})

					v = neuron_pars[neuron_name]['tau_syn_d2'][n_rec]
					rec_time_constants['d2'].update({k: v})

					v = neuron_pars[neuron_name]['rec_bal'][n_rec]
					rec_time_constants['rec_bal'].update({k: v})

					v = neuron_pars[neuron_name]['rec_cond'][n_rec]
					rec_time_constants['rec_cond'].update({k: v})

				neuron_pars_description.update({neuron_name: {
					'Parameters': [
						Parameter(name='C_{m}', value=neuron_pars[neuron_name]['C_m'], units=r'\pF'),
						Parameter(name='V_{\mathrm{rest}}', value=neuron_pars[neuron_name]['E_L'], units=r'\mV'),
						Parameter(name='V_{\mathrm{th}}', value=neuron_pars[neuron_name]['V_th'], units=r'\mV'),
						Parameter(name='V_{\mathrm{reset}}', value=neuron_pars[neuron_name]['V_reset'], units=r'\mV'),
						Parameter(name='g_{\mathrm{leak}}', value=neuron_pars[neuron_name]['g_L'], units=r'\nS'),
						Parameter(name='a', value=neuron_pars[neuron_name]['a'], units=r'\nS'),
						Parameter(name='b', value=neuron_pars[neuron_name]['b'], units=r'\mV'),
						Parameter(name='N_{\syn}', value=len(neuron_pars[neuron_name]['rec_bal']), units=None),
						Parameter(name='R_{\syn}', value=neuron_pars[neuron_name]['rec_names'], units=None),
						Parameter(name='E_{\syn}', value=neuron_pars[neuron_name]['rec_reversal'], units=r'\mV'),
						Parameter(name=r'\bar{g}_{\syn}', value=rec_time_constants['rec_cond'], units=r'\nS'),
						Parameter(name=r'\tau^{r}_{\syn}', value=rec_time_constants['rise'], units=r'\ms'),
						Parameter(name=r'\tau^{d1}_{\syn}', value=rec_time_constants['d1'], units=r'\ms'),
						Parameter(name=r'\tau^{d2}_{\syn}', value=rec_time_constants['d2'], units=r'\ms'),
						Parameter(name='r_{\syn}', value=rec_time_constants['rec_bal'], units=None),
						Parameter(name='t_{\mathrm{ref}}', value=neuron_pars[neuron_name]['t_ref'], units=r'\ms'),
						Parameter(name=r'\tau_{w}', value=neuron_pars[neuron_name]['tau_w'], units=r'\ms')],
					'Descriptions': [
						'Membrane Capacitance',
						'Resting Membrane Potential',
						'Fixed Firing Threshold',
						'Reset Potential',
						'Leak Conductance',
						'Sub-threshold intrinsic adaptation parameter',
						'Spike-triggered intrinsic adaptation parameter',
						'Absolute number of synaptic receptors',
						'Synaptic Receptor types',
						'Reversal Potentials',
						'Fixed synaptic conductance',
						'Synaptic conductance rise time constants',
						'Synaptic conductance first decay time constants',
						'Synaptic conductance second decay time constants',
						'Balance between decay time constants',
						'Absolute Refractory time',
						'Adaptation time constant']}
				})

		neuron_pars = list(itertools.chain(net_pars['neuron_pars']))
		if len(neuron_pars) == 1:
			neuron_name = [pops[0]]
		else:
			if np.mean([compare_dict(x, y) for x, y in zip(neuron_pars, neuron_pars[1:])]) == 1.:
				neuron_name = [pops[0]]
			else:
				neuron_name = [n_pop for idx_pop, n_pop in enumerate(pops) if neuron_pars[idx_pop][
					'model']!='parrot_neuron']

		neuron_parameters = ''
		for idd, nam in enumerate(neuron_name):
			neuron_parameters += (
			'{0} & {1} & {2}'.format('Name', str(nam), 'Neuron name') + r'\tabularnewline ' + ' \hline ')

			for iid, par in enumerate(neuron_pars_description[nam]['Parameters']):
				if par.name == 'R_{\syn}':
					name = par.name
					value = ''
					for i in par.value:
						value += str(i) + ', '
					units = ''
				elif par.name == r'\tau^{r}_{\syn}' or par.name == r'\tau^{d1}_{\syn}' or par.name == r'\tau^{d2}_{\syn}' or \
								par.name == 'r_{\syn}' or par.name == r'\bar{g}_{\syn}':
					name = par.name
					value = str(par.value.values())
					units = str(par.units)
				else:
					name = par.name
					value = str(par.value)
					units = str(par.units)

				neuron_parameters += ('{0} & {1} & {2}'.format('$' + name + '$', '$' + value + units + '$',
				                                               neuron_pars_description[nam]['Descriptions'][iid]) +
				                      r'\tabularnewline ' + ' \hline ')

		#### Synapse parameters ######################################
		synapse_parameters = 'None & None & None' + r'\tabularnewline' + ' \hline '

		#### Plasticity Parameters ###################################
		plasticity_parameters = 'None & None & None' + r'\tabularnewline' + ' \hline '

		########################################
		report_pars = {
			'report_templates_path': paths[run_type]['report_templates_path'],
			'report_path': paths[run_type]['report_path'],
			'report_filename': kernel_pars['data_prefix'] + '.tex',
			'table_fields': {
				'{{ net_populations_description }}': '{0} Main Populations: {1}'.format(str(net_pars['n_populations']),
				                                                                        str(net_pars['pop_names'])),
				'{{ net_populations_topology }}': topology_description,
				'{{ net_connectivity_description }}': connectivity_description,
				'{{ net_neuron_models }}': neurons_description,
				'{{ net_synapse_models }}': synapse_description,
				'{{ net_plasticity }}': plasticity_description,
				'{{ input_general_description }}': input_description,
				'{{ input_description }}': input_full_description,
				'{{ measurements_description }}': measurements_description,

				'{{ population_names_elements_sizes }}': population_full_description,
				'{{ connectivity_name_src_tgt_pattern }}': connectivity_full_description,
				'{{ measurements }}': measurements_description,
				'{{ population_parameters }}': population_parameters,
				'{{ connectivity_parameters }}': connectivity_parameters,
				'{{ neuron_parameters }}': neuron_parameters,
				'{{ synapse_parameters }}': synapse_parameters,
				'{{ plasticity_parameters }}': plasticity_parameters,
				'{{ input_parameters }}': input_parameters,
			},
		}
	elif default_set == 2:
		# ### Extract single neuron parameters from dictionaries ###############
		keys = neuron_pars.keys()
		keys.remove('description')
		keys.remove('label')
		neuron_pars_description = {}
		for neuron_name in keys:
			neuron_pars_description.update({neuron_name: {
				'Parameters': [
					Parameter(name='\tau_{m}', value=neuron_pars[neuron_name]['tau_m'], units=r'\ms'),
					Parameter(name='V_{\mathrm{rest}}', value=neuron_pars[neuron_name]['E_L'], units=r'\mV'),
					Parameter(name='V_{\mathrm{th}_{0}}', value=neuron_pars[neuron_name]['omega'], units=r'\mV'),
					Parameter(name='\alpha_{1}', value=neuron_pars[neuron_name]['alpha_1'], units=None),
					Parameter(name='\alpha_{2}', value=neuron_pars[neuron_name]['alpha_2'], units=None),
					Parameter(name='\beta', value=neuron_pars[neuron_name]['beta'], units=None),
					Parameter(name=r'\tau^{E}_{\syn}', value=neuron_pars[neuron_name]['tau_syn_ex'], units=r'\ms'),
					Parameter(name=r'\tau^{I}_{\syn}', value=neuron_pars[neuron_name]['tau_syn_in'], units=r'\ms'),
					Parameter(name='t_{\mathrm{ref}}', value=neuron_pars[neuron_name]['t_ref'], units=r'\ms')],
				'Descriptions': [
					'Membrane Capacitance',
					'Resting Membrane Potential',
					'Initial Firing Threshold',
					'Weight of the first adaptation time constant',
					'Weight of the second adaptation time constant',
					'Weight of Voltage-dependent term',
					'Synaptic conductance rise time constants',
					'Synaptic conductance first decay time constants',
					'Absolute Refractory time']}
			})

		neuron_pars = list(itertools.chain(net_pars['neuron_pars']))
		if len(neuron_pars) == 1:
			neuron_name = [pops[0]]
		else:
			if np.mean([compare_dict(x, y) for x, y in zip(neuron_pars, neuron_pars[1:])]) == 1.:
				neuron_name = [pops[0]]
			else:
				neuron_name = [n_pop for idx_pop, n_pop in enumerate(pops) if neuron_pars[idx_pop][
					'model'] != 'parrot_neuron']

		neuron_parameters = ''
		for idd, nam in enumerate(neuron_name):
			neuron_parameters += (
				'{0} & {1} & {2}'.format('Name', str(nam), 'Neuron name') + r'\tabularnewline ' + ' \hline ')

			for iid, par in enumerate(neuron_pars_description[nam]['Parameters']):
				name = par.name
				value = str(par.value)
				units = str(par.units)

				neuron_parameters += ('{0} & {1} & {2}'.format('$' + name + '$', '$' + value + units + '$',
				                                               neuron_pars_description[nam]['Descriptions'][iid]) +
				                      r'\tabularnewline ' + ' \hline ')

		#### Synapse parameters ######################################
		synapse_parameters = 'None & None & None' + r'\tabularnewline' + ' \hline '

		#### Plasticity Parameters ###################################
		plasticity_parameters = 'None & None & None' + r'\tabularnewline' + ' \hline '

		########################################
		report_pars = {
			'report_templates_path': paths[run_type]['report_templates_path'],
			'report_path': paths[run_type]['report_path'],
			'report_filename': kernel_pars['data_prefix'] + '.tex',
			'table_fields': {
				'{{ net_populations_description }}': '{0} Main Populations: {1}'.format(str(net_pars['n_populations']),
				                                                                        str(net_pars['pop_names'])),
				'{{ net_populations_topology }}': topology_description,
				'{{ net_connectivity_description }}': connectivity_description,
				'{{ net_neuron_models }}': neurons_description,
				'{{ net_synapse_models }}': synapse_description,
				'{{ net_plasticity }}': plasticity_description,
				'{{ input_general_description }}': input_description,
				'{{ input_description }}': input_full_description,
				'{{ measurements_description }}': measurements_description,

				'{{ population_names_elements_sizes }}': population_full_description,
				'{{ connectivity_name_src_tgt_pattern }}': connectivity_full_description,
				'{{ measurements }}': measurements_description,
				'{{ population_parameters }}': population_parameters,
				'{{ connectivity_parameters }}': connectivity_parameters,
				'{{ neuron_parameters }}': neuron_parameters,
				'{{ synapse_parameters }}': synapse_parameters,
				'{{ plasticity_parameters }}': plasticity_parameters,
				'{{ input_parameters }}': input_parameters,
			},
		}
	elif default_set == 3:
		# ### Extract single neuron parameters from dictionaries ###############
		keys = neuron_pars.keys()
		keys.remove('description')
		keys.remove('label')
		neuron_pars_description = {}
		for neuron_name in keys:
			neuron_pars_description.update({neuron_name: {
				'Parameters': [
					Parameter(name='C_{m}', value=neuron_pars[neuron_name]['C_m'], units=r'\pF'),
					Parameter(name='V_{\mathrm{rest}}', value=neuron_pars[neuron_name]['E_L'], units=r'\mV'),
					Parameter(name='V_{\mathrm{reset}}', value=neuron_pars[neuron_name]['V_reset'], units=r'\mV'),
					Parameter(name='V_{\mathrm{th}}', value=neuron_pars[neuron_name]['V_th'], units=r'\mV'),
					Parameter(name=r'\tau_{m}', value=neuron_pars[neuron_name]['tau_m'], units=r'\ms'),
					Parameter(name=r'\tau_{E}', value=neuron_pars[neuron_name]['tau_syn_ex'], units=r'\ms'),
					Parameter(name=r'\tau_{I}', value=neuron_pars[neuron_name]['tau_syn_in'], units=r'\ms'),
					Parameter(name='t_{\mathrm{ref}}', value=neuron_pars[neuron_name]['t_ref'], units=r'\ms')],
				'Descriptions': [
					'Membrane Capacitance',
					'Resting Membrane Potential',
					'Reset Potential',
					'Firing Threshold',
					'Membrane time constant',
					'Excitatory Synaptic time constants',
					'Inhibitory Synaptic time constants',
					'Absolute Refractory time']}
			})

		neuron_pars = list(itertools.chain(net_pars['neuron_pars']))
		if len(neuron_pars) == 1:
			neuron_name = [pops[0]]
		else:
			if np.mean([compare_dict(x, y) for x, y in zip(neuron_pars, neuron_pars[1:])]) == 1.:
				neuron_name = [pops[0]]
			else:
				neuron_name = [n_pop for idx_pop, n_pop in enumerate(pops) if neuron_pars[idx_pop][
					'model'] != 'parrot_neuron']

		neuron_parameters = ''
		for idd, nam in enumerate(neuron_name):
			neuron_parameters += (
				'{0} & {1} & {2}'.format('Name', str(nam), 'Neuron name') + r'\tabularnewline ' + ' \hline ')

			for iid, par in enumerate(neuron_pars_description[nam]['Parameters']):
				name = par.name
				value = str(par.value)
				units = str(par.units)

				neuron_parameters += ('{0} & {1} & {2}'.format('$' + name + '$', '$' + value + units + '$',
				                                               neuron_pars_description[nam]['Descriptions'][iid]) +
				                      r'\tabularnewline ' + ' \hline ')

		#### Synapse parameters ######################################
		synapse_parameters = 'None & None & None' + r'\tabularnewline' + ' \hline '

		#### Plasticity Parameters ###################################
		plasticity_parameters = 'None & None & None' + r'\tabularnewline' + ' \hline '

		########################################
		report_pars = {
			'report_templates_path': paths[run_type]['report_templates_path'],
			'report_path': paths[run_type]['report_path'],
			'report_filename': kernel_pars['data_prefix'] + '.tex',
			'table_fields': {
				'{{ net_populations_description }}': '{0} Main Populations: {1}'.format(str(net_pars['n_populations']),
				                                                                        str(net_pars['pop_names'])),
				'{{ net_populations_topology }}': topology_description,
				'{{ net_connectivity_description }}': connectivity_description,
				'{{ net_neuron_models }}': neurons_description,
				'{{ net_synapse_models }}': synapse_description,
				'{{ net_plasticity }}': plasticity_description,
				'{{ input_general_description }}': input_description,
				'{{ input_description }}': input_full_description,
				'{{ measurements_description }}': measurements_description,

				'{{ population_names_elements_sizes }}': population_full_description,
				'{{ connectivity_name_src_tgt_pattern }}': connectivity_full_description,
				'{{ measurements }}': measurements_description,
				'{{ population_parameters }}': population_parameters,
				'{{ connectivity_parameters }}': connectivity_parameters,
				'{{ neuron_parameters }}': neuron_parameters,
				'{{ synapse_parameters }}': synapse_parameters,
				'{{ plasticity_parameters }}': plasticity_parameters,
				'{{ input_parameters }}': input_parameters,
			},
		}
	elif default_set == 4:
		# ### Extract single neuron parameters from dictionaries ###############
		keys = neuron_pars.keys()
		keys.remove('description')
		keys.remove('label')
		neuron_pars_description = {}
		for neuron_name in keys:
			neuron_pars_description.update({neuron_name: {
				'Parameters': [
					Parameter(name='C_{m}', value=neuron_pars[neuron_name]['C_m'], units=r'\pF'),
					Parameter(name='V_{\mathrm{rest}}', value=neuron_pars[neuron_name]['E_L'], units=r'\mV'),
					Parameter(name='V_{\mathrm{th}}', value=neuron_pars[neuron_name]['V_th'], units=r'\mV'),
					Parameter(name='V_{\mathrm{reset}}', value=neuron_pars[neuron_name]['V_reset'], units=r'\mV'),
					Parameter(name='g_{\mathrm{leak}}', value=neuron_pars[neuron_name]['g_L'], units=r'\nS'),
					Parameter(name='\Delta_{T}', value=neuron_pars[neuron_name]['Delta_T'], units=r'None'),
					Parameter(name='g_{E}', value=neuron_pars[neuron_name]['g_ex'], units=r'\nS'),
					Parameter(name='g_{I}', value=neuron_pars[neuron_name]['g_in'], units=r'\nS'),
					Parameter(name='E_{E}', value=neuron_pars[neuron_name]['E_ex'], units=r'\mV'),
					Parameter(name='E_{I}', value=neuron_pars[neuron_name]['E_in'], units=r'\mV'),
					Parameter(name='a', value=neuron_pars[neuron_name]['a'], units=r'\nS'),
					Parameter(name='b', value=neuron_pars[neuron_name]['b'], units=r'\mV'),
					Parameter(name='t_{\mathrm{ref}}', value=neuron_pars[neuron_name]['t_ref'], units=r'\ms'),
					Parameter(name=r'\tau_{E}', value=neuron_pars[neuron_name]['tau_syn_ex'], units=r'\ms'),
					Parameter(name=r'\tau_{I}', value=neuron_pars[neuron_name]['tau_syn_in'], units=r'\ms'),
					Parameter(name=r'\tau_{w}', value=neuron_pars[neuron_name]['tau_w'], units=r'\ms')],
				'Descriptions': [
					'Membrane Capacitance',
					'Resting Membrane Potential',
					'Fixed Firing Threshold',
					'Reset Potential',
					'Leak Conductance',
					'Adaptation sharpness parameter',
					'Excitatory conductance',
					'Inhibitory conductance',
					'Excitatory reversal potential',
					'Inhibitory reversal potential',
					'Sub-threshold intrinsic adaptation parameter',
					'Spike-triggered intrinsic adaptation parameter',
					'Absolute Refractory time',
					'Excitatory synaptic time constant',
					'Inhbitory synaptic time constant',
					'Adaptation time constant']}
			})

		neuron_pars = list(itertools.chain(net_pars['neuron_pars']))
		if len(neuron_pars) == 1:
			neuron_name = [pops[0]]
		else:
			if np.mean([compare_dict(x, y) for x, y in zip(neuron_pars, neuron_pars[1:])]) == 1.:
				neuron_name = [pops[0]]
			else:
				neuron_name = [n_pop for idx_pop, n_pop in enumerate(pops) if neuron_pars[idx_pop][
					'model'] != 'parrot_neuron']

		neuron_parameters = ''
		for idd, nam in enumerate(neuron_name):
			neuron_parameters += (
				'{0} & {1} & {2}'.format('Name', str(nam), 'Neuron name') + r'\tabularnewline ' + ' \hline ')

			for iid, par in enumerate(neuron_pars_description[nam]['Parameters']):
				name = par.name
				value = str(par.value)
				units = str(par.units)

				neuron_parameters += ('{0} & {1} & {2}'.format('$' + name + '$', '$' + value + units + '$',
				                                               neuron_pars_description[nam]['Descriptions'][iid]) +
				                      r'\tabularnewline ' + ' \hline ')

		#### Synapse parameters ######################################
		synapse_parameters = 'None & None & None' + r'\tabularnewline' + ' \hline '

		#### Plasticity Parameters ###################################
		plasticity_parameters = 'None & None & None' + r'\tabularnewline' + ' \hline '

		########################################
		report_pars = {
			'report_templates_path': paths[run_type]['report_templates_path'],
			'report_path': paths[run_type]['report_path'],
			'report_filename': kernel_pars['data_prefix'] + '.tex',
			'table_fields': {
				'{{ net_populations_description }}': '{0} Main Populations: {1}'.format(str(net_pars['n_populations']),
				                                                                        str(net_pars['pop_names'])),
				'{{ net_populations_topology }}': topology_description,
				'{{ net_connectivity_description }}': connectivity_description,
				'{{ net_neuron_models }}': neurons_description,
				'{{ net_synapse_models }}': synapse_description,
				'{{ net_plasticity }}': plasticity_description,
				'{{ input_general_description }}': input_description,
				'{{ input_description }}': input_full_description,
				'{{ measurements_description }}': measurements_description,

				'{{ population_names_elements_sizes }}': population_full_description,
				'{{ connectivity_name_src_tgt_pattern }}': connectivity_full_description,
				'{{ measurements }}': measurements_description,
				'{{ population_parameters }}': population_parameters,
				'{{ connectivity_parameters }}': connectivity_parameters,
				'{{ neuron_parameters }}': neuron_parameters,
				'{{ synapse_parameters }}': synapse_parameters,
				'{{ plasticity_parameters }}': plasticity_parameters,
				'{{ input_parameters }}': input_parameters,
			},
		}
	elif default_set == 5:
		# ### Extract single neuron parameters from dictionaries ###############
		keys = neuron_pars.keys()
		keys.remove('description')
		keys.remove('label')
		neuron_pars_description = {}
		for neuron_name in keys:
			neuron_pars_description.update({neuron_name: {
				'Parameters': [
					Parameter(name='C_{m}', value=neuron_pars[neuron_name]['C_m'], units=r'\pF'),
					Parameter(name='V_{\mathrm{rest}}', value=neuron_pars[neuron_name]['E_L'], units=r'\mV'),
					Parameter(name='V_{\mathrm{reset}}', value=neuron_pars[neuron_name]['V_reset'], units=r'\mV'),
					Parameter(name='V_{\mathrm{th}}', value=neuron_pars[neuron_name]['V_th'], units=r'\mV'),
					Parameter(name=r'g_{\mathrm{leak}}', value=neuron_pars[neuron_name]['g_L'], units=r'\ms'),
					Parameter(name=r'E_{E}', value=neuron_pars[neuron_name]['E_ex'], units=r'\mV'),
					Parameter(name=r'E_{I}', value=neuron_pars[neuron_name]['E_in'], units=r'\mV'),
					Parameter(name=r'\tau_{E}', value=neuron_pars[neuron_name]['tau_syn_ex'], units=r'\ms'),
					Parameter(name=r'\tau_{I}', value=neuron_pars[neuron_name]['tau_syn_in'], units=r'\ms'),
					Parameter(name='t_{\mathrm{ref}}', value=neuron_pars[neuron_name]['t_ref'], units=r'\ms')],
				'Descriptions': [
					'Membrane Capacitance',
					'Resting Membrane Potential',
					'Reset Potential',
					'Firing Threshold',
					'Membrane leak conductance',
					'Excitatory Reversal potential',
					'Inhibitory Reversal potential',
					'Excitatory Synaptic time constants',
					'Inhibitory Synaptic time constants',
					'Absolute Refractory time']}
			})

		neuron_pars = list(itertools.chain(net_pars['neuron_pars']))
		if len(neuron_pars) == 1:
			neuron_name = [pops[0]]
		else:
			if np.mean([compare_dict(x, y) for x, y in zip(neuron_pars, neuron_pars[1:])]) == 1.:
				neuron_name = [pops[0]]
			else:
				neuron_name = [n_pop for idx_pop, n_pop in enumerate(pops) if neuron_pars[idx_pop][
					'model'] != 'parrot_neuron']

		neuron_parameters = ''
		for idd, nam in enumerate(neuron_name):
			neuron_parameters += (
				'{0} & {1} & {2}'.format('Name', str(nam), 'Neuron name') + r'\tabularnewline ' + ' \hline ')

			for iid, par in enumerate(neuron_pars_description[nam]['Parameters']):
				name = par.name
				value = str(par.value)
				units = str(par.units)

				neuron_parameters += ('{0} & {1} & {2}'.format('$' + name + '$', '$' + value + units + '$',
				                                               neuron_pars_description[nam]['Descriptions'][iid]) +
				                      r'\tabularnewline ' + ' \hline ')

		#### Synapse parameters ######################################
		synapse_parameters = 'None & None & None' + r'\tabularnewline' + ' \hline '

		#### Plasticity Parameters ###################################
		plasticity_parameters = 'None & None & None' + r'\tabularnewline' + ' \hline '

		########################################
		report_pars = {
			'report_templates_path': paths[run_type]['report_templates_path'],
			'report_path': paths[run_type]['report_path'],
			'report_filename': kernel_pars['data_prefix'] + '.tex',
			'table_fields': {
				'{{ net_populations_description }}': '{0} Main Populations: {1}'.format(str(net_pars['n_populations']),
				                                                                        str(net_pars['pop_names'])),
				'{{ net_populations_topology }}': topology_description,
				'{{ net_connectivity_description }}': connectivity_description,
				'{{ net_neuron_models }}': neurons_description,
				'{{ net_synapse_models }}': synapse_description,
				'{{ net_plasticity }}': plasticity_description,
				'{{ input_general_description }}': input_description,
				'{{ input_description }}': input_full_description,
				'{{ measurements_description }}': measurements_description,

				'{{ population_names_elements_sizes }}': population_full_description,
				'{{ connectivity_name_src_tgt_pattern }}': connectivity_full_description,
				'{{ measurements }}': measurements_description,
				'{{ population_parameters }}': population_parameters,
				'{{ connectivity_parameters }}': connectivity_parameters,
				'{{ neuron_parameters }}': neuron_parameters,
				'{{ synapse_parameters }}': synapse_parameters,
				'{{ plasticity_parameters }}': plasticity_parameters,
				'{{ input_parameters }}': input_parameters,
			},
		}
	elif default_set == 6:
		# ### Extract single neuron parameters from dictionaries ###############
		keys = neuron_pars.keys()
		keys.remove('description')
		keys.remove('label')
		neuron_pars_description = {}
		for neuron_name in keys:
			neuron_pars_description.update({neuron_name: {
				'Parameters': [
					Parameter(name='C_{m}', value=neuron_pars[neuron_name]['C_m'], units=r'\pF'),
					Parameter(name='V_{\mathrm{rest}}', value=neuron_pars[neuron_name]['E_L'], units=r'\mV'),
					Parameter(name='V_{\mathrm{reset}}', value=neuron_pars[neuron_name]['V_reset'], units=r'\mV'),
					Parameter(name='V_{\mathrm{th}}', value=neuron_pars[neuron_name]['V_th'], units=r'\mV'),
					Parameter(name=r'g_{\mathrm{leak}}', value=neuron_pars[neuron_name]['g_L'], units=r'\nS'),
					Parameter(name=r'E_{E}', value=neuron_pars[neuron_name]['E_ex'], units=r'\mV'),
					Parameter(name=r'E_{I}', value=neuron_pars[neuron_name]['E_in'], units=r'\mV'),
					Parameter(name=r'\tau_{E}', value=neuron_pars[neuron_name]['tau_syn_ex'], units=r'\ms'),
					Parameter(name=r'\tau_{I}', value=neuron_pars[neuron_name]['tau_syn_in'], units=r'\ms'),
					Parameter(name='t_{\mathrm{ref}}', value=neuron_pars[neuron_name]['t_ref'], units=r'\ms'),
					Parameter(name='q_{\mathrm{sra}}', value=neuron_pars[neuron_name]['q_sfa'], units=r'\nS'),
					Parameter(name='q_{\mathrm{ref}}', value=neuron_pars[neuron_name]['q_rr'], units=r'\nS'),
					Parameter(name='\tau_{\mathrm{sra}}', value=neuron_pars[neuron_name]['tau_sfa'], units=r'\ms'),
					Parameter(name='\tau_{\mathrm{ref}}', value=neuron_pars[neuron_name]['tau_rr'], units=r'\ms'),
					Parameter(name='E_{\mathrm{sra}}', value=neuron_pars[neuron_name]['E_sfa'], units=r'\mV'),
					Parameter(name='E_{\mathrm{rr}}', value=neuron_pars[neuron_name]['E_rr'], units=r'\mV'),
					Parameter(name='I_{\mathrm{E}}', value=neuron_pars[neuron_name]['I_e'], units=r'\pA')],
				'Descriptions': [
					'Membrane Capacitance',
					'Resting Membrane Potential',
					'Reset Potential',
					'Firing Threshold',
					'Membrane leak conductance',
					'Excitatory Reversal potential',
					'Inhibitory Reversal potential',
					'Excitatory Synaptic time constants',
					'Inhibitory Synaptic time constants',
					'Absolute Refractory time',
					'Quantal increment of spike rate adaptation',
					'Quantal increment of relative refractory mechanism',
					'Spike rate adaptation time constant',
					'Relative refractory period time constant',
					'Reversal potential for SRA mechanism',
					'Reversal potential for RR mechanism',
					'Background input current']}
			})
		neuron_pars = list(itertools.chain(net_pars['neuron_pars']))
		if len(neuron_pars) == 1:
			neuron_name = [pops[0]]
		else:
			if np.mean([compare_dict(x, y) for x, y in zip(neuron_pars, neuron_pars[1:])]) == 1.:
				neuron_name = [pops[0]]
			else:
				neuron_name = [n_pop for idx_pop, n_pop in enumerate(pops) if neuron_pars[idx_pop][
					'model'] != 'parrot_neuron']

		neuron_parameters = ''
		for idd, nam in enumerate(neuron_name):
			neuron_parameters += (
				'{0} & {1} & {2}'.format('Name', str(nam), 'Neuron name') + r'\tabularnewline ' + ' \hline ')

			for iid, par in enumerate(neuron_pars_description[nam]['Parameters']):
				name = par.name
				value = str(par.value)
				units = str(par.units)

				neuron_parameters += ('{0} & {1} & {2}'.format('$' + name + '$', '$' + value + units + '$',
				                                               neuron_pars_description[nam]['Descriptions'][iid]) +
				                      r'\tabularnewline ' + ' \hline ')

		#### Synapse parameters ######################################
		synapse_parameters = 'None & None & None' + r'\tabularnewline' + ' \hline '

		#### Plasticity Parameters ###################################
		plasticity_parameters = 'None & None & None' + r'\tabularnewline' + ' \hline '

		########################################
		report_pars = {
			'report_templates_path': paths[run_type]['report_templates_path'],
			'report_path': paths[run_type]['report_path'],
			'report_filename': kernel_pars['data_prefix'] + '.tex',
			'table_fields': {
				'{{ net_populations_description }}': '{0} Main Populations: {1}'.format(
					str(net_pars['n_populations']),
					str(net_pars['pop_names'])),
				'{{ net_populations_topology }}': topology_description,
				'{{ net_connectivity_description }}': connectivity_description,
				'{{ net_neuron_models }}': neurons_description,
				'{{ net_synapse_models }}': synapse_description,
				'{{ net_plasticity }}': plasticity_description,
				'{{ input_general_description }}': input_description,
				'{{ input_description }}': input_full_description,
				'{{ measurements_description }}': measurements_description,
				'{{ population_names_elements_sizes }}': population_full_description,
				'{{ connectivity_name_src_tgt_pattern }}': connectivity_full_description,
				'{{ measurements }}': measurements_description,
				'{{ population_parameters }}': population_parameters,
				'{{ connectivity_parameters }}': connectivity_parameters,
				'{{ neuron_parameters }}': neuron_parameters,
				'{{ synapse_parameters }}': synapse_parameters,
				'{{ plasticity_parameters }}': plasticity_parameters,
				'{{ input_parameters }}': input_parameters,
			},
		}
	return ParameterSet(report_pars)
