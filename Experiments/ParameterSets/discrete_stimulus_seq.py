import numpy as np
import sys
from Modules.parameters import *
import itertools
import sys
wd = '/home/neuro/Desktop/CODE/NetworkSimulationTestbed/Experiments/Parameters_files/'
sys.path.append(wd)
from Modules.parameters import *
import itertools
from Experiments.Parameters_files.Preset.Paths import paths

__author__ = 'duarte'
"""
discrete_stimulus_seq
- to run with function .. in Computation
or with DiscreteInput_NStep_Analysis.py (debugging)
"""

run = 'local'
nSteps = 10

nodes = 1
ppn = 8
mem = 32
N_vp = nodes * ppn
walltime = '00:20:00:00'
queue = 'singlenode'

np_seed = np.random.randint(1000000000)+1
np.random.seed(np_seed)
msd = np.random.randint(100000000000)

kernel_pars = {
			'resolution': 0.1,
			'sim_time': 2000.0,
			'transient_t': 0.,

			'data_prefix': 'task_test',
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
				'parameters_file': paths[run]['parameters_file'],
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

E_neuron = {
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

I_neuron = {
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
#############################################################################################################
# NETWORK Parameters
# ============================================================================================================
N = 1250
nE = 0.8 * N
nI = 0.2 * N

net_pars = {
	'n_populations': 2,
	'pop_names': ['E', 'I'],
	'n_neurons': [int(nE), int(nI)],
	'neuron_pars': [E_neuron, E_neuron],
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
	'analog_device_pars': [copy_dict(rec_devices,
	                                 {'model': 'multimeter',
	                                  'record_from': ['V_m', 'I_ex', 'I_in'],
	                                  'record_n': 1,
	                                  'record_to': ['memory'],
	                                  'interval': 0.1,
	                                  'label': 'E_population_multimeter'}),
	                       copy_dict(rec_devices,
	                                 {'model': 'multimeter',
	                                  'record_from': ['V_m', 'I_ex', 'I_in'],
	                                  'record_n': 1,
	                                  'record_to': ['memory'],
	                                  'label': 'I_population_multimeter'})],
}

#############################################################################################################
# SYNAPSE Parameters
# ============================================================================================================
delay = 1.5
epsilon = 0.1
kE = epsilon * nE
kI = epsilon * nI
kX = epsilon * nE
synapses = {'static': {}}

gE = 3.
gI = 2.

wE_AMPA = 1.
wE_GABA = gE * wE_AMPA
wI_AMPA = 1.
wI_GABA = gI * wI_AMPA

conn_specs = {'rule': 'pairwise_bernoulli', 'p': epsilon}
syn_specs = {'receptor_type': 1}
##############################################################
connection_pars = {
	'n_synapse_types': 4,
	'synapse_types': [
		('E', 'E'),
		('E', 'I'),
		('I', 'E'),
		('I', 'I')],
	'synapse_names': ['EE', 'EI', 'IE', 'II'],
	'topology_dependent': [False, False, False, False],
	'models': [
		'static_synapse',
		'static_synapse',
		'static_synapse',
		'static_synapse'],
	'model_pars': [
		synapses['static'],
		synapses['static'],
		synapses['static'],
		synapses['static']],
	'pre_computedW': [None, None, None, None],
	'weight_dist': [wE_AMPA,
	                wE_GABA,
	                wI_AMPA,
	                wI_GABA],
	'delay_dist': [delay,
	               delay,
	               delay,
	               delay],
	'conn_specs': [
		copy_dict(conn_specs, {}),
		copy_dict(conn_specs, {}),
		copy_dict(conn_specs, {}),
		copy_dict(conn_specs, {})],
	'syn_specs': [
		copy_dict(syn_specs, {'receptor_type': 1}),
		copy_dict(syn_specs, {'receptor_type': 2}),
		copy_dict(syn_specs, {'receptor_type': 1}),
		copy_dict(syn_specs, {'receptor_type': 2})]}

##############################################################################################################
# STIMULI Parameters
# ============================================================================================================
n_trials = 10
n_discard = 2

stim_pars = {'n_stim': 3,
 		    'elements': ['b', 'a', 'd'],
	        'grammar': None, # {'pre_set': None,
					    #'pre_set_path': None},
			'full_set_length': int(n_trials + n_discard),
			'transient_set_length': int(n_discard),
		    'train_set_length': int(n_trials * 0.8),
		    'test_set_length': int(n_trials * 0.2),
}

##############################################################################################################
# INPUT and ENCODING Parameters
# ============================================================================================================
inp_resolution = 1.
inp_amplitude = 50.
input_pars = {'signal': {
				'N': 3,
					'durations': [(np.random.uniform, {'low': 500., 'high': 500., 'size': n_trials})],
					'i_stim_i': [(np.random.uniform, {'low': 100., 'high': 100., 'size': n_trials - 1})],
					'kernel': ('double_exp', {'tau_1': 20., 'tau_2': 100.}), #('box', {}),
					'start_time': 0.,
					'stop_time': sys.float_info.max,
					'max_amplitude': [(np.random.uniform, {'low': 50., 'high': 50., 'size': n_trials})],
					'min_amplitude': 0.,
					'resolution': inp_resolution},
				'noise': {
					'N': 0,
					'noise_source': ['GWN'],
					'noise_pars': {'amplitude': 5., 'mean': 1., 'std': 0.1},
	                'rectify': True,
	                'start_time': 0.,
	                'stop_time': sys.float_info.max,
	                'resolution': inp_resolution, }}
input_conn_specs = {
	'autapses': False,
	'multapses': False,
	'rule': 'all_to_all'}
input_syn_specs = {
	'model': 'static_synapse',
	'model_pars': {},
	'delay': {
		'distribution': 'normal',
		'mu': 0.1,
		'sigma': 0.0},
	'weight': {
		'distribution': 'normal',
		'mu': 1.,
		'sigma': 0.0}
}
input_device_specs = {
	'precise_times': False,
	'allow_offgrid_times': True}

#########################
filter_tau = 20.
readout_labels = []

for nn in range(int(nSteps/2.)):
	readout_labels.append('mem{0}'.format(nn+1))
readout_labels.append('class0')
for nn in range(int(nSteps/2.)):
	readout_labels.append('pred{0}'.format(nn+1))
########################

encoding_pars = {
	'encoder': {
		'N': 2,
		'labels': ['NEF'],
		'models': ['NEF'],
		'model_pars': [None],
		'n_neurons': [1000],
		'neuron_pars': [copy_dict(E_neuron, {'I_e': 400.})],
		'topology': [False],
		'topology_dict': [None],
		'record_spikes': [False],
		'spike_device_pars': [{}],
		'record_analogs': [False],
		'analog_device_pars': [None]},

	'generator': {
		'N': 1,
		'labels': ['StepGen'],
		'models': ['step_current_generator'],
		'model_pars': [{'start': 0., 'stop': sys.float_info.max, 'origin': 0.}],
		'topology': [False],
		'topology_pars': [None]},

	'connectivity': {
		'connections': [('NEF', 'StepGen'),
						('E', 'NEF'),
						('I', 'NEF')],
		'topology_dependent': [False,
		                       False,
		                       False],
		'conn_specs': [copy_dict(conn_specs, {}),
		               copy_dict(conn_specs, {}),
		               copy_dict(conn_specs, {})],
		'models': [input_syn_specs['model'],
		           input_syn_specs['model'],
		           input_syn_specs['model']],
		'model_pars': [input_syn_specs['model_pars'],
		               input_syn_specs['model_pars'],
		               input_syn_specs['model_pars']],
		'weight_dist': [copy_dict(input_syn_specs['weight'], {}),
		                wE_AMPA,
		                wE_AMPA],
		'delay_dist': [0.1,
		               delay,
		               delay],
		'preset_W': [None,
		             None,
		             None],
		'syn_specs': [{},
		              copy_dict(syn_specs, {'receptor_type': 1}),
		              copy_dict(syn_specs, {'receptor_type': 1})]},
	'input_decoder': {
		'state_extractor': {
			'N': 2,
			'filter_tau': None,
			'source_population': ['NEF', 'NEF'],
			'state_variable': ['V_m', 'spikes'],
			'state_specs': [
				copy_dict(rec_devices, {'model': 'multimeter',  # specifications of the devices/models that acqui state
				                        'record_from': ['V_m'],
				                        'record_n': None,
				                        'record_to': ['memory'],
				                        'interval': inp_resolution,
				                        'start': kernel_pars['transient_t'] - inp_resolution}),
				{'tau_m': filter_tau, 'interval': inp_resolution}]},
			'readout': [{
				'N': nSteps+1,
				'labels': readout_labels,
				'algorithm': ['ridge']},
				{'N': nSteps + 1,
				 'labels': readout_labels,
				 'algorithm': ['ridge']}],}
}


########################################################################################################################
# STATE and DECODING Parameters
# ============================================================================================================
decoding_pars = {
	'state_extractor': {
		'N': 4,
		'filter_tau': filter_tau,  # for convenience
		'source_population': ['E', ['E', 'I'], 'E', ['E', 'I']],
		'state_variable': ['V_m', 'spikes', 'spikes', 'V_m'], #spikes, spikes_post, V_m
		'state_specs':	[
			copy_dict(rec_devices, {'model': 'multimeter',  # specifications of the devices/models that acqui state
			                        'record_from': ['V_m'],
			                        'record_n': None,
			                        'record_to': ['memory'],
			                        'interval': inp_resolution,
			                        'start': kernel_pars['transient_t']-inp_resolution}),
			{'tau_m': filter_tau, 'interval': inp_resolution},
			{'tau_m': filter_tau, 'interval': inp_resolution},
			copy_dict(rec_devices, {'model': 'multimeter',  # specifications of the devices/models that acqui state
			                        'record_from': ['V_m'],
			                        'record_n': None,
			                        'record_to': ['memory'],
			                        'interval': inp_resolution,
			                        'start': kernel_pars['transient_t'] - inp_resolution})]},
	'readout': [{
		'N': nSteps+1,
		'labels': readout_labels,
		'algorithm': ['ridge']},
		{
		'N': nSteps + 1,
		'labels': readout_labels,
		'algorithm': ['ridge']},
		{
		'N': nSteps + 1,
		'labels': readout_labels,
		'algorithm': ['ridge']},
		{
		'N': nSteps + 1,
		'labels': readout_labels,
		'algorithm': ['ridge']}],
	'global_sampling_times': None
}
