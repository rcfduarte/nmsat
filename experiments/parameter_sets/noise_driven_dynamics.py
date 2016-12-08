import numpy as np
import sys

sys.path.append('../parameter_sets/')
from defaults.paths import paths

run = 'local'
data_label = 'test0'
project_label = 'IPS'

#wd = paths[run]['parameters_file']
md = paths[run]['modules_path']
#sys.path.append(wd)
sys.path.append(md)
from modules.parameters import *
import itertools

__author__ = 'duarte'
"""
noise_driven_dynamics
- to run with function noise_driven_dynamics in Computation
or with NoiseDrivenDynamics.py (debugging)
"""

nodes = 1
ppn = 8
mem = 32
N_vp = nodes * ppn
# not really relevant for a local run, but need to be included
walltime = '00:20:00:00'
queue = 'singlenode'

np_seed = np.random.randint(1000000000)+1
np.random.seed(np_seed)
msd = np.random.randint(100000000000)

kernel_pars = {
	'resolution': 0.1,
	'sim_time': 1000.,
	'transient_t': 1000.,
	'data_prefix': data_label,
	'data_path': paths[run]['data_path'],
	'mpl_path': paths[run]['matplotlib_rc'],
	'overwrite_files': True,
	'print_time': True,
	'rng_seeds': range(msd + N_vp + 1, msd + 2 * N_vp + 1),
	'grng_seed': msd + N_vp,
	'total_num_virtual_procs': N_vp,
	'local_num_threads': ppn,
	'np_seed': np_seed,

	'system': {
		'local': True,
		'system_label': run,
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

rec_devices = {
	'start': kernel_pars['transient_t'],
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
neurons = {'E': {
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
			}}
#############################################################################################################
# NETWORK Parameters
#============================================================================================================
N = 1000
nE = 0.8 * N
nI = 0.2 * N


net_pars = {
	'n_populations': 2,
	'pop_names': ['E', 'I'],
	'n_neurons': [int(nE), int(nI)],
	'neuron_pars': [neurons['E'], neurons['I']],
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
	'record_analogs': [True, False],
	'analog_device_pars': [copy_dict(rec_devices,
	                                 {'model': 'multimeter',
	                                  'record_from': ['V_m', 'g_ex', 'g_in'],
	                                  'record_n': 1,
	                                  'record_to': ['memory'],
	                                  'label': 'E_population_multimeter'}),
	                       copy_dict(rec_devices,
	                                 {'model': 'multimeter',
	                                  'record_from': ['V_m', 'I_ex', 'I_in'],
	                                  'record_n': 1,
	                                  'record_to': ['memory'],
	                                  'label': 'I_population_multimeter'})
	                       ],
	'description': ''
}

#############################################################################################################
# SYNAPSE Parameters
#============================================================================================================
# connection delays
dE = 1.0
dI = 0.8

# Connection probabilities
pEE = 0.1
pEI = 0.2
pIE = 0.1
pII = 0.2

# connection weights
g = 22.5
wE = 1.2
wI = -g * wE

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
	'model_pars': [{}, {}, {}, {}],
	'pre_computedW': [None, None, None, None],
	'weight_dist': [{'distribution': 'normal_clipped', 'mu': wE, 'sigma': 0.5 * wE, 'low': 0.0001, 'high': 10. * wE},
	                {'distribution': 'normal_clipped', 'mu': wI, 'sigma': np.abs(0.5 * wI), 'low': 10. * wI,
	                 'high': 0.0001},
	                {'distribution': 'normal_clipped', 'mu': wE, 'sigma': 0.5 * wE, 'low': 0.0001, 'high': 10. * wE},
	                {'distribution': 'normal_clipped', 'mu': wI, 'sigma': np.abs(0.5 * wI), 'low': 10. * wI,
	                 'high': 0.0001}],
	'delay_dist': [{'distribution': 'normal_clipped', 'mu': dE, 'sigma': 0.5 * dE, 'low': 0.1, 'high': 10. * dE},
	               {'distribution': 'normal_clipped', 'mu': dI, 'sigma': 0.5 * dI, 'low': 0.1, 'high': 10. * dI},
	               {'distribution': 'normal_clipped', 'mu': dE, 'sigma': 0.5 * dE, 'low': 0.1, 'high': 10. * dE},
	               {'distribution': 'normal_clipped', 'mu': dI, 'sigma': 0.5 * dI, 'low': 0.1, 'high': 10. * dI}],
	'conn_specs': [{'rule': 'pairwise_bernoulli', 'p': pEE},
              {'rule': 'pairwise_bernoulli', 'p': pEI},
              {'rule': 'pairwise_bernoulli', 'p': pIE},
              {'rule': 'pairwise_bernoulli', 'p': pII}],
	'syn_specs': [{}, {}, {}, {}],
	'description': ''}

########################################################################################################################
# INPUT and ENCODING Parameters
#============================================================================================================
nu_x = 20.
k_x = pEE * nE
w_in = 1.


encoding_pars = {
	'generator': {
		'models': ['poisson_generator'],
		'labels': ['X_noise'],
		'N': 1,
		'model_pars': [{'origin': 0.0, 'start': 0.0, 'rate': 16000.0, 'stop': 1.7976931348623157e+308}],
		'topology_pars': [None],
		'topology': [False]},

	'connectivity': {
		'synapse_name': ['X_noise', 'X_noise'],
		'connections': [('E', 'X_noise'), ('I', 'X_noise')],
		'topology_dependent': [False, False],
		'conn_specs': [
			{'rule': 'all_to_all'},
			{'rule': 'all_to_all'}],
	    'syn_specs': [{}, {}],
		'models': ['static_synapse', 'static_synapse'],
		'model_pars': [{}, {}],
		'weight_dist': [{'distribution': 'normal_clipped', 'mu': w_in, 'sigma': 0.5*w_in, 'low': 0.0001,
		                'high': 10.*w_in},
		                {'distribution': 'normal_clipped', 'mu': w_in, 'sigma': 0.5 * w_in, 'low': 0.0001,
		                 'high': 10. * w_in}],
		'delay_dist': [0.1, 0.1],
		'preset_W': [None, None]},
	'encoder': {
		'topology_dict': [],
		'record_analogs': [],
		'models': [],
		'n_neurons': [],
		'labels': [],
		'label': "global",
		'N': 0,
		'model_pars': [],
		'neuron_pars': [],
		'record_spikes': [],
		'spike_device_pars': [],
		'analog_device_pars': [],
		'topology': [],
	},
	'description': '',
}
################################################ REPORT ################################################################
report_pars = { # just taken directly from saved Parameters...
      "report_filename": "test0.tex",
      "report_path": "/home/neuro/Desktop/Data/",
      "table_fields": {
         "{{ net_plasticity }}": "None",
         "{{ net_synapse_models }}": "Fast, exponential, conductance-based synapses Non-adapting",
         "{{ population_parameters }}": "$N^{E}$ & 8000 & E Population Size\tabularnewline \hline $N^{I}$ & 2000 & I Population Size\tabularnewline \hline ",
         "{{ net_neuron_models }}": "Adaptive Leaky integrate-and-fire, fixed firing threshold, fixed absolute refractory time",
         "{{ net_connectivity_description }}": "Sparse, Random with density 0.1 (all connections)",
         "{{ connectivity_parameters }}": "\epsilon & $0.1$ & Connection Probability (for all synapses)\tabularnewline \hline $w^{EE}$ & {'mu': 1.2, 'high': 12.0, 'distribution': 'normal_clipped', 'sigma': 0.6, 'low': 0.0001} & Connection strength\tabularnewline \hline $w^{IE}$ & {'mu': -16.2, 'high': 0.0001, 'distribution': 'normal_clipped', 'sigma': 8.0999999999999996, 'low': -162.0} & Connection strength\tabularnewline \hline $w^{EI}$ & {'mu': 1.2, 'high': 12.0, 'distribution': 'normal_clipped', 'sigma': 0.6, 'low': 0.0001} & Connection strength\tabularnewline \hline $w^{II}$ & {'mu': -16.2, 'high': 0.0001, 'distribution': 'normal_clipped', 'sigma': 8.0999999999999996, 'low': -162.0} & Connection strength\tabularnewline \hline ",
         "{{ measurements_description }}": "",
         "{{ net_populations_description }}": "2 Main Populations: ['E', 'I']",
         "{{ measurements }}": "",
         "{{ input_general_description }}": "Direct encoding (DC injection to network)",
         "{{ input_description }}": "\verb+step_current_generator+ & ['E', 'I'] & Variable current trace ($u(t)$)\tabularnewline \hline ",
         "{{ neuron_parameters }}": "Name & E & Neuron name\tabularnewline  \hline $C_{m}$ & $250.0\pF$ & Membrane Capacitance\tabularnewline  \hline $V_{\mathrm{rest}}$ & $-70.0\mV$ & Resting Membrane Potential\tabularnewline  \hline $V_{\mathrm{th}}$ & $-50.0\mV$ & Fixed Firing Threshold\tabularnewline  \hline $V_{\mathrm{reset}}$ & $-60.0\mV$ & Reset Potential\tabularnewline  \hline $g_{\mathrm{leak}}$ & $16.7\nS$ & Leak Conductance\tabularnewline  \hline $\Delta_{T}$ & $2.0None$ & Adaptation sharpness parameter\tabularnewline  \hline $g_{E}$ & $1.0\nS$ & Excitatory conductance\tabularnewline  \hline $g_{I}$ & $1.0\nS$ & Inhibitory conductance\tabularnewline  \hline $E_{E}$ & $0.0\mV$ & Excitatory reversal potential\tabularnewline  \hline $E_{I}$ & $-75.0\mV$ & Inhibitory reversal potential\tabularnewline  \hline $a$ & $4.0\nS$ & Sub-threshold intrinsic adaptation parameter\tabularnewline  \hline $b$ & $80.5\mV$ & Spike-triggered intrinsic adaptation parameter\tabularnewline  \hline $t_{\mathrm{ref}}$ & $2.0\ms$ & Absolute Refractory time\tabularnewline  \hline $\tau_{E}$ & $2.0\ms$ & Excitatory synaptic time constant\tabularnewline  \hline $\tau_{I}$ & $6.0\ms$ & Inhbitory synaptic time constant\tabularnewline  \hline $\tau_{w}$ & $144.0\ms$ & Adaptation time constant\tabularnewline  \hline ",
         "{{ synapse_parameters }}": "None & None & None\tabularnewline \hline ",
         "{{ connectivity_name_src_tgt_pattern }}": "EE & E & E & Random, \tabularnewline \hline IE & I & E & Random, \tabularnewline \hline EI & E & I & Random, \tabularnewline \hline II & I & I & Random, \tabularnewline \hline ",
         "{{ net_populations_topology }}": "None",
         "label": "global",
         "{{ population_names_elements_sizes }}": "E & \verb+aeif_cond_exp+ & 8000\tabularnewline \hline I & \verb+aeif_cond_exp+ & 2000\tabularnewline \hline ",
         "{{ plasticity_parameters }}": "None & None & None\tabularnewline \hline ",
         "{{ input_parameters }}": "$N_{\mathrm{stim}}$ & $1$ & Number of input stimuli \tabularnewline  \hline $\gamma_{\mathrm{in}}$ & $1$ & Fraction of receiving neurons \tabularnewline \hline ",
      },
      "report_templates_path": "/home/neuro/Desktop/CODE/NetworkSimulationTestbed/Defaults/ReportTemplates/IPS/",
      "label": "global",
   }