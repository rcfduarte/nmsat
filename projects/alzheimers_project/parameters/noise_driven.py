__author__ = 'duarte'
from preset import *
from defaults.paths import paths

"""
noise_driven parameter file
- standard network setup (BRN), with Poissonian input
- run with run_noise_driven function in computations
- debug with run_noise_driven script
"""

run = 'local'
data_label = 'AD_noisedriven_test'


def build_parameters():
	# ##################################################################################################################
	# System / Kernel Parameters
	# ##################################################################################################################
	system = dict(
		nodes=1,
		ppn=16,
		mem=32,
		walltime='01-00:00:00',
		queue='batch',
		transient_time=1000.,
		sim_time=2000.)

	kernel_pars = set_kernel_defaults(run_type=run, data_label=data_label, **system)

	# ##################################################################################################################
	# Neuron, Synapse and Network Parameters
	# ##################################################################################################################
	N = 1250
	delay = 1.
	epsilon = 0.1
	kE = int(epsilon * (N * 0.8))
	kI = int(epsilon * (N * 0.2))

	gamma = 6.

	wE = 32.29
	wI = -gamma * wE

	kEE = 100

	recurrent_synapses = dict(
		connected_populations=[('E', 'E'), ('E', 'I'), ('I', 'E'), ('I', 'I')],
		synapse_models=['static_synapse', 'static_synapse', 'static_synapse', 'static_synapse'],
		synapse_model_parameters=[{}, {}, {}, {}],
		pre_computedW=[None, None, None, None],
		weights=[wE, wI, wE, wI],
		delays=[delay, delay, delay, delay],
		conn_specs=[{'rule': 'fixed_indegree', 'indegree': kEE, 'autapses': False, 'multapses': False},
		            {'rule': 'fixed_indegree', 'indegree': kI, 'autapses': False, 'multapses': False},
		            {'rule': 'fixed_indegree', 'indegree': kE, 'autapses': False, 'multapses': False},
		            {'rule': 'fixed_indegree', 'indegree': kI, 'autapses': False, 'multapses': False}],
		syn_specs=[{}, {}, {}, {}]
	)
	neuron_pars, net_pars, connection_pars = set_network_defaults(N=N, kernel_pars=kernel_pars, **recurrent_synapses)

	# ######################################################################################################################
	# Encoding Parameters
	# ######################################################################################################################
	nu_x = 1.2
	k_x = 1000.

	encoding_pars = set_encoding_defaults(default_set=0)

	background_noise = dict(
		start=0., stop=sys.float_info.max, origin=0.,
		rate=nu_x * k_x, target_population_names=['E', 'I'],
		additional_parameters={
			'syn_specs': {},
			'models': 'static_synapse',
			'model_pars': {},
			'weight_dist': 90.,
			'delay_dist': delay})
	add_background_noise(encoding_pars, background_noise)

	# ##################################################################################################################
	# RETURN dictionary of Parameters dictionaries
	# ==================================================================================================================
	return dict([('kernel_pars', kernel_pars),
	             ('neuron_pars', neuron_pars),
	             ('net_pars', net_pars),
	             ('encoding_pars', encoding_pars),
	             ('connection_pars', connection_pars)])
