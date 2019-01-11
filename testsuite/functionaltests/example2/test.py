from modules.parameters import ParameterSet, ParameterSpace, extract_nestvalid_dict
from modules.input_architect import EncodingLayer, InputNoise
from modules.net_architect import Network
from modules.io import set_storage_locations
from modules.signals import iterate_obj_list
from modules.analysis import characterize_population_activity, single_neuron_responses
from modules.visualization import set_global_rcParams, InputPlots

import time
import cPickle as pickle
import numpy as np

import nest

def __initialize_test_data(params_file_):
	plot = False
	display = True
	save = True

	# ##################################################################################################################
	# Extract parameters from file and build global ParameterSet
	# ==================================================================================================================
	parameter_set = ParameterSpace(params_file_)[0]
	parameter_set = parameter_set.clean(termination='pars')

	if not isinstance(parameter_set, ParameterSet):
		if isinstance(parameter_set, basestring) or isinstance(parameter_set, dict):
			parameter_set = ParameterSet(parameter_set)
		else:
			raise TypeError("parameter_set must be ParameterSet, string with full path to parameter file or dictionary")

	# ##################################################################################################################
	# Setup extra variables and parameters
	# ==================================================================================================================
	if plot:
		set_global_rcParams(parameter_set.kernel_pars['mpl_path'])
	paths = set_storage_locations(parameter_set, save)

	np.random.seed(parameter_set.kernel_pars['np_seed'])

	# ##################################################################################################################
	# Set kernel and simulation parameters
	# ==================================================================================================================
	print '\nRuning ParameterSet {0}'.format(parameter_set.label)
	nest.ResetKernel()
	nest.set_verbosity('M_WARNING')
	nest.SetKernelStatus(extract_nestvalid_dict(parameter_set.kernel_pars.as_dict(), param_type='kernel'))

	# ##################################################################################################################
	# Build network
	# ==================================================================================================================
	net = Network(parameter_set.net_pars)

	# ##################################################################################################################
	# Randomize initial variable values
	# ==================================================================================================================
	for idx, n in enumerate(list(iterate_obj_list(net.populations))):
		if hasattr(parameter_set.net_pars, "randomize_neuron_pars"):
			randomize = parameter_set.net_pars.randomize_neuron_pars[idx]
			for k, v in randomize.items():
				n.randomize_initial_states(k, randomization_function=v[0], **v[1])

	####################################################################################################################
	# Build Input Signal Sets
	# ==================================================================================================================
	assert hasattr(parameter_set, "input_pars")

	total_stimulation_time = parameter_set.kernel_pars.sim_time + parameter_set.kernel_pars.transient_t

	# Current input (need to build 2 separate noise signals for the 2 input channels)
	# Generate input for channel 1
	input_noise_ch1 = InputNoise(parameter_set.input_pars.noise, rng=np.random, stop_time=total_stimulation_time)
	input_noise_ch1.generate()
	input_noise_ch1.re_seed(parameter_set.kernel_pars.np_seed)

	# Generate input for channel 2
	input_noise_ch2 = InputNoise(parameter_set.input_pars.noise, rng=np.random, stop_time=total_stimulation_time)
	input_noise_ch2.generate()
	input_noise_ch2.re_seed(parameter_set.kernel_pars.np_seed)

	if plot:
		inp_plot = InputPlots(stim_obj=None, input_obj=None, noise_obj=input_noise_ch1)
		inp_plot.plot_noise_component(display=display, save=paths['figures'] + "/InputNoise_CH1")

		inp_plot = InputPlots(stim_obj=None, input_obj=None, noise_obj=input_noise_ch2)
		inp_plot.plot_noise_component(display=display, save=paths['figures'] + "/InputNoise_CH2")

	# ##################################################################################################################
	# Build and connect input
	# ==================================================================================================================
	enc_layer_ch1 = EncodingLayer(parameter_set.encoding_ch1_pars, signal=input_noise_ch1)
	enc_layer_ch1.connect(parameter_set.encoding_ch1_pars, net)

	enc_layer_ch2 = EncodingLayer(parameter_set.encoding_ch2_pars, signal=input_noise_ch2)
	enc_layer_ch2.connect(parameter_set.encoding_ch2_pars, net)

	# ##################################################################################################################
	# Connect Devices
	# ==================================================================================================================
	net.connect_devices()

	# ##################################################################################################################
	# Simulate
	# ==================================================================================================================
	if parameter_set.kernel_pars.transient_t:
		net.simulate(parameter_set.kernel_pars.transient_t)
		net.flush_records()

	net.simulate(parameter_set.kernel_pars.sim_time + nest.GetKernelStatus()['resolution'])

	# ##################################################################################################################
	# Extract and store data
	# ==================================================================================================================
	net.extract_population_activity(t_start=parameter_set.kernel_pars.transient_t,
									t_stop=parameter_set.kernel_pars.sim_time + parameter_set.kernel_pars.transient_t)
	net.extract_network_activity()

	# ##################################################################################################################
	# Analyse / plot data
	# ==================================================================================================================
	analysis_interval = [parameter_set.kernel_pars.transient_t,
						 parameter_set.kernel_pars.transient_t + parameter_set.kernel_pars.sim_time]

	results = dict()

	for idd, nam in enumerate(net.population_names):
		results.update({nam: {}})
		results[nam] = single_neuron_responses(net.populations[idd],
											   parameter_set, pop_idx=idd,
											   start=analysis_interval[0],
											   stop=analysis_interval[1],
											   plot=plot, display=display,
											   save=paths['figures'] + paths['label'])
		if results[nam]['rate']:
			print('Output Rate [{0}] = {1} spikes/s'.format(str(nam), str(results[nam]['rate'])))

	# ######################################################################################################################
	# Save data
	# ======================================================================================================================
	data = dict()

	data['connections_from'] = {pop.name: nest.GetConnections(source=pop.gids) for (idx, pop) in
								enumerate(net.populations)}
	data['connections_to'] = {pop.name: nest.GetConnections(target=pop.gids) for (idx, pop) in
							  enumerate(net.populations)}
	data['results'] = results

	data['input'] = {
		'channel1': input_noise_ch1.noise_signal.analog_signals[.0].signal,
		'channel2': input_noise_ch2.noise_signal.analog_signals[.0].signal
	}

	data['network'] = { 'populations': {net.populations[0].name: net.populations[0]} }

	return data


def __test_connections(ref_data_, sim_data_):
	"""

	:param ref_data_:
	:param sim_data_:
	:return:
	"""
	# NEST connections
	for pop_name in ref_data_['connections_from'].keys():
		ref_conn = ref_data_['connections_from'][pop_name]
		sim_conn = sim_data_['connections_from'][pop_name]

		assert len(ref_conn) == len(sim_conn)
		for idx in range(len(ref_conn)):
			assert ref_conn[idx] == sim_conn[idx]

	for pop_name in ref_data_['connections_to'].keys():
		ref_conn = ref_data_['connections_to'][pop_name]
		sim_conn = sim_data_['connections_to'][pop_name]

		assert len(ref_conn) == len(sim_conn)
		for idx in range(len(ref_conn)):
			assert ref_conn[idx] == sim_conn[idx]

	print('Passed connection test')


def __test_spikes(ref_data_, sim_data_):
	"""

	:param ref_data_:
	:param sim_data_:
	:return:
	"""
	# go through each population and compare member variables
	for pop_name, ref_pop in ref_data_['network']['populations'].iteritems():
		sim_pop = sim_data_['network']['populations'][pop_name]

		assert sorted(ref_pop.gids) == sorted(sim_pop.gids)

		# spiking activity
		ref_spike_list = ref_pop.spiking_activity
		sim_spike_list = sim_pop.spiking_activity

		# compare some static values
		assert np.array_equal(ref_spike_list.id_list, sim_spike_list.id_list)
		assert ref_spike_list.t_start == sim_spike_list.t_start
		assert ref_spike_list.t_stop == sim_spike_list.t_stop

		# compare spiketrains
		for id_, ref_spiketrain in ref_spike_list.spiketrains.iteritems():
			sim_spiketrain = sim_spike_list.spiketrains[id_]

			assert np.array_equal(ref_spiketrain.spike_times, sim_spiketrain.spike_times)

	print('Passed spike time test')


def __test_results(ref_data_, sim_data_):
	"""

	:param ref_data_:
	:param sim_data_:
	:return:
	"""
	ref_results = ref_data_['results']
	sim_results = sim_data_['results']

	# check some but not all results, should be enough though
	pop = 'AeifCondExp'
	assert np.allclose(ref_results[pop]['cv_isi'], sim_results[pop]['cv_isi'])
	assert np.allclose(ref_results[pop]['ff'], sim_results[pop]['ff'])
	assert np.array_equal(ref_results[pop]['isi'], sim_results[pop]['isi'])
	assert np.array_equal(ref_results[pop]['rate'], sim_results[pop]['rate'])
	assert np.array_equal(ref_results[pop]['vm'], sim_results[pop]['vm'])

	print('Passed results test')


if __name__ == "__main__":
	nest_version = 'v2.16'

	test_name = "example2"
	print("Processing test data `{0}`".format(test_name))

	params_file = './reference_{}.params'.format(nest_version)
	ref_data = pickle.load(open('./reference_{}.data'.format(nest_version), 'r'))
	sim_data = __initialize_test_data(params_file)

	print("\n############################################################################")
	print("Results for test `{0}`\n".format(test_name))

	__test_connections(ref_data, sim_data)

	__test_spikes(ref_data, sim_data)

	__test_results(ref_data, sim_data)

	print("----------------------------------------------------------------------------\n")


