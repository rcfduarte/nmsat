__author__ = 'duarte'
from modules.parameters import ParameterSet, ParameterSpace, extract_nestvalid_dict
from modules.input_architect import EncodingLayer, StimulusSet, InputSignalSet
from modules.net_architect import Network
from modules.io import set_storage_locations
from modules.signals import iterate_obj_list, empty
from modules.visualization import set_global_rcParams, InputPlots
from modules.analysis import compute_dimensionality, analyse_state_matrix, get_state_rank, readout_train, \
	readout_test, characterize_population_activity
from modules.auxiliary import iterate_input_sequence
import cPickle as pickle
import matplotlib.pyplot as pl
import numpy as np
import time
import itertools
import sys
import nest
# specific to this project..
sys.path.append('../')
from stimulus_generator import StimulusPattern


def run(parameter_set, plot=False, display=False, save=True, debug=False, online=True):
	"""
	Main stimulus processing task
	:param parameter_set: must be consistent with the computation
	:param plot: plot results - either show them or save to file
	:param display: show figures/reports
	:param save: save results
	:return results_dictionary:
	"""

	if not isinstance(parameter_set, ParameterSet):
		if isinstance(parameter_set, basestring) or isinstance(parameter_set, dict):
			parameter_set = ParameterSet(parameter_set)
		else:
			raise TypeError("parameter_set must be ParameterSet, string with full path to parameter file or dictionary")

	# ######################################################################################################################
	# Setup extra variables and parameters
	# ======================================================================================================================
	if plot:
		set_global_rcParams(parameter_set.kernel_pars['mpl_path'])
	paths = set_storage_locations(parameter_set, save)

	np.random.seed(parameter_set.kernel_pars['np_seed'])
	results = dict(rank={}, performance={}, dimensionality={})

	# ######################################################################################################################
	# Set kernel and simulation parameters
	# ======================================================================================================================
	print '\nRuning ParameterSet {0}'.format(parameter_set.label)
	nest.ResetKernel()
	nest.set_verbosity('M_WARNING')
	nest.SetKernelStatus(extract_nestvalid_dict(parameter_set.kernel_pars.as_dict(), param_type='kernel'))

	# ######################################################################################################################
	# Build network
	# ======================================================================================================================
	net = Network(parameter_set.net_pars)
	net.merge_subpopulations([net.populations[0], net.populations[1]], name='EI')

	# ######################################################################################################################
	# Randomize initial variable values
	# ======================================================================================================================
	for idx, n in enumerate(list(iterate_obj_list(net.populations))):
		if hasattr(parameter_set.net_pars, "randomize_neuron_pars"):
			randomize = parameter_set.net_pars.randomize_neuron_pars[idx]
			for k, v in randomize.items():
				n.randomize_initial_states(k, randomization_function=v[0], **v[1])

	###################################################################################
	# Build Stimulus Set
	# =================================================================================
	stim_set_startbuild = time.time()

	# Create or Load StimulusPattern
	stim_pattern = StimulusPattern(parameter_set.task_pars)
	stim_pattern.generate()

	input_sequence, output_sequence = stim_pattern.as_index()

	# Convert to StimulusSet object
	stim_set = StimulusSet(unique_set=False)
	stim_set.load_data(input_sequence, type='full_set_labels')
	stim_set.discard_from_set(parameter_set.stim_pars.transient_set_length)
	stim_set.divide_set(parameter_set.stim_pars.transient_set_length, parameter_set.stim_pars.train_set_length,
	                    parameter_set.stim_pars.test_set_length)

	# Specify target and convert to StimulusSet object
	target_set = StimulusSet(unique_set=False)
	target_set.load_data(output_sequence, type='full_set_labels')
	target_set.discard_from_set(parameter_set.stim_pars.transient_set_length)
	target_set.divide_set(parameter_set.stim_pars.transient_set_length, parameter_set.stim_pars.train_set_length,
	                      parameter_set.stim_pars.test_set_length)

	print "- Elapsed Time: {0}".format(str(time.time() - stim_set_startbuild))
	stim_set_buildtime = time.time() - stim_set_startbuild

	###################################################################################
	# Build Input Signal Set
	# =================================================================================
	input_set_time = time.time()
	parameter_set.input_pars.signal.N = len(np.unique(input_sequence))

	# Create InputSignalSet
	inputs = InputSignalSet(parameter_set, stim_set, online=online)
	if not empty(stim_set.transient_set_labels):
		inputs.generate_transient_set(stim_set)
		parameter_set.kernel_pars.transient_t = inputs.transient_stimulation_time
	if not online:
		inputs.generate_full_set(stim_set)
	# inputs.generate_unique_set(stim)
	inputs.generate_train_set(stim_set)
	inputs.generate_test_set(stim_set)

	# Plot example signal
	if plot and debug and not online:
		fig_inp = pl.figure()
		ax1 = fig_inp.add_subplot(211)
		ax2 = fig_inp.add_subplot(212)
		fig_inp.suptitle('Input Stimulus / Signal')
		inp_plot = InputPlots(stim_obj=stim_set, input_obj=inputs.test_set_signal, noise_obj=inputs.test_set_noise)
		inp_plot.plot_stimulus_matrix(set='test', ax=ax1, save=False, display=False)
		inp_plot.plot_input_signal(ax=ax2, save=paths['figures'] + paths['label'], display=display)
		inp_plot.plot_input_signal(save=paths['figures'] + paths['label'], display=display)
		inp_plot.plot_signal_and_noise(save=paths['figures'] + paths['label'], display=display)
	parameter_set.kernel_pars.sim_time = inputs.train_stimulation_time + inputs.test_stimulation_time

	if save:
		stim_pattern.save(paths['inputs'])
		stim_set.save(paths['inputs'])
		if debug:
			inputs.save(paths['inputs'])

	print "- Elapsed Time: {0}".format(str(time.time() - input_set_time))
	inputs_build_time = time.time() - input_set_time

	# ######################################################################################################################
	# Encode Input
	# ======================================================================================================================
	if not online:
		input_signal = inputs.full_set_signal
	else:
		input_signal = inputs.transient_set_signal
	enc_layer = EncodingLayer(parameter_set.encoding_pars, signal=input_signal, online=online)
	enc_layer.connect(parameter_set.encoding_pars, net)
	enc_layer.extract_connectivity(net, sub_set=True, progress=True)

	# ######################################################################################################################
	# Set-up Analysis
	# ======================================================================================================================
	net.connect_devices()
	net.connect_decoders(parameter_set.decoding_pars)

	# Attach decoders to input encoding populations
	if not empty(enc_layer.encoders) and hasattr(parameter_set.encoding_pars, "input_decoder") and \
			parameter_set.encoding_pars.input_decoder is not None:
		enc_layer.connect_decoders(parameter_set.encoding_pars.input_decoder)

	# ######################################################################################################################
	# Connect Network
	# ======================================================================================================================
	net.connect_populations(parameter_set.connection_pars)

	# ######################################################################################################################
	# Simulate (Transient Set)
	# ======================================================================================================================
	set_name = 'transient'
	if not empty(stim_set.transient_set_labels):
		epochs_transient = iterate_input_sequence(net, enc_layer, parameter_set, stim_set, inputs, set_name=set_name,
		                                          record=False, store_activity=False)
		for n_pop in list(itertools.chain(*[net.merged_populations, net.populations, enc_layer.encoders])):
			if n_pop.decoding_layer is not None:
				n_pop.decoding_layer.flush_states()

	# ######################################################################################################################
	# Simulate (Unique Set)
	# ======================================================================================================================
	if hasattr(stim_set, "unique_set"):
		set_name = 'unique'
		epochs_unique = iterate_input_sequence(net, enc_layer, parameter_set, stim_set, inputs, set_name=set_name,
		                                       record=True, store_activity=False)

		# compute ranks
		for n_pop in list(itertools.chain(*[net.merged_populations, net.populations, enc_layer.encoders])):
			if n_pop.decoding_layer is not None:
				dec_layer = n_pop.decoding_layer
				labels = getattr(stim_set, "{0}_set_labels".format(set_name))

				results['rank'].update({n_pop.name: {}})
				for idx_var, var in enumerate(dec_layer.state_variables):
					state_matrix = dec_layer.state_matrix[idx_var]

					if not empty(labels) and not empty(state_matrix):
						print "\nPopulation {0}, variable {1}, set {2}: {3}".format(n_pop.name, var, set_name,
						                                                            str(state_matrix.shape))
						results['rank'][n_pop.name].update({var + str(idx_var): get_state_rank(state_matrix)})

						if save:
							np.save(paths['activity'] + paths['label'] + '_population{0}_state{1}_{2}.npy'.format(
								n_pop.name,
								var, set_name), state_matrix)
						analyse_state_matrix(state_matrix, labels, label=n_pop.name + var + set_name,
						                     plot=plot, display=display, save=paths['figures'] + paths['label'])
				dec_layer.flush_states()

	# ######################################################################################################################
	# Simulate (Train Set)
	# ======================================================================================================================
	set_name = 'train'
	epochs_train = iterate_input_sequence(net, enc_layer, parameter_set, stim_set, inputs, set_name=set_name,
	                                      record=True,
	                                      store_activity=False)

	accept_idx = np.where(np.array(stim_pattern.Output['Accepted']) == 'A')[0]
	# print accept_idx

	# train readouts
	for n_pop in list(itertools.chain(*[net.merged_populations, net.populations, enc_layer.encoders])):
		if n_pop.decoding_layer is not None:
			dec_layer = n_pop.decoding_layer
			labels = getattr(target_set, "{0}_set_labels".format(set_name))
			target = np.array(getattr(target_set, "{0}_set".format(set_name)).todense())

			train_idx = []
			for idx in accept_idx:
				stop = parameter_set.stim_pars.train_set_length + parameter_set.stim_pars.transient_set_length
				if idx >= parameter_set.stim_pars.transient_set_length and idx < stop:
					train_idx.append(idx - parameter_set.stim_pars.transient_set_length)
			assert (len(train_idx) == target.shape[1]), "Incorrect train labels"
			# print train_idx

			for idx_var, var in enumerate(dec_layer.state_variables):
				readouts = dec_layer.readouts[idx_var]
				state_matrix = dec_layer.state_matrix[idx_var]
				if not empty(labels) and not empty(state_matrix):
					print "\nPopulation {0}, variable {1}, set {2}: {3}".format(n_pop.name, var, set_name,
					                                                            str(state_matrix.shape))
					for readout in readouts:
						readout_train(readout, state_matrix, target=target, index=None, accepted=train_idx,
						              display=display, plot=plot, save=paths['figures'] + paths['label'])
					if save:
						np.save(
							paths['activity'] + paths['label'] + '_population{0}_state{1}_{2}.npy'.format(n_pop.name,
							                                                                              var,
							                                                                              set_name),
							state_matrix)
					analyse_state_matrix(state_matrix, labels, label=n_pop.name + var + set_name,
					                     plot=plot, display=display, save=paths['figures'] + paths['label'])
			dec_layer.flush_states()
	# ######################################################################################################################
	# Simulate (Test Set)
	# ======================================================================================================================
	set_name = 'test'
	epochs_test = iterate_input_sequence(net, enc_layer, parameter_set, stim_set, inputs, set_name=set_name,
	                                     record=True,
	                                     store_activity=parameter_set.analysis_pars.store_activity)

	# characterize population activity
	if parameter_set.analysis_pars.store_activity:
		start_analysis = time.time()
		analysis_interval = [epochs_test['analysis_start'], nest.GetKernelStatus()['time'] - nest.GetKernelStatus()[
			'resolution']]
		results.update(characterize_population_activity(net, parameter_set, analysis_interval, epochs=None,
		                                                color_map='jet', plot=plot,
		                                                display=display, save=paths['figures'] + paths['label'],
		                                                **parameter_set.analysis_pars.population_state))
		print "\nElapsed time (state characterization): {0}".format(str(time.time() - start_analysis))

		if not empty(results['analog_activity']) and 'mean_I_ex' in results['analog_activity']['E'].keys():
			inh = np.array(results['analog_activity']['E']['mean_I_in'])
			exc = np.array(results['analog_activity']['E']['mean_I_ex'])
			ei_ratios = np.abs(np.abs(inh) - np.abs(exc))
			ei_ratios_corrected = np.abs(np.abs(inh - np.mean(inh)) - np.abs(exc - np.mean(exc)))
			print "EI amplitude difference: {0} +- {1}".format(str(np.mean(ei_ratios)), str(np.std(ei_ratios)))
			print "EI amplitude difference (amplitude corrected): {0} +- {1}".format(str(np.mean(ei_ratios_corrected)),
			                                                                         str(np.std(ei_ratios_corrected)))
			results['analog_activity']['E']['IE_ratio'] = np.mean(ei_ratios)
			results['analog_activity']['E']['IE_ratio_corrected'] = np.mean(ei_ratios_corrected)

	# test readouts and analyse state matrix
	for n_pop in list(itertools.chain(*[net.merged_populations, net.populations, enc_layer.encoders])):
		if n_pop.decoding_layer is not None:
			dec_layer = n_pop.decoding_layer
			results['performance'].update({n_pop.name: {}})
			results['dimensionality'].update({n_pop.name: {}})
			if parameter_set.analysis_pars.store_activity:
				if isinstance(parameter_set.analysis_pars.store_activity, int) and \
								parameter_set.analysis_pars.store_activity <= parameter_set.stim_pars.test_set_length:
					dec_layer.sampled_times = dec_layer.sampled_times[-parameter_set.analysis_pars.store_activity:]
				else:
					dec_layer.sampled_times = dec_layer.sampled_times[-parameter_set.stim_pars.test_set_length:]
				dec_layer.evaluate_decoding(n_neurons=10, display=display, save=paths['figures'] + paths['label'])

			labels = getattr(target_set, "{0}_set_labels".format(set_name))
			target = np.array(getattr(target_set, "{0}_set".format(set_name)).todense())

			test_idx = []
			start_t = parameter_set.stim_pars.transient_set_length + parameter_set.stim_pars.train_set_length
			for idx in accept_idx:
				if idx >= start_t:
					test_idx.append(idx - start_t)
			assert (len(test_idx) == target.shape[1]), "Incorrect test labels"

			for idx_var, var in enumerate(dec_layer.state_variables):
				results['performance'][n_pop.name].update({var + str(idx_var): {}})
				readouts = dec_layer.readouts[idx_var]
				state_matrix = dec_layer.state_matrix[idx_var]

				if not empty(labels) and not empty(state_matrix):
					print "\nPopulation {0}, variable {1}, set {2}: {3}".format(n_pop.name, var, set_name,
					                                                            str(state_matrix.shape))
					results['dimensionality'][n_pop.name].update({var + str(idx_var): compute_dimensionality(
						state_matrix)})
					for readout in readouts:
						results['performance'][n_pop.name][var + str(idx_var)].update(
							{readout.name: readout_test(readout,
							                            state_matrix, target=target, index=None, accepted=test_idx,
							                            display=display)})
						results['performance'][n_pop.name][var + str(idx_var)][readout.name].update({'norm_wOut':
							                                                                             readout.norm_wout})
					if save:
						np.save(
							paths['activity'] + paths['label'] + '_population{0}_state{1}_{2}.npy'.format(n_pop.name,
							                                                                              var,
							                                                                              set_name),
							state_matrix)
					analyse_state_matrix(state_matrix, labels, label=n_pop.name + var + set_name,
					                     plot=plot, display=display, save=paths['figures'] + paths['label'])
			dec_layer.flush_states()
	# ######################################################################################################################
	# Save data
	# ======================================================================================================================
	if save:
		with open(paths['results'] + 'Results_' + parameter_set.label, 'w') as f:
			pickle.dump(results, f)
		parameter_set.save(paths['parameters'] + 'Parameters_' + parameter_set.label)

