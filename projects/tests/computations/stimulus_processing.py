__author__ = 'duarte'
from modules.parameters import ParameterSet, extract_nestvalid_dict
from modules.input_architect import EncodingLayer, StimulusSet, InputSignalSet
from modules.net_architect import Network
from modules.io import set_storage_locations
from modules.signals import iterate_obj_list, empty
from modules.visualization import set_global_rcParams, plot_input_example
from modules.auxiliary import process_input_sequence, process_states, set_decoder_times
import cPickle as pickle
import numpy as np
import itertools
import time
import sys
import nest


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
			raise TypeError("parameter_set must be ParameterSet, string with full path to parameter file or "
			                "dictionary")

	# ######################################################################################################################
	# Setup extra variables and parameters
	# ======================================================================================================================
	if plot:
		set_global_rcParams(parameter_set.kernel_pars['mpl_path'])
	paths = set_storage_locations(parameter_set, save)

	np.random.seed(parameter_set.kernel_pars['np_seed'])

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
	net.merge_subpopulations([net.populations[0], net.populations[1]], name='EI')  # merge for EI case

	# ######################################################################################################################
	# Randomize initial variable values
	# ======================================================================================================================
	for idx, n in enumerate(list(iterate_obj_list(net.populations))):
		if hasattr(parameter_set.net_pars, "randomize_neuron_pars"):
			randomize = parameter_set.net_pars.randomize_neuron_pars[idx]
			for k, v in randomize.items():
				n.randomize_initial_states(k, randomization_function=v[0], **v[1])

	########################################################################################################################
	# Build Stimulus/Target Sets
	# ======================================================================================================================
	stim_set_startbuild = time.time()

	if hasattr(parameter_set, "task_pars"):
		sys.path.append('../')
		from stimulus_generator import StimulusPattern

		# Create or Load StimulusPattern
		stim_pattern = StimulusPattern(parameter_set.task_pars)
		stim_pattern.generate()

		input_sequence, output_sequence = stim_pattern.as_index()

		# Convert to StimulusSet object
		stim_set = StimulusSet(unique_set=False)
		stim_set.generate_datasets(parameter_set.stim_pars, external_sequence=input_sequence)

		# Specify target and convert to StimulusSet object
		target_set = StimulusSet(unique_set=False)
		target_set.generate_datasets(parameter_set.stim_pars, external_sequence=output_sequence)
	else:
		stim_set = StimulusSet(parameter_set, unique_set=False)
		stim_set.generate_datasets(parameter_set.stim_pars)

		target_set = StimulusSet(parameter_set, unique_set=False)  # for identity task.
		output_sequence = list(itertools.chain(*stim_set.full_set_labels))
		target_set.generate_datasets(parameter_set.stim_pars, external_sequence=output_sequence)

	# correct N for small sequences
	parameter_set.input_pars.signal.N = len(np.unique(stim_set.full_set_labels))

	stim_set_buildtime = time.time() - stim_set_startbuild
	print "- Elapsed Time: {0}".format(str(stim_set_buildtime))

	########################################################################################################################
	# Build Input Signal Sets
	# ======================================================================================================================
	input_set_time = time.time()

	inputs = InputSignalSet(parameter_set, stim_set, online=online)
	inputs.generate_datasets(stim_set)

	input_set_buildtime = time.time() - input_set_time
	print "- Elapsed Time: {0}".format(str(input_set_buildtime))

	parameter_set.kernel_pars.sim_time = inputs.train_stimulation_time + inputs.test_stimulation_time

	# Plot example signal
	if plot and debug and not online:
		plot_input_example(stim_set, inputs, set_name='test', display=display, save=paths['figures'] + paths[
			'label'])
	if save:
		if hasattr(parameter_set, "task_pars"):
			stim_pattern.save(paths['inputs'])
		stim_set.save(paths['inputs'])
		if debug:
			inputs.save(paths['inputs'])

	# ######################################################################################################################
	# Encode Input
	# ======================================================================================================================
	enc_layer = EncodingLayer(parameter_set.encoding_pars, signal=inputs.full_set_signal, online=online)
	enc_layer.connect(parameter_set.encoding_pars, net)
	enc_layer.extract_connectivity(net, sub_set=True, progress=False)

	# ######################################################################################################################
	# Connect Network
	# ======================================================================================================================
	net.connect_populations(parameter_set.connection_pars)

	# ######################################################################################################################
	# Set-up Analysis
	# ======================================================================================================================
	net.connect_devices()
	set_decoder_times(enc_layer, parameter_set)  # iff using the fast sampling method!
	net.connect_decoders(parameter_set.decoding_pars)

	# Attach decoders to input encoding populations
	if not empty(enc_layer.encoders) and hasattr(parameter_set.encoding_pars, "input_decoder") and \
					parameter_set.encoding_pars.input_decoder is not None:
		enc_layer.connect_decoders(parameter_set.encoding_pars.input_decoder)

	# ######################################################################################################################
	# Run Simulation (full sequence)
	# ======================================================================================================================
	epochs, timing = process_input_sequence(parameter_set, net, enc_layer, stim_set, inputs, set_name='full',
	                                        record=True)

	# Slow state sampling
	# epochs, timing = iterate_input_sequence(net, enc_layer, parameter_set, stim_set, inputs, set_name='full',
	# record=True,
	#                        store_activity=False)

	# ######################################################################################################################
	# Process data
	# ======================================================================================================================
	if hasattr(parameter_set, "task_pars"):
		accept_idx = np.where(np.array(stim_pattern.Output['Accepted']) == 'A')[0]
	else:
		accept_idx = None

	target_matrix = np.array(target_set.full_set.todense())
	results = process_states(net, enc_layer, target_matrix, stim_set, data_sets=None, accepted_idx=accept_idx,
	                         plot=plot,
	                         display=display, save=save, save_paths=paths)
	results.update({'timing_info': timing, 'epochs': epochs})

	# ######################################################################################################################
	# Save data
	# ======================================================================================================================
	if save:
		with open(paths['results'] + 'Results_' + parameter_set.label, 'w') as f:
			pickle.dump(results, f)
		parameter_set.save(paths['parameters'] + 'Parameters_' + parameter_set.label)

