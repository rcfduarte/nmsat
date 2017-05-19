__author__ = 'duarte'
from modules.parameters import ParameterSet, ParameterSpace, extract_nestvalid_dict
from modules.input_architect import EncodingLayer, StimulusSet, InputSignalSet
from modules.net_architect import Network
from modules.io import set_storage_locations, Standardh5File
from modules.signals import iterate_obj_list, empty
from modules.visualization import set_global_rcParams, InputPlots, TopologyPlots, plot_input_example
from modules import auxiliary
import cPickle as pickle
import numpy as np
import nest
import time
import matplotlib.pyplot as pl
import itertools
import sys

def run(parameter_set, plot=False, display=False, save=True, debug=False, online=True):
	"""
	Analyse network dynamics when driven by Poisson input
	:param parameter_set: must be consistent with the computation, i.e. input must be poisson...
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
	results = dict()

	# ######################################################################################################################
	# Set kernel and simulation parameters
	# ======================================================================================================================
	print('\nRuning ParameterSet {0}'.format(parameter_set.label))
	nest.ResetKernel()
	nest.set_verbosity('M_WARNING')
	nest.SetKernelStatus(extract_nestvalid_dict(parameter_set.kernel_pars.as_dict(), param_type='kernel'))

	###################################################################################
	# Build network
	# =================================================================================
	net = Network(parameter_set.net_pars)
	net.merge_subpopulations([net.populations[0], net.populations[1]], name='EI')

	###################################################################################
	# Randomize initial variable values
	# =================================================================================
	for idx, n in enumerate(list(iterate_obj_list(net.populations))):
		if hasattr(parameter_set.net_pars, "randomize_neuron_pars"):
			randomize = parameter_set.net_pars.randomize_neuron_pars[idx]
			for k, v in randomize.items():
				n.randomize_initial_states(k, randomization_function=v[0], **v[1])
	# TODO @barni: different neuron models have different resting/starting potential, check here and throw error?

	###################################################################################
	# Build and connect input
	# =================================================================================
	# Create StimulusSet
	if hasattr(parameter_set, "task_pars"):
		sys.path.append('../')
		from stimulus_generator import StimulusPattern

		# Create or Load StimulusPattern
		stim_pattern = StimulusPattern(parameter_set.task_pars)
		stim_pattern.generate()

		input_sequence, output_sequence = stim_pattern.as_index()

		# Convert to StimulusSet object
		stim = StimulusSet(unique_set=False)
		stim.generate_datasets(parameter_set.stim_pars, external_sequence=input_sequence)
		exit(0)
	else:
		stim = StimulusSet(parameter_set, unique_set=False)
		stim.generate_datasets(parameter_set.stim_pars)

		# Prepare training targets

		# create 10 random dichotomies on the spike templates, in each of which half of the templates are 1,
		# other half is 0
		n_dichotomies = 10

		# divide half the templates into class 0, other half to class 1
		target_template = np.array(([0] * (stim.dims / 2)) + ([1] * (stim.dims / 2)))
		assert target_template.shape[0] == stim.dims, "Target template dimension inconsistent with #input stimuli"

		# n_templates X n_dichotomies
		dichotomies = np.array([np.random.permutation(target_template) for _ in range(n_dichotomies)])

		train_target_matrix = []
		test_target_matrix = []

		for label in stim.train_set_labels:
			stim_id = label[0] if isinstance(label, list) else label
			assert isinstance(stim_id, int), "Wrong label type (should be int)"
			# append column to target matrix (corresponds to 1 stimuli)
			train_target_matrix.append(list(dichotomies[:, stim_id]))

		for label in stim.test_set_labels:
			stim_id = label[0] if isinstance(label, list) else label
			assert isinstance(stim_id, int), "Wrong label type (should be int)"
			test_target_matrix.append(list(dichotomies[:, stim_id]))

		# empty matrix for transient set
		transient_target_matrix = np.zeros((n_dichotomies, len(stim.transient_set_labels)))
		# n_dichotomies X n_train_set_size  (after transposing)
		train_target_matrix = np.array(train_target_matrix).transpose()
		# n_dichotomies X n_test_set_size  (after transposing)
		test_target_matrix = np.array(test_target_matrix).transpose()

		# build a combined target matrix: n_dichotomies X n_full_set_size
		target_matrix = np.concatenate([transient_target_matrix, train_target_matrix, test_target_matrix], 1)

		# do some checks: for the training set at least, go through each column / stimulus
		# in the training set and make sure they are consistent with the dichotomy partitioning
		for idx, label in enumerate(stim.train_set_labels):
			stim_id = label[0] if isinstance(label, list) else label
			dichotomy = dichotomies[:, stim_id].tolist()
			set_elem = train_target_matrix[:, idx].tolist()
			assert dichotomy == set_elem
			
	########################################################################################################################
	# Build Input Signal Sets
	# ======================================================================================================================
	input_set_time = time.time()

	inputs = InputSignalSet(parameter_set, stim, online=online)
	inputs.generate_datasets(stim)

	print "- Elapsed Time: {0}".format(str(time.time() - input_set_time))

	parameter_set.kernel_pars.sim_time = inputs.train_stimulation_time + inputs.test_stimulation_time

	# Plot example signal
	if plot and debug and not online:
		plot_input_example(stim, inputs, set_name='test', display=display, save=paths['figures'] + paths['label'])
	if save:
		if hasattr(parameter_set, "task_pars"):
			stim_pattern.save(paths['inputs'])
		stim.save(paths['inputs'])
		if debug:
			inputs.save(paths['inputs'])

	# ######################################################################################################################
	# Encode Input
	# ======================================================================================================================
	enc_layer = EncodingLayer(parameter_set.encoding_pars, signal=inputs.full_set_signal, online=online)
	enc_layer.connect(parameter_set.encoding_pars, net)
	enc_layer.extract_connectivity(net, sub_set=True, progress=False)

	######################################################################################
	# Connect Network
	# ====================================================================================
	net.connect_populations(parameter_set.connection_pars)

	if plot and debug and False:
		fig_W = pl.figure()
		topology = TopologyPlots(parameter_set.connection_pars, net)
		topology.print_network(depth=3)
		ax1 = pl.subplot2grid((6, 6), (0, 0), rowspan=4, colspan=4)
		ax2 = pl.subplot2grid((6, 6), (5, 0), rowspan=1, colspan=4)
		ax3 = pl.subplot2grid((6, 6), (0, 5), rowspan=4, colspan=1)
		ax4 = pl.subplot2grid((6, 6), (5, 5), rowspan=1, colspan=1)
		topology.plot_connectivity(parameter_set.connection_pars.synapse_types,
								   ax=[ax1, ax3, ax2, ax4], display=display, save=paths['figures'] + paths['label'])

	#######################################################################################
	# Set-up Analysis
	# =====================================================================================
	net.connect_devices()
	auxiliary.set_decoder_times(enc_layer, parameter_set)  # iff using the fast sampling method!
	net.connect_decoders(parameter_set.decoding_pars)

	# Attach decoders to input encoding populations
	if not empty(enc_layer.encoders) and hasattr(parameter_set.encoding_pars, "input_decoder") and \
					parameter_set.encoding_pars.input_decoder is not None:
		enc_layer.connect_decoders(parameter_set.encoding_pars.input_decoder)

	# ######################################################################################################################
	# Run Simulation (full sequence)
	# ======================================================================================================================
	epochs, timing = auxiliary.process_input_sequence(parameter_set, net, enc_layer, stim,
													  inputs, set_name='full', record=True)

	# Slow state sampling
	# epochs, timing = iterate_input_sequence(net, enc_layer, parameter_set, stim_set, inputs, set_name='full', record=True,
	#                        store_activity=False)

	# ######################################################################################################################
	# Process data
	# ======================================================================================================================
	if hasattr(parameter_set, "task_pars"):
		accept_idx = np.where(np.array(stim_pattern.Output['Accepted']) == 'A')[0]
	else:
		accept_idx = None

	# target_matrix = np.array(target_set.full_set.todense())
	results = auxiliary.process_states(net, enc_layer, target_matrix, stim, data_sets=None,
									   accepted_idx=accept_idx, plot=plot, evaluation_method='L_RSc',
									   display=display, save=save, save_paths=paths)
	results.update({'timing_info': timing, 'epochs': epochs})

	# ######################################################################################################################
	# Save data
	# ======================================================================================================================
	if save:
		with open(paths['results'] + 'Results_' + parameter_set.label, 'w') as f:
			pickle.dump(results, f)
		parameter_set.save(paths['parameters'] + 'Parameters_' + parameter_set.label)