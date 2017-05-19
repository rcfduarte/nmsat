__author__ = 'duarte'
from modules.parameters import ParameterSet, extract_nestvalid_dict
from modules.input_architect import EncodingLayer, StimulusSet, InputSignalSet
from modules.net_architect import Network
from modules.io import set_storage_locations
from modules.signals import iterate_obj_list, empty
from modules.visualization import set_global_rcParams, InputPlots, TopologyPlots, plot_input_example
from modules import auxiliary
from modules import analysis
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
	for n in list(iterate_obj_list(net.populations)):
		n.randomize_initial_states('V_m', randomization_function=np.random.uniform, low=0.0, high=15.)
	# TODO @barni: different neuron models have different resting/starting potential, check here and throw error?

	###################################################################################
	# Build and connect input
	# =================================================================================
	stim = StimulusSet(parameter_set, unique_set=True)
	stim.generate_datasets(parameter_set.stim_pars)

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
		stim.save(paths['inputs'])
		if debug:
			inputs.save(paths['inputs'])

	print "- Elapsed Time: {0}".format(str(time.time() - input_set_time))

	#######################################################################################
	# Encode Input
	# =====================================================================================
	enc_layer = EncodingLayer(parameter_set.encoding_pars, signal=inputs.full_set_signal, online=online)
	enc_layer.connect(parameter_set.encoding_pars, net)
	enc_layer.extract_connectivity(net, sub_set=True, progress=False)

	######################################################################################
	# Connect Network
	# ====================================================================================
	net.connect_populations(parameter_set.connection_pars)

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
	# Run Simulation (full sequence, but only transient and unique are there)
	# ======================================================================================================================
	auxiliary.process_input_sequence(parameter_set, net, enc_layer, stim,
													  inputs, set_name='full', record=True)

	# results = analysis.calculate_ranks(net)
	results = auxiliary.process_states(net, enc_layer, None, stim, data_sets=["unique"], accepted_idx=None,
									   plot=False, display=False, save=save, save_paths=paths)

	# ######################################################################################################################
	# Save data
	# ======================================================================================================================
	if save:
		with open(paths['results'] + 'Results_' + parameter_set.label, 'w') as f:
			pickle.dump(results, f)
		parameter_set.save(paths['parameters'] + 'Parameters_' + parameter_set.label)