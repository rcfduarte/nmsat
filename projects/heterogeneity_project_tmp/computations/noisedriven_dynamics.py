__author__ = 'duarte'
import sys
from modules.parameters import ParameterSpace, ParameterSet, extract_nestvalid_dict
from modules.net_architect import Network
from modules.io import set_storage_locations
from modules.signals import iterate_obj_list, empty
from modules.input_architect import EncodingLayer
from modules.visualization import set_global_rcParams, TopologyPlots
from modules.analysis import characterize_population_activity
import time
import numpy as np
import nest
import cPickle as pickle
import matplotlib.pyplot as pl


def run(parameter_set, plot=False, display=False, save=True):
	"""
	Measure and characterize the population dynamics under Poissonian input
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
	results = dict()

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

	# ######################################################################################################################
	# Randomize initial variable values
	# ======================================================================================================================
	for idx, n in enumerate(list(iterate_obj_list(net.populations))):
		if hasattr(parameter_set.net_pars, "randomize_neuron_pars"):
			randomize = parameter_set.net_pars.randomize_neuron_pars[idx]
			for k, v in randomize.items():
				n.randomize_initial_states(k, randomization_function=v[0], **v[1])

	# ######################################################################################################################
	# Build and connect input
	# ======================================================================================================================
	enc_layer = EncodingLayer(parameter_set.encoding_pars)
	enc_layer.connect(parameter_set.encoding_pars, net)

	# ######################################################################################################################
	# Set-up Analysis
	# ======================================================================================================================
	net.connect_devices()

	# ######################################################################################################################
	# Connect Network
	# ======================================================================================================================
	net.connect_populations(parameter_set.connection_pars, progress=True)

	# ######################################################################################################################
	# Simulate
	# ======================================================================================================================
	if parameter_set.kernel_pars.transient_t:
		net.simulate(parameter_set.kernel_pars.transient_t)
		net.flush_records()

	net.simulate(parameter_set.kernel_pars.sim_time + parameter_set.kernel_pars.resolution)

	# ######################################################################################################################
	# Extract and store data
	# ======================================================================================================================
	analysis_interval = [parameter_set.kernel_pars.transient_t,
	                     parameter_set.kernel_pars.transient_t + parameter_set.kernel_pars.sim_time]
	net.extract_population_activity(t_start=analysis_interval[0], t_stop=analysis_interval[1])
	net.extract_network_activity()

	# ######################################################################################################################
	# Analyse / plot data
	# ======================================================================================================================
	parameter_set.analysis_pars.pop('label')
	start_analysis = time.time()
	results.update(characterize_population_activity(net, parameter_set, analysis_interval, epochs=None,
	                                                color_map='jet', plot=plot,
	                                                display=display, save=paths['figures'] + paths['label'],
	                                                color_subpop=True, analysis_pars=parameter_set.analysis_pars))
	print "\nElapsed time (state characterization): {0}".format(str(time.time() - start_analysis))

	if 'mean_I_ex' in results['analog_activity']['E'].keys():
		inh = np.array(results['analog_activity']['E']['mean_I_in'])
		exc = np.array(results['analog_activity']['E']['mean_I_ex'])
		ei_ratios = np.abs(np.abs(inh) - np.abs(exc))
		ei_ratios_corrected = np.abs(np.abs(inh - np.mean(inh)) - np.abs(exc - np.mean(exc)))
		print "EI amplitude difference: {0} +- {1}".format(str(np.mean(ei_ratios)), str(np.std(ei_ratios)))
		print "EI amplitude difference (amplitude corrected): {0} +- {1}".format(str(np.mean(ei_ratios_corrected)),
		                                                                         str(np.std(ei_ratios_corrected)))
		results['analog_activity']['E']['IE_ratio'] = np.mean(ei_ratios)
		results['analog_activity']['E']['IE_ratio_corrected'] = np.mean(ei_ratios_corrected)

	# ######################################################################################################################
	# Save data
	# ======================================================================================================================
	if save:
		with open(paths['results'] + 'Results_' + parameter_set.label, 'w') as f:
			pickle.dump(results, f)
		parameter_set.save(paths['parameters'] + 'Parameters_' + parameter_set.label)
