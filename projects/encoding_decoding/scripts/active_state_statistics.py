__author__ = 'duarte'
from modules.parameters import ParameterSet, ParameterSpace, extract_nestvalid_dict
from modules.input_architect import EncodingLayer, StimulusSet, InputSignalSet
from modules.net_architect import Network
from modules.io import set_storage_locations
from modules.signals import iterate_obj_list, empty
from modules.visualization import set_global_rcParams, plot_input_example
from modules.auxiliary import process_input_sequence, process_states, set_decoder_times, iterate_input_sequence
from modules.analysis import characterize_population_activity
import cPickle as pickle
import numpy as np
import itertools
import time
import sys
import nest

# ######################################################################################################################
# Experiment options
# ======================================================================================================================
plot = True
display = True
save = True
debug = False
online = True

# ######################################################################################################################
# Extract parameters from file and build global ParameterSet
# ======================================================================================================================
# params_file = '../parameters/dc_stimulus_input.py'
params_file = '../../encoding_decoding/parameters/dcinput_activestate.py'
# params_file = '../../encoding_decoding/parameters/spike_pattern_input.py'

parameter_set = ParameterSpace(params_file)[0]
parameter_set = parameter_set.clean(termination='pars')

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

stim_set_buildtime = time.time()-stim_set_startbuild
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
if hasattr(parameter_set, "decoding_pars"):
	set_decoder_times(enc_layer, parameter_set) # iff using the fast sampling method!
	net.connect_decoders(parameter_set.decoding_pars)

# Attach decoders to input encoding populations
if not empty(enc_layer.encoders) and hasattr(parameter_set.encoding_pars, "input_decoder") and \
				parameter_set.encoding_pars.input_decoder is not None:
	enc_layer.connect_decoders(parameter_set.encoding_pars.input_decoder)

# ######################################################################################################################
# Run Simulation (full sequence)
# ======================================================================================================================
# epochs, timing = process_input_sequence(parameter_set, net, enc_layer, stim_set, inputs, set_name='full', record=True)

# Slow state sampling
epochs, timing = iterate_input_sequence(net, enc_layer, parameter_set, stim_set, inputs, set_name='full', record=True,
                       store_activity=False)

# ######################################################################################################################
# Process data
# ======================================================================================================================
if hasattr(parameter_set, "task_pars"):
	accept_idx = np.where(np.array(stim_pattern.Output['Accepted']) == 'A')[0]
else:
	accept_idx = None

target_matrix = np.array(target_set.full_set.todense())
results = process_states(net, enc_layer, target_matrix, stim_set, data_sets=None, accepted_idx=accept_idx, plot=plot,
                   display=display, save=save, save_paths=paths)
results.update({'timing_info': timing, 'epochs': epochs})

# ######################################################################################################################
# Analyse active state
# ======================================================================================================================
# characterize population activity
if parameter_set.analysis_pars.store_activity:
	start_analysis = time.time()
	analysis_interval = [epochs['analysis_start'], nest.GetKernelStatus()['time'] - nest.GetKernelStatus()[
		'resolution']]
	results.update(characterize_population_activity(net, parameter_set, analysis_interval, epochs=None,
	                                                color_map='jet', plot=plot,
	                                                display=display, save=paths['figures']+paths['label'],
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
# for n_pop in list(itertools.chain(*[net.merged_populations, net.populations, enc_layer.encoders])):
# 	if n_pop.decoding_layer is not None:
# 		dec_layer = n_pop.decoding_layer
# 		results['performance'].update({n_pop.name: {}})
# 		results['dimensionality'].update({n_pop.name: {}})
# 		if parameter_set.analysis_pars.store_activity:
# 			if isinstance(parameter_set.analysis_pars.store_activity, int) and \
# 					parameter_set.analysis_pars.store_activity <= parameter_set.stim_pars.test_set_length:
# 				dec_layer.sampled_times = dec_layer.sampled_times[-parameter_set.analysis_pars.store_activity:]
# 			else:
# 				dec_layer.sampled_times = dec_layer.sampled_times[-parameter_set.stim_pars.test_set_length:]
# 			dec_layer.evaluate_decoding(n_neurons=50, display=display, save=paths['figures'] + paths['label'])
#
# 		labels = getattr(target_set, "{0}_set_labels".format(set_name))
# 		target = np.array(getattr(target_set, "{0}_set".format(set_name)).todense())
#
# 		test_idx = []
# 		start_t = parameter_set.stim_pars.transient_set_length + parameter_set.stim_pars.train_set_length
# 		for idx in accept_idx:
# 			if idx >= start_t:
# 				test_idx.append(idx - start_t)
# 		assert (len(test_idx) == target.shape[1]), "Incorrect test labels"
#
# 		for idx_var, var in enumerate(dec_layer.state_variables):
# 			results['performance'][n_pop.name].update({var + str(idx_var): {}})
# 			readouts = dec_layer.readouts[idx_var]
# 			state_matrix = dec_layer.state_matrix[idx_var]
#
# 			if not empty(labels) and not empty(state_matrix):
# 				print "\nPopulation {0}, variable {1}, set {2}: {3}".format(n_pop.name, var, set_name,
# 				                                                          str(state_matrix.shape))
# 				# results['rank'][n_pop.name].update({var + str(idx_var) + '_{0}'.format(set_name): get_state_rank(
# 				# 	state_matrix)})
# 				results['dimensionality'][n_pop.name].update({var + str(idx_var): compute_dimensionality(
# 					state_matrix)})
# 				for readout in readouts:
# 					results['performance'][n_pop.name][var + str(idx_var)].update({readout.name: readout_test(readout,
# 					                                state_matrix, target=target, index=None, accepted=test_idx,
# 					                                display=display)})
# 					results['performance'][n_pop.name][var + str(idx_var)][readout.name].update({'norm_wOut':
# 						                                                                             readout.norm_wout})
# 				if save:
# 					np.save(paths['activity']+paths['label']+'_population{0}_state{1}_{2}.npy'.format(n_pop.name,
# 					                                                    var, set_name), state_matrix)
# 				analyse_state_matrix(state_matrix, labels, label=n_pop.name + var + set_name,
# 					                     plot=plot, display=display, save=paths['figures'] + paths['label'])
# 		dec_layer.flush_states()



# ######################################################################################################################
# Save data
# ======================================================================================================================
if save:
	with open(paths['results'] + 'Results_' + parameter_set.label, 'w') as f:
		pickle.dump(results, f)
	parameter_set.save(paths['parameters'] + 'Parameters_' + parameter_set.label)

