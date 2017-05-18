# TODO check for input / simulation resolution match = error

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

plot 	= True
display = True
save 	= True
debug 	= True
online 	= True
###################################################################################
# Extract parameters from file and build global ParameterSet
# =================================================================================
params_file = '../parameters/p_identity.py'
set_global_rcParams('../../../defaults/matplotlib_rc')

parameter_set = ParameterSpace(params_file)[0]
parameter_set = parameter_set.clean(termination='pars')

if not isinstance(parameter_set, ParameterSet):
	if isinstance(parameter_set, basestring) or isinstance(parameter_set, dict):
		parameter_set = ParameterSet(parameter_set)
	else:
		raise TypeError("parameter_set must be ParameterSet, string with full path to parameter file or dictionary")

###################################################################################
# Setup extra variables and parameters
# =================================================================================
if plot:
	set_global_rcParams(parameter_set.kernel_pars['mpl_path'])
paths = set_storage_locations(parameter_set, save)

np.random.seed(parameter_set.kernel_pars['np_seed'])
results = dict()

##################################################################################
# Set kernel and simulation parameters
# ================================================================================
print '\nRuning ParameterSet {0}'.format(parameter_set.label)
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
	stim_set = StimulusSet(unique_set=False)
	stim_set.generate_datasets(parameter_set.stim_pars, external_sequence=input_sequence)

	# Specify target and convert to StimulusSet object
	target_set = StimulusSet(unique_set=False)
	target_set.generate_datasets(parameter_set.stim_pars, external_sequence=output_sequence)
else:
	stim = StimulusSet(parameter_set, unique_set=False)
	stim.generate_datasets(parameter_set.stim_pars)

	# for identity task
	target_set = StimulusSet(parameter_set, unique_set=False)
	output_sequence = list(itertools.chain(*stim.full_set_labels))
	target_set.generate_datasets(parameter_set.stim_pars, external_sequence=output_sequence)


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

#######################################################################################
# Set-up Analysis
# =====================================================================================
net.connect_devices()
auxiliary.set_decoder_times(enc_layer, parameter_set) # iff using the fast sampling method!
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

target_matrix = np.array(target_set.full_set.todense())
results = auxiliary.process_states(net, enc_layer, target_matrix, stim, data_sets=None,
								   accepted_idx=accept_idx, plot=plot,
								   display=display, save=save, save_paths=paths)
results.update({'timing_info': timing, 'epochs': epochs})

# ######################################################################################################################
# Save data
# ======================================================================================================================
if save:
	with open(paths['results'] + 'Results_' + parameter_set.label, 'w') as f:
		pickle.dump(results, f)
	parameter_set.save(paths['parameters'] + 'Parameters_' + parameter_set.label)

# ####################
# if plot and debug and False:
# 	fig_W = pl.figure()
# 	topology = TopologyPlots(parameter_set.connection_pars, net)
# 	topology.print_network(depth=3)
# 	ax1 = pl.subplot2grid((6, 6), (0, 0), rowspan=4, colspan=4)
# 	ax2 = pl.subplot2grid((6, 6), (5, 0), rowspan=1, colspan=4)
# 	ax3 = pl.subplot2grid((6, 6), (0, 5), rowspan=4, colspan=1)
# 	ax4 = pl.subplot2grid((6, 6), (5, 5), rowspan=1, colspan=1)
# 	topology.plot_connectivity(parameter_set.connection_pars.synapse_types,
#  	                           ax=[ax1, ax3, ax2, ax4], display=display, save=paths['figures']+paths['label'])

# ######################################################################################
# # Simulate (Initial Transient)
# # ====================================================================================
# if stim.transient_set_labels:
# 	if not online:
# 		print "\nTransient time = {0} ms".format(str(parameter_set.kernel_pars.transient_t))
#
# 	# iterate_input_sequence(net, inputs.transient_set_signal, enc_layer, sampling_times=None, stim_set=stim,
# 	#                        input_set=inputs, set_name='transient', store_responses=False, record=False,
# 	#                        jitter=parameter_set.encoding_pars.generator.jitter)
# 	auxiliary.process_input_sequence(parameter_set, net, enc_layer, stim, inputs,
# 									 set_name='full', record=True)
# 	parameter_set.kernel_pars.transient_t = nest.GetKernelStatus()['time']
# 	net.extract_population_activity(t_start=0., t_stop=parameter_set.kernel_pars.transient_t -
# 	                                                   parameter_set.kernel_pars.resolution)
# 	net.extract_network_activity()
#
# 	# sanity check
# 	activity = []
# 	for spikes in net.spiking_activity:
# 		activity.append(spikes.mean_rate())
# 	if not np.mean(activity) > 0:
# 		raise ValueError("No activity recorded in main network! Stopping simulation..")
#
# 	if parameter_set.kernel_pars.transient_t > 1000.:
# 		analysis_interval = [1000, parameter_set.kernel_pars.transient_t]
# 		results['population_activity'] = population_state(net, parameter_set=parameter_set,
# 		                                                  nPairs=500, time_bin=1.,
# 		                                                  start=analysis_interval[0],
# 		                                                  stop=analysis_interval[1] -
# 		                                                       parameter_set.kernel_pars.resolution,
# 		                                                  plot=plot, display=display,
# 		                                                  save=paths['figures']+paths['label'])
# 		enc_layer.extract_encoder_activity()
# 		# results.update(evaluate_encoding(enc_layer, parameter_set, analysis_interval,
# 		#                                  inputs.transient_set_signal, plot=plot, display=display,
# 		#                                  save=paths['figures']+paths['label']))
#
# 	net.flush_records()
# 	enc_layer.flush_records()
#
#
# #######################################################################################
# # Simulate (Train period)
# # =====================================================================================
# if not online:
# 	print "\nTrain time = {0} ms".format(str(inputs.train_stimulation_time))
# iterate_input_sequence(net, inputs.train_set_signal, enc_layer,
#                        sampling_times=parameter_set.decoding_pars.global_sampling_times,
#                        stim_set=stim, input_set=inputs, set_name='train', store_responses=False,
#                        jitter=parameter_set.encoding_pars.generator.jitter)
# #######################################################################################
# # Train Readouts
# # =====================================================================================
#
# train_all_readouts(parameter_set, net, stim, inputs.train_set_signal, encoding_layer=enc_layer, flush=True, debug=debug,
# 				   plot=plot, display=display, save=paths)
#
# #######################################################################################
# # Simulate (Test period)
# # =====================================================================================
# if not online:
# 	print "\nTest time = {0} ms".format(str(inputs.test_stimulation_time))
# iterate_input_sequence(net, inputs.test_set_signal, enc_layer,
#                        sampling_times=parameter_set.decoding_pars.global_sampling_times,
#                        stim_set=stim, input_set=inputs, set_name='test', store_responses=False,#paths['activity'],
#                        jitter=parameter_set.encoding_pars.generator.jitter)
#
# #######################################################################################
# # Test Readouts
# # =====================================================================================
# test_all_readouts(parameter_set, net, stim, inputs.test_set_signal, encoding_layer=enc_layer, flush=False, debug=debug,
# 				  plot=plot, display=display, save=paths)
#
# results['Performance'] = {}
# results['Performance'].update(analyse_performance_results(net, enc_layer, plot=plot, display=display,
# 														  save=paths['figures']+paths['label']))
#
# # #######################################################################################
# # # Save data
# # # =====================================================================================
# if save:
# 	with open(paths['results'] + 'Results_' + parameter_set.label, 'w') as f:
# 		pickle.dump(results, f)
# 	parameter_set.save(paths['parameters'] + 'Parameters_' + parameter_set.label)