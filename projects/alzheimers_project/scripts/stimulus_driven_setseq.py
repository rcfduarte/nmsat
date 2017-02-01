__author__ = 'duarte'
from modules.parameters import ParameterSet, ParameterSpace, extract_nestvalid_dict
from modules.input_architect import EncodingLayer, StimulusSet, InputSignalSet
from modules.net_architect import Network
from modules.io import set_storage_locations
from modules.signals import iterate_obj_list, empty
from modules.visualization import set_global_rcParams, InputPlots, extract_encoder_connectivity, TopologyPlots
from modules.analysis import analyse_state_matrix, get_state_rank, readout_train, readout_test
from modules.auxiliary import iterate_input_sequence
import cPickle as pickle
import matplotlib.pyplot as pl
import numpy as np
import time
import itertools
import nest

# ######################################################################################################################
# Experiment options
# ======================================================================================================================
plot = True
display = True
save = False
debug = False
online = True

# ######################################################################################################################
# Extract parameters from file and build global ParameterSet
# ======================================================================================================================
params_file = '../parameters/stimulus_driven.py'

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
results = dict(rank={}, performance={})

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
net.merge_subpopulations([net.populations[0], net.populations[1]])

# ######################################################################################################################
# Randomize initial variable values
# ======================================================================================================================
for n in list(iterate_obj_list(net.populations)):
	n.randomize_initial_states('V_m', randomization_function=np.random.uniform, low=0.0, high=15.)

# ######################################################################################################################
# Build and connect input
# ======================================================================================================================
# Create StimulusSet
stim_set_time = time.time()
stim = StimulusSet(parameter_set, unique_set=True)
stim.create_set(parameter_set.stim_pars.full_set_length)
stim.discard_from_set(parameter_set.stim_pars.transient_set_length)
stim.divide_set(parameter_set.stim_pars.transient_set_length, parameter_set.stim_pars.train_set_length,
                parameter_set.stim_pars.test_set_length)
print "- Elapsed Time: {0}".format(str(time.time()-stim_set_time))

# Create InputSignalSet
input_set_time = time.time()
inputs = InputSignalSet(parameter_set, stim, online=online)
inputs.generate_datasets(stim)
print "- Elapsed Time: {0}".format(str(time.time() - input_set_time))

# Plot example signal
if plot and debug and not online:
	fig_inp = pl.figure()
	ax1 = fig_inp.add_subplot(211)
	ax2 = fig_inp.add_subplot(212)
	fig_inp.suptitle('Input Stimulus / Signal')
	inp_plot = InputPlots(stim_obj=stim, input_obj=inputs.train_set_signal, noise_obj=inputs.train_set_noise)
	inp_plot.plot_stimulus_matrix(set='train', ax=ax1, save=False, display=False)
	inp_plot.plot_input_signal(ax=ax2, save=paths['figures']+paths['label'], display=display)
	inp_plot.plot_input_signal(save=paths['figures']+paths['label'], display=display)
	inp_plot.plot_signal_and_noise(save=paths['figures']+paths['label'], display=display)
parameter_set.kernel_pars.sim_time = inputs.train_stimulation_time + inputs.test_stimulation_time

if save:
	stim.save(paths['inputs'])
	if debug:
		inputs.save(paths['inputs'])

# ######################################################################################################################
# Encode Input
# ======================================================================================================================
if not online:
	input_signal = inputs.full_set_signal
else:
	input_signal = inputs.transient_set_signal
enc_layer = EncodingLayer(parameter_set.encoding_pars, signal=input_signal, online=online)
enc_layer.connect(parameter_set.encoding_pars, net)
enc_layer.extract_connectivity(net)

# ######################################################################################################################
# Set-up Analysis
# ======================================================================================================================
net.connect_devices()
net.connect_decoders(parameter_set.decoding_pars)

# Attach decoders to input encoding populations
if not empty(enc_layer.encoders) and hasattr(parameter_set.encoding_pars, "input_decoder"):
	enc_layer.connect_decoders(parameter_set.encoding_pars.input_decoder)

# ######################################################################################################################
# Connect Network
# ======================================================================================================================
net.connect_populations(parameter_set.connection_pars)

# ######################################################################################################################
# Simulate (Transient Set)
# ======================================================================================================================
set_name = 'transient'
if stim.transient_set_labels:
	iterate_input_sequence(net, enc_layer, parameter_set, stim, inputs, set_name=set_name, record=True,
	                       store_activity=False)
	for n_pop in list(itertools.chain(*[net.merged_populations, net.populations, enc_layer.encoders])):
		if n_pop.decoding_layer is not None:
			n_pop.decoding_layer.flush_states()

# ######################################################################################################################
# Simulate (Unique Set)
# ======================================================================================================================
if hasattr(stim, "unique_set"):
	set_name = 'unique'
	iterate_input_sequence(net, enc_layer, parameter_set, stim, inputs, set_name=set_name, record=True,
	                       store_activity=False)

	# compute ranks
	for n_pop in list(itertools.chain(*[net.merged_populations, net.populations, enc_layer.encoders])):
		if n_pop.decoding_layer is not None:
			dec_layer = n_pop.decoding_layer
			labels = getattr(stim, "{0}_set_labels".format(set_name))

			results['rank'].update({n_pop.name: {}})
			for idx_var, var in enumerate(dec_layer.state_variables):
				state_matrix = dec_layer.state_matrix[idx_var]

				if not empty(labels) and not empty(state_matrix):
					print "Population {0}, variable {1}, set {2}: {3}".format(n_pop.name, var, set_name,
					                                                          str(state_matrix.shape))
					results['rank'][n_pop.name].update({var + str(idx_var): get_state_rank(state_matrix)})

					if save:
						np.save(paths['activity']+paths['label']+'_population{0}_state{1}_{2}.npy'.format(n_pop.name,
						                                                    var, set_name), state_matrix)
					analyse_state_matrix(state_matrix, labels, label=n_pop.name+var+set_name,
					                     plot=plot, display=display, save=paths['figures']+paths['label'])
			dec_layer.flush_states()

# ######################################################################################################################
# Simulate (Train Set)
# ======================================================================================================================
set_name = 'train'
iterate_input_sequence(net, enc_layer, parameter_set, stim, inputs, set_name=set_name, record=True,
                       store_activity=False)

# train readouts
for n_pop in list(itertools.chain(*[net.merged_populations, net.populations, enc_layer.encoders])):
	if n_pop.decoding_layer is not None:
		dec_layer = n_pop.decoding_layer
		labels = getattr(stim, "{0}_set_labels".format(set_name))
		target = np.array(getattr(stim, "{0}_set".format(set_name)).todense())

		for idx_var, var in enumerate(dec_layer.state_variables):
			readouts = dec_layer.readouts[idx_var]
			state_matrix = dec_layer.state_matrix[idx_var]
			if not empty(labels) and not empty(state_matrix):
				print "Population {0}, variable {1}, set {2}: {3}".format(n_pop.name, var, set_name,
				                                                          str(state_matrix.shape))
				for readout in readouts:
					readout_train(readout, state_matrix, target=target, index=None, accepted=None,
					              display=display, plot=plot, save=paths['figures']+paths['label'])
				if save:
					np.save(paths['activity']+paths['label']+'_population{0}_state{1}_{2}.npy'.format(n_pop.name,
					                                                    var, set_name), state_matrix)
				analyse_state_matrix(state_matrix, labels, label=n_pop.name + var + set_name,
				                     plot=plot, display=display, save=paths['figures'] + paths['label'])
		dec_layer.flush_states()
# ######################################################################################################################
# Simulate (Test Set)
# ======================================================================================================================
set_name = 'test'
iterate_input_sequence(net, enc_layer, parameter_set, stim, inputs, set_name=set_name, record=True,
                       store_activity=False)

# test readouts
for n_pop in list(itertools.chain(*[net.merged_populations, net.populations, enc_layer.encoders])):
	if n_pop.decoding_layer is not None:
		dec_layer = n_pop.decoding_layer
		labels = getattr(stim, "{0}_set_labels".format(set_name))
		target = np.array(getattr(stim, "{0}_set".format(set_name)).todense())

		for idx_var, var in enumerate(dec_layer.state_variables):
			results['performance'].update({n_pop.name: {var: {}}})
			readouts = dec_layer.readouts[idx_var]
			state_matrix = dec_layer.state_matrix[idx_var]
			if not empty(labels) and not empty(state_matrix):
				print "Population {0}, variable {1}, set {2}: {3}".format(n_pop.name, var, set_name,
				                                                          str(state_matrix.shape))
				for readout in readouts:
					results['performance'][n_pop.name][var].update({readout.name: readout_test(readout, state_matrix,
								                    target=target, index=None, accepted=None, display=display)})
				if save:
					np.save(paths['activity']+paths['label']+'_population{0}_state{1}_{2}.npy'.format(n_pop.name,
					                                                    var, set_name), state_matrix)
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


########################################################################################################################
