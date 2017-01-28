import itertools

__author__ = 'duarte'
from modules.parameters import ParameterSet, ParameterSpace, extract_nestvalid_dict
from modules.input_architect import EncodingLayer, StimulusSet, InputSignalSet
from modules.net_architect import Network
from modules.io import set_storage_locations
from modules.signals import iterate_obj_list, empty
from modules.visualization import set_global_rcParams, InputPlots, extract_encoder_connectivity, TopologyPlots
from modules.analysis import characterize_population_activity, compute_ainess
from modules.auxiliary import iterate_input_sequence
import cPickle as pickle
import matplotlib.pyplot as pl
import numpy as np
import time
import nest

# ######################################################################################################################
# Experiment options
# ======================================================================================================================
plot = True
display = True
save = True
debug = True
online = False

# ######################################################################################################################
# Extract parameters from file and build global ParameterSet
# ======================================================================================================================
params_file = '../parameters/dc_input.py'
# params_file = '../parameters/spike_pattern_input.py'

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
net.merge_subpopulations([net.populations[0], net.populations[1]])

# ######################################################################################################################
# Randomize initial variable values
# ======================================================================================================================
for idx, n in enumerate(list(iterate_obj_list(net.populations))):
	if hasattr(parameter_set.net_pars, "randomize_neuron_pars"):
		randomize = parameter_set.net_pars.randomize_neuron_pars[idx]
		for k, v in randomize.items():
			n.randomize_initial_states(k, randomization_function=v[0], **v[1])

# ######################################################################################################################
# Connect Network
# ======================================================================================================================
# net.connect_populations(parameter_set.connection_pars)

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

inputs.generate_full_set(stim)
if stim.transient_set_labels:
	inputs.generate_transient_set(stim)

inputs.generate_unique_set(stim)
inputs.generate_train_set(stim)
inputs.generate_test_set(stim)
print "- Elapsed Time: {0}".format(str(time.time() - input_set_time))

# Plot example signal
if plot and debug and not online:
	fig_inp = pl.figure()
	ax1 = fig_inp.add_subplot(211)
	ax2 = fig_inp.add_subplot(212)
	fig_inp.suptitle('Input Stimulus / Signal')
	inp_plot = InputPlots(stim_obj=stim, input_obj=inputs.full_set_signal, noise_obj=inputs.full_set_noise)
	inp_plot.plot_stimulus_matrix(set='full', ax=ax1, save=False, display=False)
	inp_plot.plot_input_signal(ax=ax2, save=paths['figures']+paths['label'], display=display)
	inp_plot.plot_input_signal(save=paths['figures']+paths['label'], display=display)
	inp_plot.plot_signal_and_noise(save=paths['figures']+paths['label'], display=display)
# parameter_set.kernel_pars.sim_time = inputs.train_stimulation_time + inputs.test_stimulation_time

if save:
	stim.save(paths['inputs'])
	if debug:
		inputs.save(paths['inputs'])

# ######################################################################################################################
# Encode Input
# ======================================================================================================================
input_signal = inputs.full_set_signal

enc_layer = EncodingLayer(parameter_set.encoding_pars, signal=input_signal, online=online)
enc_layer.connect(parameter_set.encoding_pars, net)


# if plot and debug:
# 	extract_encoder_connectivity(enc_layer, net, display, save=paths['figures']+paths['label'])

# ######################################################################################################################
# Set-up Analysis
# ======================================================================================================================
net.connect_devices()
net.connect_decoders(parameter_set.decoding_pars)

# Attach decoders to input encoding populations
if not empty(enc_layer.encoders) and hasattr(parameter_set.encoding_pars, "input_decoder"):
	enc_layer.connect_decoders(parameter_set.encoding_pars.input_decoder)


enc_layer.extract_connectivity(net)


# determine timing compensations required
enc_layer.determine_total_delay()


######################################################################################################
for n_pop in list(itertools.chain(*[net.merged_populations, net.populations])):
	if n_pop.decoding_layer is not None:
		n_pop.decoding_layer.determine_total_delay()

fig, ax = pl.subplots()
ax.plot(nest.GetStatus(enc_layer.generators[0].gids[0])[0]['amplitude_times'], nest.GetStatus(enc_layer.generators[
	                                                                            0].gids[0])[0]['amplitude_values'])
pl.show()