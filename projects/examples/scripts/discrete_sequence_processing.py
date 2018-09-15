__author__ = 'duarte'
from modules.parameters import ParameterSet, ParameterSpace, extract_nestvalid_dict
from modules.input_architect import EncodingLayer, StimulusSet, InputSignalSet
from modules.net_architect import Network
from modules.io import set_storage_locations
from modules.signals import iterate_obj_list, empty
from modules.visualization import set_global_rcParams, plot_input_example
from modules.auxiliary import process_input_sequence, process_states, set_decoder_times
from modules.analysis import compile_performance_results
import cPickle as pickle
import numpy as np
import itertools
import matplotlib.pyplot as pl
import time
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
params_file = '../parameters/spike_pattern_input.py'

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
print('\nRuning ParameterSet {0}'.format(parameter_set.label))
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
# Build Stimulus/Target datasets
# ======================================================================================================================
stim_set_startbuild = time.time()

stim_set = StimulusSet(parameter_set, unique_set=False)
stim_set.generate_datasets(parameter_set.stim_pars)

target_set = StimulusSet(parameter_set, unique_set=False)  # for identity task.
output_sequence = list(itertools.chain(*stim_set.full_set_labels))
target_set.generate_datasets(parameter_set.stim_pars, external_sequence=output_sequence)

# correct N for small sequences
parameter_set.input_pars.signal.N = len(np.unique(stim_set.full_set_labels))

stim_set_buildtime = time.time() - stim_set_startbuild
print("- Elapsed Time: {0}".format(str(stim_set_buildtime)))

########################################################################################################################
# Build Input Signal Sets
# ======================================================================================================================
input_set_time = time.time()

inputs = InputSignalSet(parameter_set, stim_set, online=online)
inputs.generate_datasets(stim_set)

input_set_buildtime = time.time() - input_set_time
print("- Elapsed Time: {0}".format(str(input_set_buildtime)))

parameter_set.kernel_pars.sim_time = inputs.train_stimulation_time + inputs.test_stimulation_time

# Plot example signal
if plot and debug and not online:
    plot_input_example(stim_set, inputs, set_name='test', display=display, save=paths['figures'] + paths[
        'label'])
if save:
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
    set_decoder_times(enc_layer, parameter_set.input_pars, parameter_set.encoding_pars,
                      parameter_set.decoding_pars)  # iff using the fast sampling method!
    net.connect_decoders(parameter_set.decoding_pars)

# Attach decoders to input encoding populations
if not empty(enc_layer.encoders) and hasattr(parameter_set.encoding_pars, "input_decoder") and \
        parameter_set.encoding_pars.input_decoder is not None:
    enc_layer.connect_decoders(parameter_set.encoding_pars.input_decoder)

# ######################################################################################################################
# Run Simulation (full sequence)
# ======================================================================================================================
# fast state sampling
epochs, timing = process_input_sequence(parameter_set, net, [enc_layer], [stim_set], [inputs], set_name='full',
                                        record=True)

# ######################################################################################################################
# Process data
# ======================================================================================================================
target_matrix = dict(EI=np.array(target_set.full_set.todense()))

results = process_states(net, target_matrix, stim_set, data_sets=None, save=save,
                         accepted_idx=None, plot=plot, display=display, save_paths=paths)

results.update({'timing_info': timing, 'epochs': epochs})
processed_results = dict()
for ctr, n_pop in enumerate(list(itertools.chain(*[net.merged_populations, net.populations, enc_layer.encoders]))):
    if n_pop.decoding_layer is not None:
        processed_results.update({n_pop.name: {}})
        dec_layer = n_pop.decoding_layer
        for idx_var, var in enumerate(dec_layer.state_variables):
            processed_results[n_pop.name].update({var: compile_performance_results(dec_layer.readouts[idx_var], var)})

            # The labels are often not properly ordered
            readout_labels = processed_results[n_pop.name][var]['labels']
            all_indices = np.array([int(''.join(c for c in x if c.isdigit())) for x in readout_labels])
            ordered_indices = np.argsort(all_indices)

            # Extract and plot example results
            if plot:
                ordered_accuracy = processed_results[n_pop.name][var]['accuracy'][ordered_indices]
                fig, ax = pl.subplots()
                ax.plot(all_indices[ordered_indices], ordered_accuracy, 'o-', lw=2)
                ax.plot(all_indices[ordered_indices],
                        1. / np.ones_like(ordered_accuracy) * parameter_set.stim_pars.n_stim, '--r')
                ax.set_xlabel(r'$lag [n]$')
                ax.set_ylabel(r'Accuracy')
                if display:
                    pl.show(False)
                if save:
                    fig.savefig(paths['figures'] + paths['label'])

# ######################################################################################################################
# Save data
# ======================================================================================================================
if save:
    with open(paths['results'] + 'Results_' + parameter_set.label, 'w') as f:
        pickle.dump(results, f)
    with open(paths['results'] + 'ProcessedResults_' + parameter_set.label, 'w') as f:
        pickle.dump(processed_results, f)
    parameter_set.save(paths['parameters'] + 'Parameters_' + parameter_set.label)
