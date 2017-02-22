__author__ = 'duarte'
from modules.parameters import ParameterSet, ParameterSpace, extract_nestvalid_dict
from modules.input_architect import EncodingLayer, StimulusSet, InputSignalSet
from modules.net_architect import Network
from modules.io import set_storage_locations
from modules.signals import iterate_obj_list, empty
from modules.visualization import set_global_rcParams, InputPlots, extract_encoder_connectivity, TopologyPlots, plot_input_example
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
save = True
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

###################################################################################
# Build Stimulus Set
# =================================================================================
stim_set_time = time.time()

# Create StimulusSet object
stim = StimulusSet(parameter_set, unique_set=False)
stim.generate_datasets(parameter_set.stim_pars)
print "- Elapsed Time: {0}".format(str(time.time()-stim_set_time))

###################################################################################
# Build Input Signal Set
# =================================================================================
input_set_time = time.time()
inputs = InputSignalSet(parameter_set, stim, online=online)
inputs.generate_datasets(stim)
print "- Elapsed Time: {0}".format(str(time.time() - input_set_time))

parameter_set.kernel_pars.sim_time = inputs.train_stimulation_time + inputs.test_stimulation_time

# Plot example signal
if plot and debug and not online:
	plot_input_example(stim, inputs, set_name='test', display=display, save=paths['figures'] + paths[
		'label'])
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
# Simulate (Full Set)
# ======================================================================================================================
store_activity = False  # put in analysis_pars
iterate_input_sequence(net, enc_layer, parameter_set, stim, inputs, set_name='full', record=True,
                       store_activity=store_activity)

sub_sets = ['transient', 'unique', 'train', 'test']
target_matrix = stim.full_set.todense()
for ctr, n_pop in enumerate(list(itertools.chain(*[net.merged_populations,
					                net.populations]))):#, enc_layer.encoders]))):
	if n_pop.decoding_layer is not None:
		dec_layer = n_pop.decoding_layer
		if store_activity and debug:
			dec_layer.evaluate_decoding(n_neurons=10, display=display, save=paths['figures']+paths['label'])

		results['rank'].update({n_pop.name: {}})
		results['performance'].update({n_pop.name: {}})

		# parse state variables
		for idx_var, var in enumerate(dec_layer.state_variables):
			results['performance'][n_pop.name].update({var: {}})
			time_steps = 0
			end_step = 0
			state_matrix = dec_layer.state_matrix[idx_var]
			readouts = dec_layer.readouts[idx_var]
			for stim_set in sub_sets:
				labels = getattr(stim, "{0}_set_labels".format(stim_set))
				if not empty(labels) and not empty(state_matrix):
					end_step += len(labels)
					state = state_matrix[:, time_steps:end_step]
					print "Population {0}, variable {1}, set {2}: {3}".format(n_pop.name, var, stim_set,
					                                                          str(state.shape))
					target = target_matrix[:, time_steps:end_step]
					time_steps += len(labels)
					if stim_set == 'unique':
						results['rank'][n_pop.name].update({var+str(idx_var): get_state_rank(state)})
					elif stim_set == 'train':
						for readout in readouts:
							readout_train(readout, state, target=np.array(target), index=None, accepted=None,
							              display=display, plot=plot, save=paths['figures']+paths['label'])
					elif stim_set == 'test':
						for readout in readouts:
							results['performance'][n_pop.name][var].update({readout.name: readout_test(readout, state,
							                    target=np.array(target), index=None, accepted=None, display=display)})
					if plot:
						analyse_state_matrix(state_matrix, stim.full_set_labels, label=n_pop.name+var+stim_set,
						                     plot=plot, display=display, save=paths['figures']+paths['label'])

# ######################################################################################################################
# Save data
# ======================================================================================================================
if save:
	with open(paths['results'] + 'Results_' + parameter_set.label, 'w') as f:
		pickle.dump(results, f)
	parameter_set.save(paths['parameters'] + 'Parameters_' + parameter_set.label)