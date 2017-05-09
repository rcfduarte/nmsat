__author__ = 'duarte'
from modules.parameters import ParameterSet, ParameterSpace, extract_nestvalid_dict
from modules.input_architect import EncodingLayer, InputNoise
from modules.net_architect import Network
from modules.io import set_storage_locations
from modules.signals import iterate_obj_list, empty, narma
from modules.analysis import characterize_population_activity, analyse_state_matrix
from modules.visualization import set_global_rcParams, InputPlots, plot_target_out
import cPickle as pickle
import numpy as np
import itertools
import time
import nest

# ######################################################################################################################
# Experiment options
# ======================================================================================================================
plot = True
display = True
save = True
debug = False

# ######################################################################################################################
# Extract parameters from file and build global ParameterSet
# ======================================================================================================================
params_file = '../parameters/dc_noise_input.py'

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
results = dict(performance={}, activity={})

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
# Build Inputs/Targets
# ======================================================================================================================
stim_set_startbuild = time.time()

# Current input (build noise signal)
total_stimulation_time = parameter_set.kernel_pars.sim_time + parameter_set.kernel_pars.transient_t
input_noise = InputNoise(parameter_set.input_pars.noise, stop_time=total_stimulation_time)
input_noise.generate()
input_noise.re_seed(parameter_set.kernel_pars.np_seed)

if plot:
	inp_plot = InputPlots(stim_obj=None, input_obj=None, noise_obj=input_noise)
	inp_plot.plot_noise_component(display=display, save=False)

# ######################################################################################################################
# Encode Input
# ======================================================================================================================
enc_layer = EncodingLayer(parameter_set.encoding_pars, signal=input_noise)
enc_layer.connect(parameter_set.encoding_pars, net)

stim_set_buildtime = time.time()-stim_set_startbuild
print "- Elapsed Time: {0}".format(str(stim_set_buildtime))

# ######################################################################################################################
# Set-up Devices
# ======================================================================================================================
net.connect_devices()
if hasattr(parameter_set, "decoding_pars"):
	# set_decoder_times(enc_layer, parameter_set) # iff using the fast sampling method!
	net.connect_decoders(parameter_set.decoding_pars)

# Attach decoders to input encoding populations
if not empty(enc_layer.encoders) and hasattr(parameter_set.encoding_pars, "input_decoder") and \
				parameter_set.encoding_pars.input_decoder is not None:
	enc_layer.connect_decoders(parameter_set.encoding_pars.input_decoder)

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

net.simulate(parameter_set.kernel_pars.sim_time + nest.GetKernelStatus()['resolution'])

# ######################################################################################################################
# Extract and store data
# ======================================================================================================================
net.extract_population_activity()
net.extract_network_activity()
net.flush_records()

# ######################################################################################################################
# Analyse / plot data
# ======================================================================================================================
analysis_interval = [parameter_set.kernel_pars.transient_t,
	                     parameter_set.kernel_pars.sim_time + parameter_set.kernel_pars.transient_t]
parameter_set.analysis_pars.pop('label')
start_analysis = time.time()

# Characterize population activity
results['activity'].update(characterize_population_activity(net, parameter_set, analysis_interval, epochs=None,
                                                color_map='jet', plot=plot,
                                                display=display, save=paths['figures']+paths['label'],
                                                color_subpop=True, analysis_pars=parameter_set.analysis_pars))
print "\nElapsed time (state characterization): {0}".format(str(time.time() - start_analysis))


# Read and process all decoded data
for ctr, n_pop in enumerate(list(itertools.chain(*[net.merged_populations,
                                                   net.populations, enc_layer.encoders]))):
	if n_pop.decoding_layer is not None:
		results['performance'].update({n_pop.name: {}})

		# extract decoder data
		if empty(n_pop.decoding_layer.activity):
			n_pop.decoding_layer.extract_activity(start=analysis_interval[0],
			                                      stop=analysis_interval[1] - nest.GetKernelStatus()['resolution'], save=True)
		# parse state matrices
		for idx_state, n_state in enumerate(n_pop.decoding_layer.state_variables):
			results['performance'][n_pop.name].update({n_state + str(idx_state): {}})

			n_pop.decoding_layer.state_matrix[idx_state] = n_pop.decoding_layer.activity[idx_state].as_array()

			# readout each state matrix
			readouts = n_pop.decoding_layer.readouts[idx_state]
			state_matrix = n_pop.decoding_layer.state_matrix[idx_state]

			# analyse responses
			analyse_state_matrix(state_matrix, label=n_state + str(idx_state), plot=plot, display=display,
			                          save=paths['figures'] + paths['label'])

			# time slice and normalize
			input_signal = input_noise.noise_signal.time_slice(analysis_interval[0], analysis_interval[1]).as_array()\
			               / parameter_set.input_pars.noise.noise_pars['amplitude']

			# generate targets
			target_signal = narma(input_signal, n=30)

			for readout in readouts:
				# train
				readout.train(state_matrix, target_signal, display=display)
				readout.measure_stability(display=display)

				# test
				output, tgt = readout.test(state_matrix, target_signal, display=display)
				# TODO - calculate_error(output, target) instead of measure_performance
				results['performance'][n_pop.name][n_state + str(idx_state)].update(
					{readout.name: readout.measure_performance(tgt, output, display=display)})
				results['performance'][n_pop.name][n_state + str(idx_state)][readout.name].update(
					{'norm_wOut': readout.norm_wout})

				if plot:
					readout.plot_weights(display=display, save=paths['figures'] + paths['label'])

					time_axis = np.arange(analysis_interval[0], analysis_interval[1], input_noise.dt)
					plot_target_out(tgt, output, time_axis=np.arange(analysis_interval[0], analysis_interval[1],
					                input_noise.dt), label=n_state + str(idx_state) + readout.name,
					                display=display, save=paths['figures'] + paths['label'])

# ######################################################################################################################
# Save data
# ======================================================================================================================
if save:
	with open(paths['results'] + 'Results_' + parameter_set.label, 'w') as f:
		pickle.dump(results, f)
	parameter_set.save(paths['parameters'] + 'Parameters_' + parameter_set.label)

