__author__ = 'duarte'
import sys
sys.path.insert(0, "../")
from modules.parameters import ParameterSet, ParameterSpace, extract_nestvalid_dict
from modules.input_architect import EncodingLayer, StimulusSet, InputSignalSet, InputNoise
from modules.net_architect import Network
from modules.io import set_storage_locations
from modules.signals import iterate_obj_list, empty
from modules.visualization import set_global_rcParams, InputPlots, ActivityAnimator, plot_input_example
from modules.analysis import characterize_population_activity
from stimulus_generator import StimulusPattern
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as pl
import time
import nest


# ######################################################################################################################
# Experiment options
# ======================================================================================================================
online = False # strictly False!
plot = True
display = False
save = True
debug = True

# ######################################################################################################################
# Extract parameters from file and build global ParameterSet
# ======================================================================================================================
# params_file = '../parameters/dcinput_noise_vs_stimulus.py'
params_file = '../parameters/spike_noise_vs_stimulus.py'

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

###################################################################################
# Build network
# =================================================================================
net = Network(parameter_set.net_pars)
# net.merge_subpopulations([net.populations[0], net.populations[1]], name='EI')

###################################################################################
# Randomize initial variable values
# =================================================================================
for idx, n in enumerate(list(iterate_obj_list(net.populations))):
	if hasattr(parameter_set.net_pars, "randomize_neuron_pars"):
		randomize = parameter_set.net_pars.randomize_neuron_pars[idx]
		for k, v in randomize.items():
			n.randomize_initial_states(k, randomization_function=v[0], **v[1])

# ######################################################################################################################
# Build and connect noise input
# ======================================================================================================================
if hasattr(parameter_set, "noise_pars"):
	# Current input (need to build noise signal)
	input_noise = InputNoise(parameter_set.noise_pars.noise,
	                         stop_time=parameter_set.kernel_pars.transient_t)
	input_noise.generate()
	input_noise.re_seed(parameter_set.kernel_pars.np_seed)

	if plot:
		inp_plot = InputPlots(stim_obj=None, input_obj=None, noise_obj=input_noise)
		inp_plot.plot_noise_component(display=display, save=False)

	enc_layer_noise = EncodingLayer(parameter_set.noise_encoding_pars, signal=input_noise,
	                                prng=parameter_set.kernel_pars.np_seed)
	enc_layer_noise.connect(parameter_set.noise_encoding_pars, net)
else:
	# Poisson input
	enc_layer_noise = EncodingLayer(parameter_set.noise_encoding_pars)
	enc_layer_noise.connect(parameter_set.noise_encoding_pars, net)

###################################################################################
# Build Stimulus Set
# =================================================================================
stim_set_startbuild = time.time()

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

print "- Elapsed Time: {0}".format(str(time.time()-stim_set_startbuild))
stim_set_buildtime = time.time()-stim_set_startbuild

###################################################################################
# Build Input Signal Set
# =================================================================================
input_set_time = time.time()
parameter_set.input_pars.signal.N = len(np.unique(input_sequence))

# Create InputSignalSet
inputs = InputSignalSet(parameter_set, stim_set, online=online)
inputs.generate_datasets(stim_set)

# inputs.time_offset(parameter_set.kernel_pars.transient_t)

input_set_buildtime = time.time() - input_set_time
print "- Elapsed Time: {0}".format(str(input_set_buildtime))

# Plot example signal
if plot and debug and not online:
	plot_input_example(stim_set, inputs, set_name='full', display=display, save=paths['figures'] + paths[
		'label'])
if save:
	stim_pattern.save(paths['inputs'])
	stim_set.save(paths['inputs'])
	if debug:
		inputs.save(paths['inputs'])

#######################################################################################
# Encode Input
# =====================================================================================
if not online:
	input_signal = inputs.full_set_signal
else:
	input_signal = inputs.transient_set_signal

enc_layer = EncodingLayer(parameter_set.stim_encoding_pars, signal=input_signal, online=online)
enc_layer.connect(parameter_set.stim_encoding_pars, net)

#######################################################################################
# Set-up Analysis
# =====================================================================================
net.connect_devices()

if hasattr(parameter_set, "decoding_pars"):
	net.connect_decoders(parameter_set.decoding_pars)

# Attach decoders to input encoding populations
if not empty(enc_layer.encoders) and hasattr(parameter_set.encoding_pars, "input_decoder") and \
				parameter_set.encoding_pars.input_decoder is not None:
	enc_layer.connect_decoders(parameter_set.encoding_pars.input_decoder)

######################################################################################
# Connect Network
# ====================================================================================
net.connect_populations(parameter_set.connection_pars)

######################################################################################
# Simulate (Ongoing+evoked Epoch)
# ====================================================================================
net.simulate(parameter_set.kernel_pars.transient_t + parameter_set.kernel_pars.sim_time + nest.GetKernelStatus()[
	'min_delay'])
net.extract_population_activity()
net.extract_network_activity()

# sanity check
activity = []
for spikes in net.spiking_activity:
	activity.append(spikes.mean_rate())
if not np.mean(activity) > 0:
	raise ValueError("No activity recorded in main network! Stopping simulation..")

# ######################################################################################################################
# Analyse / plot data
# ======================================================================================================================
analysis_interval = [100, parameter_set.kernel_pars.transient_t] # discard the first 100 ms

results['ongoing'] = characterize_population_activity(net, parameter_set, analysis_interval, epochs=None,
                                                color_map='Accent', plot=plot, color_subpop=True,
                                                display=display, save=paths['figures']+paths['label']+'Ongoing',
                                                analysis_pars=parameter_set.analysis_pars)

analysis_interval = [parameter_set.kernel_pars.transient_t, parameter_set.kernel_pars.transient_t +
                     parameter_set.kernel_pars.sim_time]

results['evoked'] = characterize_population_activity(net, parameter_set, analysis_interval, epochs=None,
                                                color_map='Accent', plot=plot, color_subpop=True,
                                                display=display, save=paths['figures']+paths['label']+'Evoked',
                                                     analysis_pars=parameter_set.analysis_pars)

analysis_interval = [parameter_set.kernel_pars.transient_t - 2000., parameter_set.kernel_pars.transient_t + 2000.]

parameter_set.analysis_pars.population_activity.update({
	'time_bin': 1.,
	'time_resolved': True,
	'window_len': 50})


epochs = {'ongoing': (analysis_interval[0], parameter_set.kernel_pars.transient_t),
          'evoked': (parameter_set.kernel_pars.transient_t, analysis_interval[1])}
results['transition'] = characterize_population_activity(net, parameter_set, analysis_interval, epochs=epochs,
                                                color_map='Accent', plot=plot,
                                                display=display, save=paths['figures']+paths['label']+'Evoked',
												color_subpop=True, analysis_pars=parameter_set.analysis_pars)

# net.flush_records()
# enc_layer.flush_records()

# animate raster
net.merge_subpopulations([net.populations[0], net.populations[1]], name='EI')
spk_list, _ = net.merge_population_activity(start=analysis_interval[0], stop=analysis_interval[1])
gids = [x.gids for x in net.populations]
ai = ActivityAnimator(net.populations[0].spiking_activity, populations=net, ids=gids, vm_list=[])
ai.animate_activity(time_interval=50, time_window=50, sim_res=0.1, colors=['b', 'r'], activities=[
	"raster"], save=True, filename=paths['figures']+paths['label']+'test', display=False)
print ("gonna animate raster plot... @done")
#######################################################################################
# Save data
# =====================================================================================
if save:
	with open(paths['results'] + 'Results_' + parameter_set.label, 'w') as f:
		pickle.dump(results, f)
	parameter_set.save(paths['parameters'] + 'Parameters_' + parameter_set.label)