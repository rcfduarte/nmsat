__author__ = 'duarte'
from modules.parameters import ParameterSet, ParameterSpace, extract_nestvalid_dict
from modules.input_architect import EncodingLayer, InputNoise
from modules.net_architect import Network
from modules.io import set_storage_locations
from modules.signals import iterate_obj_list
from modules.visualization import set_global_rcParams, InputPlots
from modules.analysis import characterize_population_activity, compute_ainess
import cPickle as pickle
import numpy as np
import nest
import time

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
# params_file = '../parameters/dc_noise_input.py'
params_file = '../parameters/spike_noise_input.py'

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
if hasattr(parameter_set, "input_pars"):
	# Current input (need to build noise signal)
	total_stimulation_time = parameter_set.kernel_pars.sim_time + parameter_set.kernel_pars.transient_t
	input_noise = InputNoise(parameter_set.input_pars.noise,
	                         stop_time=total_stimulation_time)
	input_noise.generate()
	input_noise.re_seed(parameter_set.kernel_pars.np_seed)

	if plot:
		inp_plot = InputPlots(stim_obj=None, input_obj=None, noise_obj=input_noise)
		inp_plot.plot_noise_component(display=display, save=False)

	enc_layer = EncodingLayer(parameter_set.encoding_pars, signal=input_noise)
	enc_layer.connect(parameter_set.encoding_pars, net)
else:
	# Poisson input
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

net.simulate(parameter_set.kernel_pars.sim_time)  # +.1 to acquire last step...

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
results.update(characterize_population_activity(net, parameter_set, analysis_interval, epochs=None,
                                                color_map='jet', plot=plot,
                                                display=display, save=paths['figures']+paths['label'],
                                                **parameter_set.analysis_pars))
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

main_metrics = ['ISI_distance', 'SPIKE_distance', 'ccs_pearson', 'cvs', 'cvs_log', 'd_vp', 'd_vr', 'ents', 'ffs']
results.update({'ainess': compute_ainess(results, main_metrics, template_duration=analysis_interval[1] -
                                                                             analysis_interval[0],
               template_resolution=parameter_set.kernel_pars.resolution, **parameter_set.analysis_pars)})

# ######################################################################################################################
# Save data
# ======================================================================================================================
if save:
	with open(paths['results'] + 'Results_' + parameter_set.label, 'w') as f:
		pickle.dump(results, f)
	parameter_set.save(paths['parameters'] + 'Parameters_' + parameter_set.label)