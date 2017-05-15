__author__ = 'duarte'
from modules.parameters import ParameterSet, ParameterSpace, extract_nestvalid_dict
from modules.input_architect import EncodingLayer, InputSignalSet, InputNoise, InputSignal
from modules.net_architect import Network
from modules.io import set_storage_locations
from modules.signals import iterate_obj_list
from modules.analysis import single_neuron_responses
from modules.visualization import  InputPlots
import cPickle as pickle
import numpy as np
import scipy.stats as stats
import nest

"""
:param parameter_set: must be consistent with the computation
:param plot: plot results - either show them or save to file
:param display: show figures/reports
:param save: save results
:return results_dictionary:
"""

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
params_file = '../parameters/single_neuron_patterned_synaptic_input.py'

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
	import modules.visualization as vis
	vis.set_global_rcParams(parameter_set.kernel_pars['mpl_path'])
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

########################################################################################################################
# Build Input Signal Sets
# ======================================================================================================================
assert hasattr(parameter_set, "input_pars")
# Current input (need to build noise signal)
total_stimulation_time = parameter_set.kernel_pars.sim_time + parameter_set.kernel_pars.transient_t
input_noise = InputNoise(parameter_set.input_pars.noise, stop_time=total_stimulation_time)
input_noise.generate()
input_noise.re_seed(parameter_set.kernel_pars.np_seed)

# if plot:
# 	inp_plot = InputPlots(stim_obj=None, input_obj=None, noise_obj=input_noise)
# 	inp_plot.plot_noise_component(display=display, save=False)

# ######################################################################################################################
# Build and connect input
# ======================================================================================================================
enc_layer = EncodingLayer(parameter_set.encoding_pars, signal=input_noise)
enc_layer.connect(parameter_set.encoding_pars, net)

# ######################################################################################################################
# Connect Network
# ======================================================================================================================
net.connect_devices()
net.connect_populations(parameter_set.connection_pars)

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
net.extract_population_activity(
	t_start=parameter_set.kernel_pars.transient_t + nest.GetKernelStatus()['resolution'],
	t_stop=parameter_set.kernel_pars.sim_time + parameter_set.kernel_pars.transient_t)
net.extract_network_activity()
net.flush_records()

# ######################################################################################################################
# Analyse / plot data
# ======================================================================================================================
results = dict()
input_pops = ['E_inputs', 'I_inputs']
analysis_interval = [parameter_set.kernel_pars.transient_t,
                     parameter_set.kernel_pars.transient_t+parameter_set.kernel_pars.sim_time]
for idd, nam in enumerate(net.population_names):
	if nam not in input_pops:
		results.update({nam: {}})
		results[nam] = single_neuron_responses(net.populations[idd],
		                                       parameter_set, pop_idx=idd,
		                                       start=analysis_interval[0],
		                                       stop=analysis_interval[1],
		                                       plot=plot, display=display,
		                                       save=paths['figures']+paths['label'])
		if results[nam]['rate']:
			print 'Output Rate [{0}] = {1} spikes/s'.format(str(nam), str(results[nam]['rate']))

# ######################################################################################################################
# Save data
# ======================================================================================================================
if save:
	with open(paths['results'] + 'Results_' + parameter_set.label, 'w') as f:
		pickle.dump(results, f)
	parameter_set.save(paths['parameters'] + 'Parameters_' + parameter_set.label)