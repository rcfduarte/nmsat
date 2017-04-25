__author__ = 'duarte'
import sys
sys.path.insert(0, "../")
from modules.parameters import ParameterSpace, ParameterSet, extract_nestvalid_dict
from modules.io import set_storage_locations
from modules.signals import iterate_obj_list, empty, gather_analog_activity
from modules.net_architect import Network
from modules.input_architect import EncodingLayer
from modules.visualization import set_global_rcParams
from modules.analysis import single_neuron_responses
import numpy as np
import nest
from auxiliary_fcns import PSP_kinetics, PSC_kinetics
import matplotlib.pyplot as pl
import cPickle as pickle

plot = True
display = True
save = False
debug = False

# TODO * correct (if need be)
###################################################################################
# Extract parameters from file and build global ParameterSet
# =================================================================================
params_file = '../parameters/spike_triggered_average.py'

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
# Set-up Analysis
# ======================================================================================================================
net.connect_devices()
# ######################################################################################################################
# Connect Network
# ======================================================================================================================
# net.connect_populations(parameter_set.connection_pars)

# ######################################################################################################################
# Build and Connect copy network
# ======================================================================================================================
clone_net = net.clone(parameter_set, devices=True, decoders=False)

# print nest.GetStatus(net.populations[0].gids)
# print nest.GetStatus(clone_net.populations[0].gids)

# ######################################################################################################################
# Build and connect input
# ======================================================================================================================
enc_layer = EncodingLayer(parameter_set.encoding_pars)
enc_layer.connect(parameter_set.encoding_pars, net)
enc_layer.replicate_connections(net, clone_net, progress=True)

# enc_layer.connect(parameter_set.clone_enc_pars, clone_net)

# enc_layer.extract_synaptic_weights()

# enc_layer.create_with_clone(parameter_set.encoding_pars, net, clone_net, copy_pars=False)

perturbations = EncodingLayer(parameter_set.perturbation_pars)
perturbations.connect(parameter_set.perturbation_pars, clone_net)

# ######################################################################################################################
# Simulate
# ======================================================================================================================
if parameter_set.kernel_pars.transient_t:
	net.simulate(parameter_set.kernel_pars.transient_t)
	net.flush_records()
	clone_net.flush_records()

net.simulate(parameter_set.kernel_pars.sim_time + parameter_set.kernel_pars.resolution)
# ######################################################################################################################
# Extract and store data
# ======================================================================================================================
analysis_interval = [parameter_set.kernel_pars.transient_t,
                     parameter_set.kernel_pars.transient_t+parameter_set.kernel_pars.sim_time]
net.extract_population_activity(t_start=analysis_interval[0], t_stop=analysis_interval[1])
net.extract_network_activity()
clone_net.extract_population_activity(t_start=analysis_interval[0], t_stop=analysis_interval[1])
clone_net.extract_network_activity()

# ######################################################################################################################
# Analyse / plot data
# ======================================================================================================================
input_pops = ['E_input', 'I1_input', 'I2_inputs']
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

for idd, nam in enumerate(clone_net.population_names):
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

###################
time_window = parameter_set.kernel_pars.window
t_axis = np.arange(analysis_interval[0], analysis_interval[1],
                 parameter_set.net_pars.analog_device_pars[0]['interval'])

E_input_times = parameter_set.perturbation_pars.generator.model_pars[parameter_set.perturbation_pars.generator.labels.index(
			'E_input')]['spike_times']
if not empty(E_input_times):
	E_input_times = E_input_times[E_input_times>analysis_interval[0]]
I1_input_times = parameter_set.perturbation_pars.generator.model_pars[
	parameter_set.perturbation_pars.generator.labels.index(
			'I1_input')]['spike_times']
if not empty(I1_input_times):
	I1_input_times = I1_input_times[I1_input_times>analysis_interval[0]]
I2_input_times = parameter_set.perturbation_pars.generator.model_pars[
	parameter_set.perturbation_pars.generator.labels.index(
			'I2_input')]['spike_times']
if not empty(I2_input_times):
	I2_input_times = I2_input_times[I2_input_times>analysis_interval[0]]

# Look at a single neuron
net_activity = gather_analog_activity(parameter_set, net, t_start=analysis_interval[0], t_stop=analysis_interval[1])
clone_activity = gather_analog_activity(parameter_set, clone_net, t_start=analysis_interval[0], t_stop=analysis_interval[1])

fig0 = pl.figure()
ax01 = fig0.add_subplot(211)
ax01.plot(E_input_times, -60. * np.ones(len(E_input_times)), 'o')
ax01.plot(t_axis, net_activity[net.populations[0].name]['V_m'][0], 'r')
ax01.plot(t_axis, clone_activity[clone_net.populations[0].name]['V_m'][0], 'g')

ax02 = fig0.add_subplot(212)
ax02.plot(t_axis, clone_activity[clone_net.populations[0].name]['V_m'][0] - net_activity[net.populations[0].name][
	'V_m'][0], 'k', lw=2)
pl.show()

# compute difference between network and clone
diff_activity = {}
for n_pop in net_activity.keys():
	diff_activity.update({n_pop: {}})
	for k, v in net_activity[n_pop].items():
		diff_activity[n_pop][k] = clone_activity[n_pop+'_clone'][k] - v

# compute PSC/PSP properties
if not empty(E_input_times):
	PSC_kinetics(diff_activity, time_window, E_input_times, t_axis, response_type='E', plot=plot, display=display,
	             save=paths['figures']+paths['label'])
	PSP_kinetics(diff_activity, time_window, E_input_times, t_axis, response_type='E', plot=plot, display=display,
	             save=paths['figures']+paths['label'])

if not empty(I1_input_times):
	PSC_kinetics(diff_activity, time_window, I1_input_times, t_axis, response_type='I', plot=plot, display=display,
	             save=paths['figures']+paths['label'])
	PSP_kinetics(diff_activity, time_window, I1_input_times, t_axis, response_type='I', plot=plot, display=display,
	             save=paths['figures']+paths['label'])

if not empty(I2_input_times):
	PSC_kinetics(diff_activity, time_window, I2_input_times, t_axis, response_type='I', plot=plot, display=display,
	             save=paths['figures']+paths['label'])
	PSP_kinetics(diff_activity, time_window, I2_input_times, t_axis, response_type='I', plot=plot, display=display,
	             save=paths['figures']+paths['label'])

pl.show()

# ######################################################################################################################
# Save data
# ======================================================================================================================
if save:
	with open(paths['results'] + 'Results_' + parameter_set.label, 'w') as f:
		pickle.dump(results, f)
	parameter_set.save(paths['parameters'] + 'Parameters_' + parameter_set.label)