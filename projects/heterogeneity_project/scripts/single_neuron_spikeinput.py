__author__ = 'duarte'
from modules.parameters import ParameterSpace, ParameterSet, extract_nestvalid_dict
from modules.io import set_storage_locations
from modules.signals import iterate_obj_list, empty
from modules.net_architect import Network
from modules.input_architect import EncodingLayer
from modules.visualization import set_global_rcParams
from modules.analysis import single_neuron_responses
import numpy as np
import nest
import cPickle as pickle


plot = True
display = True
save = False

###################################################################################
# Extract parameters from file and build global ParameterSet
# =================================================================================
params_file = '../parameters/single_neuron_synaptic_noise.py'

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
enc_layer = EncodingLayer(parameter_set.encoding_pars)
enc_layer.connect(parameter_set.encoding_pars, net)

# ######################################################################################################################
# Set-up Analysis
# ======================================================================================================================
net.connect_devices()

# ######################################################################################################################
# Connect Network
# ======================================================================================================================
net.connect_populations(parameter_set.connection_pars)

# ######################################################################################################################
# Simulate
# ======================================================================================================================
if parameter_set.kernel_pars.transient_t:
	net.simulate(parameter_set.kernel_pars.transient_t)
	net.flush_records()

net.simulate(parameter_set.kernel_pars.sim_time + parameter_set.kernel_pars.resolution)

# ######################################################################################################################
# Extract and store data
# ======================================================================================================================
analysis_interval = [parameter_set.kernel_pars.transient_t,
                     parameter_set.kernel_pars.transient_t + parameter_set.kernel_pars.sim_time]
net.extract_population_activity(t_start=analysis_interval[0], t_stop=analysis_interval[1])
net.extract_network_activity()

# ######################################################################################################################
# Analyse / plot data
# ======================================================================================================================
input_pops = ['E_inputs', 'I1_inputs', 'I2_inputs']
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

		if not empty(net.analog_activity[idd]) and parameter_set.net_pars.record_analogs[idd] and 'G_syn_tot' in \
				parameter_set.net_pars.analog_device_pars[idd]['record_from']:
			g_total_idx = parameter_set.net_pars.analog_device_pars[idd]['record_from'].index('G_syn_tot')
			Cm = nest.GetStatus(net.populations[idd].gids)[0]['C_m']
			G_total = net.analog_activity[idd][g_total_idx].as_array()[0]
			tau_eff = G_total / Cm
			print nam, np.mean(tau_eff), np.std(tau_eff)
			results[nam]['tau_eff'] = (np.mean(tau_eff), np.std(tau_eff))

			vm_idx = parameter_set.net_pars.analog_device_pars[idd]['record_from'].index('V_m')
			vm = net.analog_activity[idd][vm_idx].as_array()[0]
			print nam, np.mean(vm), np.std(vm)
			results[nam]['mean_V'] = (np.mean(vm), np.std(vm))

# ######################################################################################################################
# Save data
# ======================================================================================================================
if save:
	with open(paths['results'] + 'Results_' + parameter_set.label, 'w') as f:
		pickle.dump(results, f)
	parameter_set.save(paths['parameters'] + 'Parameters_' + parameter_set.label)