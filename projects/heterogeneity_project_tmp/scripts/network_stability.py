__author__ = 'duarte'
import sys
from modules.parameters import ParameterSpace, ParameterSet, extract_nestvalid_dict
from modules.net_architect import Network
from modules.io import set_storage_locations
from modules.signals import iterate_obj_list, empty
from modules.input_architect import EncodingLayer
from modules.visualization import set_global_rcParams, TopologyPlots, progress_bar
from modules.analysis import analyse_state_divergence
import time
import numpy as np
import nest
import cPickle as pickle
import matplotlib.pyplot as pl
import itertools


plot = True
display = True
save = True
debug = True

###################################################################################
# Extract parameters from file and build global ParameterSet
# =================================================================================
params_file = '../parameters/perturbation_analysis.py'

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
# Connect Network
# ======================================================================================================================
net.connect_populations(parameter_set.connection_pars, progress=True)

# ######################################################################################################################
# Set-up Analysis
# ======================================================================================================================
net.connect_devices()
net.connect_decoders(parameter_set.decoding_pars)

###################################################
# Build and Connect copy network
#==================================================
clone_net = net.clone(parameter_set, devices=True, decoders=True)

# if debug:
# 	net.extract_synaptic_weights()
# 	net.extract_synaptic_delays()
# 	clone_net.extract_synaptic_weights()
# 	clone_net.extract_synaptic_delays()
# 	for idx, k in enumerate(net.synaptic_weights.keys()):
# 		print np.array_equal(np.array(net.synaptic_weights[k].todense()),
#  		                     np.array(clone_net.synaptic_weights[(k[0]+'_clone', k[1]+'_clone')].todense()))

# if plot and debug:
# 	fig_W = pl.figure()
# 	topology = vis.TopologyPlots(parameter_set.connection_pars, net, colors=['b', 'r'])
# 	#topology.print_network(depth=3)
# 	ax1 = pl.subplot2grid((6, 6), (0, 0), rowspan=4, colspan=4)
# 	ax2 = pl.subplot2grid((6, 6), (5, 0), rowspan=1, colspan=4)
# 	ax3 = pl.subplot2grid((6, 6), (0, 5), rowspan=4, colspan=1)
# 	ax4 = pl.subplot2grid((6, 6), (5, 5), rowspan=1, colspan=1)
# 	topology.plot_connectivity(parameter_set.connection_pars.synapse_types,
#  	                           ax=[ax1, ax3, ax2, ax4],
#  	                           display=display, save=save_path)
# if plot and debug:
# 	fig_W = pl.figure()
# 	topology = vis.TopologyPlots(parameter_set.connection_pars, list(iterate_obj_list(clone_net.populations)),
#  	                         colors=['b', 'r'])
# 	topology.print_network(depth=3)
# 	ax1 = pl.subplot2grid((6, 6), (0, 0), rowspan=4, colspan=4)
# 	ax2 = pl.subplot2grid((6, 6), (5, 0), rowspan=1, colspan=4)
# 	ax3 = pl.subplot2grid((6, 6), (0, 5), rowspan=4, colspan=1)
# 	ax4 = pl.subplot2grid((6, 6), (5, 5), rowspan=1, colspan=1)
# 	topology.plot_connectivity([('E_copy', 'E_copy')],
#  	                           ax=[ax1, ax3, ax2, ax4],
#  	                           display=display, save=save_path)
# ######################################################################################################################
# Build and connect input
# ======================================================================================================================
enc_layer = EncodingLayer(parameter_set.encoding_pars)
enc_layer.connect(parameter_set.encoding_pars, net)

for pop_idx, n_enc in enumerate(enc_layer.encoders):
	print "\nReconnecting {0} to {1}".format(n_enc.name, clone_net.population_names[pop_idx])
	for idx, src_id in enumerate(n_enc.gids):
		st = nest.GetStatus(nest.GetConnections(source=[src_id]))[0]
		tget_id = np.where(np.array(net.populations[pop_idx].gids) == st['target'])[0][0]
		new_tget_id = clone_net.populations[pop_idx].gids[tget_id]

		nest.Connect([st['source']], [new_tget_id], conn_spec={'rule': 'one_to_one'},
		             syn_spec={'model': st['synapse_model'], #'receptor_types': st['receptor_types'],
		                       'weight': st['weight'], 'delay': st['delay'], 'receptor_type': 1})
		progress_bar(float(idx)/float(len(n_enc.gids)))


perturb_population_idx = net.population_names.index(parameter_set.kernel_pars.perturb_population)
perturb_gids = np.random.permutation(clone_net.populations[perturb_population_idx].gids)[
               :parameter_set.kernel_pars.perturb_n]
perturbation_generator = nest.Create('spike_generator', 1,
                                     {'spike_times': [
	parameter_set.kernel_pars.perturbation_time + parameter_set.kernel_pars.transient_t],
                                      'spike_weights': [1.]})
parrot = nest.Create('parrot_neuron', 1)
nest.Connect(perturbation_generator, parrot)
nest.Connect(parrot, list(perturb_gids), syn_spec={'model': 'EE',
        'weight': parameter_set.kernel_pars.perturbation_spike_weight, 'receptor_type': 1})

######################################################################################
# Simulate (Initial Transient)
# ====================================================================================
if parameter_set.kernel_pars.transient_t:
	net.simulate(parameter_set.kernel_pars.transient_t)

	net.extract_population_activity(t_start=0., t_stop=parameter_set.kernel_pars.transient_t -
	                                                   parameter_set.kernel_pars.resolution)
	clone_net.extract_population_activity(t_start=0., t_stop=parameter_set.kernel_pars.transient_t -
	                                                   parameter_set.kernel_pars.resolution)
	net.extract_network_activity()
	clone_net.extract_network_activity()

	net.flush_records()
	clone_net.flush_records()

print "\nSimulation time = {0} ms".format(str(parameter_set.kernel_pars.sim_time))
net.simulate(parameter_set.kernel_pars.sim_time + parameter_set.kernel_pars.resolution)

######################################################################################
# Extract and store data
# ===================================================================================
net.extract_population_activity(t_start=parameter_set.kernel_pars.transient_t,
                                t_stop=parameter_set.kernel_pars.transient_t+parameter_set.kernel_pars.sim_time)
net.extract_network_activity()
net.flush_records(decoders=False)

clone_net.extract_population_activity(t_start=parameter_set.kernel_pars.transient_t,
                                      t_stop=parameter_set.kernel_pars.transient_t + parameter_set.kernel_pars.sim_time)
clone_net.extract_network_activity()
clone_net.flush_records(decoders=False)

#######################################################################################
# Extract response matrices
# =====================================================================================
analysis_interval = [parameter_set.kernel_pars.transient_t,
                     parameter_set.kernel_pars.sim_time + parameter_set.kernel_pars.transient_t]
# Extract merged responses
if not empty(net.merged_populations):
	net.merge_population_activity(start=analysis_interval[0], stop=analysis_interval[1], save=True)
if not empty(clone_net.merged_populations):
	clone_net.merge_population_activity(start=analysis_interval[0], stop=analysis_interval[1], save=True)

for ctr, n_pop in enumerate(list(itertools.chain(*[net.merged_populations, net.populations,
                                                   clone_net.merged_populations, clone_net.populations]))):
	if n_pop.decoding_layer is not None:
		n_pop.decoding_layer.extract_activity(start=analysis_interval[0], stop=analysis_interval[1], save=True)


#######################################################################################
# Analyse results
# =====================================================================================
results['perturbation'] = {}
results['perturbation'].update(analyse_state_divergence(parameter_set, net, clone_net, plot=plot, display=display,
                                                        save=paths['figures']+paths['label']))

# ######################################################################################################################
# Save data
# ======================================================================================================================
if save:
	with open(paths['results'] + 'Results_' + parameter_set.label, 'w') as f:
		pickle.dump(results, f)
	parameter_set.save(paths['parameters'] + 'Parameters_' + parameter_set.label)