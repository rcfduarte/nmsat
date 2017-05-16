__author__ = 'duarte'
import sys
from modules.parameters import ParameterSpace, ParameterSet, extract_nestvalid_dict
from modules.net_architect import Network
from modules.io import set_storage_locations
from modules.signals import iterate_obj_list, empty
from modules.input_architect import EncodingLayer
from modules.visualization import set_global_rcParams, TopologyPlots
from modules.analysis import characterize_population_activity, cross_correlogram
import time
import numpy as np
import nest
import cPickle as pickle
import matplotlib.pyplot as pl


plot = True
display = True
save = True
debug = False

###################################################################################
# Extract parameters from file and build global ParameterSet
# =================================================================================
params_file = '../parameters/noise_driven_dynamics.py'

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
net.connect_populations(parameter_set.connection_pars, progress=True)

if plot and debug:
	net.extract_synaptic_weights()
	topology = TopologyPlots(parameter_set.connection_pars, net)
	topology.print_network(depth=3)

	fig_W = pl.figure()
	ax1 = pl.subplot2grid((6, 6), (0, 0), rowspan=4, colspan=4)
	ax2 = pl.subplot2grid((6, 6), (5, 0), rowspan=1, colspan=4)
	ax3 = pl.subplot2grid((6, 6), (0, 5), rowspan=4, colspan=1)
	ax4 = pl.subplot2grid((6, 6), (5, 5), rowspan=1, colspan=1)
	topology.plot_connectivity(parameter_set.connection_pars.synapse_types,
 	                           ax=[ax1, ax3, ax2, ax4], display=display, save=paths['figures']+paths['label'])
	fig_wdist = pl.figure()
	axw1 = fig_wdist.add_subplot(221)
	axw2 = fig_wdist.add_subplot(222)
	axw3 = fig_wdist.add_subplot(223)
	axw4 = fig_wdist.add_subplot(224)
	topology.plot_weight_histograms(parameter_set.connection_pars.synapse_types,
 	                           ax=[axw1, axw2, axw3, axw4], display=display, save=paths['figures']+paths['label'])
	net.extract_synaptic_delays()
	fig_d = pl.figure()
	axd1 = pl.subplot2grid((6, 6), (0, 0), rowspan=4, colspan=4)
	axd2 = pl.subplot2grid((6, 6), (5, 0), rowspan=1, colspan=4)
	axd3 = pl.subplot2grid((6, 6), (0, 5), rowspan=4, colspan=1)
	axd4 = pl.subplot2grid((6, 6), (5, 5), rowspan=1, colspan=1)
	topology.plot_connectivity_delays(parameter_set.connection_pars.synapse_types,
 	                           ax=[axd1, axd3, axd2, axd4], display=display, save=paths['figures']+paths['label'])
	fig_ddist = pl.figure()
	axd1 = fig_ddist.add_subplot(221)
	axd2 = fig_ddist.add_subplot(222)
	axd3 = fig_ddist.add_subplot(223)
	axd4 = fig_ddist.add_subplot(224)
	topology.plot_delay_histograms(parameter_set.connection_pars.synapse_types,
	                                ax=[axd1, axd2, axd3, axd4], display=display,
	                                save=paths['figures'] + paths['label'])
	#pl.show()

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
# net.extract_synaptic_weights()

conn = nest.GetConnections(list(np.unique(net.populations[0].gids)), list(np.unique(net.populations[0].gids)))
w_eff = np.zeros((len(np.unique(net.populations[0].gids)), len(np.unique(net.populations[0].gids))))
spks = net.populations[0].spiking_activity

tgets_gids = list(np.unique(net.populations[0].gids))
src_gids = list(np.unique(net.populations[0].gids))

for idx, n in enumerate(conn):
	source = n[0]
	target = n[1]

	spks_j = spks.spiketrains[n[0]].spikes_to_states_binary(0.1)
	spks_i = spks.spiketrains[n[1]].spikes_to_states_binary(0.1)

	print np.sum(spks_j), np.sum(spks_i)
	lag, corr = cross_correlogram(spks_i, spks_j, max_lag=1000, dt=0.1, plot=False)
	print np.mean(corr)

	w_eff[n[1] - min(tgets_gids), n[0] - min(src_gids)] = np.mean(corr)

#
#
# parameter_set.analysis_pars.pop('label')
# start_analysis = time.time()
# results.update(characterize_population_activity(net, parameter_set, analysis_interval, epochs=None,
#                                                 color_map='jet', plot=plot,
#                                                 display=display, save=paths['figures']+paths['label'],
#                                                 color_subpop=True, analysis_pars=parameter_set.analysis_pars))
# print "\nElapsed time (state characterization): {0}".format(str(time.time() - start_analysis))
#
#
# if 'mean_I_ex' in results['analog_activity']['E'].keys():
# 	inh = np.array(results['analog_activity']['E']['mean_I_in'])
# 	exc = np.array(results['analog_activity']['E']['mean_I_ex'])
# 	ei_ratios = np.abs(np.abs(inh) - np.abs(exc))
# 	ei_ratios_corrected = np.abs(np.abs(inh - np.mean(inh)) - np.abs(exc - np.mean(exc)))
# 	print "EI amplitude difference: {0} +- {1}".format(str(np.mean(ei_ratios)), str(np.std(ei_ratios)))
# 	print "EI amplitude difference (amplitude corrected): {0} +- {1}".format(str(np.mean(ei_ratios_corrected)),
# 	                                                                         str(np.std(ei_ratios_corrected)))
# 	results['analog_activity']['E']['IE_ratio'] = np.mean(ei_ratios)
# 	results['analog_activity']['E']['IE_ratio_corrected'] = np.mean(ei_ratios_corrected)
#
# # ######################################################################################################################
# # Save data
# # ======================================================================================================================
# if save:
# 	with open(paths['results'] + 'Results_' + parameter_set.label, 'w') as f:
# 		pickle.dump(results, f)
# 	parameter_set.save(paths['parameters'] + 'Parameters_' + parameter_set.label)
#
# if display:
# 	pl.show()