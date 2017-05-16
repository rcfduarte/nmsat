__author__ = 'duarte'
import sys
from modules.parameters import ParameterSpace, ParameterSet, extract_nestvalid_dict
from modules.net_architect import Network
from modules.io import set_storage_locations
from modules.signals import iterate_obj_list, empty
from modules.input_architect import EncodingLayer
from modules.visualization import set_global_rcParams, TopologyPlots
from modules.analysis import characterize_population_activity
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
	net.extract_synaptic_delays()
	topology = TopologyPlots(parameter_set.connection_pars, net)
	topology.print_network(depth=3)

	# net_graph = topology.to_graph_object().to_directed()
	#
	# # plot degree distributions
	# sub_populations = [_ for _ in net.populations]
	# colors = ['r', 'b', 'Orange']
	# fig_degree = pl.figure()
	# ax1_deg = fig_degree.add_subplot(121)
	# ax2_deg = fig_degree.add_subplot(122)
	# for ctr, n_pop in enumerate(net.populations):
	# 	pop_graph = net_graph.nodes()[-(ctr+1)].to_directed()
	#
	# 	idx = parameter_set.net_pars.n_neurons.index(len(pop_graph.nodes()))
	# 	pop_name = parameter_set.net_pars.pop_names[idx]
	# 	pop_graph.name = pop_name
	# 	sub_populations[idx] = pop_graph
	#
	# 	in_degree = sorted(set(pop_graph.in_degree().values()))
	# 	out_degree = sorted(set(pop_graph.out_degree().values()))
	#
	# 	plot_histograms([ax1_deg, ax2_deg], [in_degree, out_degree], [100, 100], [])

	# plot weight matrices
	fig_W = pl.figure()
	ax1 = pl.subplot2grid((13, 13), (1, 1), rowspan=6, colspan=6)
	ax2 = pl.subplot2grid((13, 13), (1, 8), rowspan=6, colspan=2)
	ax3 = pl.subplot2grid((13, 13), (1, 11), rowspan=6, colspan=3)
	ax4 = pl.subplot2grid((13, 13), (8, 1), rowspan=2, colspan=6)
	ax5 = pl.subplot2grid((13, 13), (11, 1), rowspan=3, colspan=6)
	ax6 = pl.subplot2grid((13, 13), (8, 8), rowspan=2, colspan=2)
	ax7 = pl.subplot2grid((13, 13), (8, 11), rowspan=2, colspan=3)
	ax8 = pl.subplot2grid((13, 13), (11, 8), rowspan=3, colspan=2)
	ax9 = pl.subplot2grid((13, 13), (11, 11), rowspan=3, colspan=3)

	topology.plot_connectivity(parameter_set.connection_pars.synapse_types,
 	                           ax=[ax1, ax3, ax2, ax4, ax5, ax6, ax7, ax8, ax9],
                               display=display, save=paths['figures']+paths['label'])

	# plot weight distributions
	fig_wdist = pl.figure()
	axw1 = fig_wdist.add_subplot(331)
	axw2 = fig_wdist.add_subplot(332)
	axw3 = fig_wdist.add_subplot(333)
	axw4 = fig_wdist.add_subplot(334)
	axw5 = fig_wdist.add_subplot(335)
	axw6 = fig_wdist.add_subplot(336)
	axw7 = fig_wdist.add_subplot(337)
	axw8 = fig_wdist.add_subplot(338)
	axw9 = fig_wdist.add_subplot(339)
	topology.plot_weight_histograms(parameter_set.connection_pars.synapse_types,
 	                           ax=[axw1, axw2, axw3, axw4, axw5, axw6, axw7, axw8, axw9],
                                    display=display, save=paths['figures']+paths['label'])

	# plot delay matrices
	fig_d = pl.figure()
	axd1 = pl.subplot2grid((13, 13), (1, 1), rowspan=6, colspan=6)
	axd2 = pl.subplot2grid((13, 13), (1, 8), rowspan=6, colspan=2)
	axd3 = pl.subplot2grid((13, 13), (1, 11), rowspan=6, colspan=3)
	axd4 = pl.subplot2grid((13, 13), (8, 1), rowspan=2, colspan=6)
	axd5 = pl.subplot2grid((13, 13), (11, 1), rowspan=3, colspan=6)
	axd6 = pl.subplot2grid((13, 13), (8, 8), rowspan=2, colspan=2)
	axd7 = pl.subplot2grid((13, 13), (8, 11), rowspan=2, colspan=3)
	axd8 = pl.subplot2grid((13, 13), (11, 8), rowspan=3, colspan=2)
	axd9 = pl.subplot2grid((13, 13), (11, 11), rowspan=3, colspan=3)
	topology.plot_connectivity_delays(parameter_set.connection_pars.synapse_types,
 	                           ax=[axd1, axd3, axd2, axd4, axd5, axd6, axd7, axd8, axd9],
                            display=display, save=paths['figures']+paths['label'])

	# plot delay histograms
	fig_ddist = pl.figure()
	axd1 = fig_ddist.add_subplot(331)
	axd2 = fig_ddist.add_subplot(332)
	axd3 = fig_ddist.add_subplot(333)
	axd4 = fig_ddist.add_subplot(334)
	axd5 = fig_ddist.add_subplot(335)
	axd6 = fig_ddist.add_subplot(336)
	axd7 = fig_ddist.add_subplot(337)
	axd8 = fig_ddist.add_subplot(338)
	axd9 = fig_ddist.add_subplot(339)
	topology.plot_delay_histograms(parameter_set.connection_pars.synapse_types,
	                                ax=[axd1, axd2, axd3, axd4, axd5, axd6, axd7, axd8, axd9],
	                               display=display, save=paths['figures'] + paths['label'])
	pl.show()

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
parameter_set.analysis_pars.pop('label')
start_analysis = time.time()
results.update(characterize_population_activity(net, parameter_set, analysis_interval, epochs=None,
                                                color_map='jet', plot=plot,
                                                display=display, save=paths['figures']+paths['label'],
                                                color_subpop=True, analysis_pars=parameter_set.analysis_pars))
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

# ######################################################################################################################
# Save data
# ======================================================================================================================
if save:
	with open(paths['results'] + 'Results_' + parameter_set.label, 'w') as f:
		pickle.dump(results, f)
	parameter_set.save(paths['parameters'] + 'Parameters_' + parameter_set.label)

if display:
	pl.show()