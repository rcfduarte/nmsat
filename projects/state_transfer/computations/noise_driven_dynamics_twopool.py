__author__ = 'duarte'
import matplotlib
matplotlib.use('Agg')

from modules.parameters import ParameterSet, extract_nestvalid_dict
from modules.input_architect import EncodingLayer
from modules.net_architect import Network
from modules.io import set_storage_locations
from modules.signals import iterate_obj_list, SpikeList
from modules.visualization import set_global_rcParams, SpikePlots
from modules.analysis import characterize_population_activity, compute_analog_stats
import cPickle as pickle
import numpy as np
import nest
import pprint
import matplotlib.pyplot as pl


def run(parameter_set, plot=False, display=False, save=True):
	"""
	Analyse network dynamics when driven by Poisson input
	:param parameter_set: must be consistent with the computation, i.e. input must be poisson...
	:param plot: plot results - either show them or save to file
	:param display: show figures/reports
	:param save: save results
	:return results_dictionary:
	"""
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

	# ######################################################################################################################
	# Set kernel and simulation parameters
	# ======================================================================================================================
	print('\nRuning ParameterSet {0}'.format(parameter_set.label))
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

	# ######################################################################################################################
	# Simulate
	# ======================================================================================================================
	if parameter_set.kernel_pars.transient_t:
		net.simulate(parameter_set.kernel_pars.transient_t)
		net.flush_records()

	net.simulate(parameter_set.kernel_pars.sim_time + 1.)
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

	pool1 = net.merge_subpopulations(sub_populations=net.populations[:2], name='P1', merge_activity=True, store=False)
	pool2 = net.merge_subpopulations(sub_populations=net.populations[2:], name='P2', merge_activity=True, store=False)

	results_P1 = dict()
	results_P2 = dict()
	results_P1.update(characterize_population_activity(pool1, parameter_set, analysis_interval, epochs=None,
	                                                   color_map='coolwarm', plot=False, display=display,
	                                                   save=paths['figures'] + paths['label'], color_subpop=False,
	                                                   analysis_pars=parameter_set.analysis_pars))
	results_P2.update(characterize_population_activity(pool2, parameter_set, analysis_interval, epochs=None,
	                                                   color_map='coolwarm', plot=False, display=display,
	                                                   save=paths['figures'] + paths['label'], color_subpop=False,
	                                                   analysis_pars=parameter_set.analysis_pars))
	results_merged = results_P1.copy()

	for key in results_merged.keys():
		results_merged[key].update(**results_P2[key])

	results_merged['analog_activity']['E1'] = compute_analog_stats(net.populations[0], parameter_set, ["V_m", "g_in", "g_ex"],
	                                                               analysis_interval, False)
	results_merged['analog_activity']['I1'] = compute_analog_stats(net.populations[1], parameter_set, ["V_m", "g_in", "g_ex"],
	                                                               analysis_interval, False)
	results_merged['analog_activity']['E2'] = compute_analog_stats(net.populations[2], parameter_set, ["V_m", "g_in", "g_ex"],
	                                                               analysis_interval, False)
	results_merged['analog_activity']['I2'] = compute_analog_stats(net.populations[3], parameter_set, ["V_m", "g_in", "g_ex"],
	                                                               analysis_interval, False)

	# ######################################################################################################################
	# Save data
	# ======================================================================================================================
	if save:
		with open(paths['results'] + 'Results_' + parameter_set.label, 'w') as f:
			pickle.dump(results_merged, f)

		with open(paths['results'] + 'Results_' + parameter_set.label + '.txt', 'w') as f:
			pprint.PrettyPrinter(indent=2, stream=f).pprint(results_merged['spiking_activity'])
		parameter_set.save(paths['parameters'] + 'Parameters_' + parameter_set.label)

	# ######################################################################################################################
	# Let the plotting begin
	# ======================================================================================================================
	start_t = 1000.
	stop_t  = 2500.

	sl_1 = net.populations[0].spiking_activity.id_slice(list(net.populations[0].gids[:1000]))
	sl_i1 = net.populations[1].spiking_activity.id_slice(list(net.populations[1].gids[:250]))
	sl_2 = net.populations[2].spiking_activity.id_slice(list(net.populations[2].gids[:1000]))
	sl_i2 = net.populations[3].spiking_activity.id_slice(list(net.populations[3].gids[:250]))

	spikelist = []
	id_list = []
	missing = []
	offset = 7000
	def merge_spikes(pop, offs):
		for id_ in  pop.id_list:
			if len(pop.spiketrains[id_]) > 0:
				id_list.append(offs + id_)
				for spike in pop.spiketrains[id_].spike_times:
					spikelist.append((offs + id_, spike))
			else:
				missing.append(offs + id_)

	merge_spikes(sl_1, 0)
	merge_spikes(sl_i1, -offset)
	merge_spikes(sl_2, 0)
	merge_spikes(sl_i2, -offset)

	merged_spikelist = SpikeList(spikelist, id_list)
	merged_spikelist.complete(missing)

	sp = SpikePlots(merged_spikelist, start=start_t, stop=stop_t)

	fig = pl.figure(figsize=(30, 20))
	fig.suptitle("Spiking activity for two reservoirs")
	ax1 = pl.subplot2grid((80, 1), (10, 0), rowspan=25, colspan=1)
	ax0 = pl.subplot2grid((80, 1), (0, 0), rowspan=6, colspan=1, sharex=ax1)
	ax2 = pl.subplot2grid((80, 1), (40, 0), rowspan=25, colspan=1)
	ax3 = pl.subplot2grid((80, 1), (68, 0), rowspan=6, colspan=1, sharex=ax1)

	ax3.set(xlabel='Time [ms]', ylabel='Rate')
	ax1.set(ylabel='P1 Neurons')
	ax2.set(ylabel='P2 Neurons')

	sp.dot_display(gids_colors=[(sl_1.id_list, 'b'), (np.array([x - offset for x in sl_i1.id_list]), 'r')], ax=[ax1, ax0],
				   with_rate=True, display=False)

	sp.dot_display(gids_colors=[(sl_2.id_list, 'b'), (np.array([x - offset for x in sl_i2.id_list]), 'r')], ax=[ax2, ax3],
				   with_rate=True, display=False)

	fig.savefig("{0}/Raster_{1}_P1_P2.pdf".format(paths['figures'], parameter_set.label))
	with open("{0}/Analog_{1}_P1_P2.pkl".format(paths['results'], parameter_set.label), 'w') as f:
		pickle.dump(net.analog_activity, f)
	with open("{0}/Spiking_{1}_P1_P2.pkl".format(paths['results'], parameter_set.label), 'w') as f:
		pickle.dump(net.spiking_activity, f)
