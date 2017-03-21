__author__ = 'duarte'
from modules.parameters import ParameterSet, ParameterSpace, extract_nestvalid_dict
from modules.input_architect import EncodingLayer
from modules.net_architect import Network
from modules.io import set_storage_locations
from modules.signals import iterate_obj_list, SpikeList
from modules.visualization import set_global_rcParams, SpikePlots
from modules.analysis import characterize_population_activity
import cPickle as pickle
import numpy as np
import nest
import matplotlib.pyplot as pl
import pprint


# ######################################################################################################################
# Experiment options
# ======================================================================================================================
plot 	= True
display = True
save 	= True
debug 	= False

# ######################################################################################################################
# Extract parameters from file and build global ParameterSet
# ======================================================================================================================
params_file = '../parameters/two_pool_noisedriven_base_8.py'
#params_file = '../parameters/one_pool_noisedriven.py'

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
nu_x    = parameter_set.analysis_pars.nu_x
gamma   = parameter_set.analysis_pars.gamma
start_t = 0.
stop_t  = 500.

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

# ##########################################################################################
############################
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

net.simulate(parameter_set.kernel_pars.sim_time + 1.)
# ######################################################################################################################
# Extract and store data
# ======================================================================================================================
net.extract_population_activity()
net.extract_network_activity()
net.flush_records()


SpikePlots(net.populations[0].spiking_activity.id_slice(list(net.populations[0].gids[:1000])), start=start_t,
           stop=stop_t).dot_display(with_rate=True, display=False,
                                    save="{0}/raster_nu_x={1}_gamma={2}_E1.pdf".format(paths['figures'], nu_x, gamma))

SpikePlots(net.populations[2].spiking_activity.id_slice(list(net.populations[2].gids[:1000])), start=start_t,
           stop=stop_t).dot_display( with_rate=True, display=False,
                                     save="{0}/raster_nu_x={1}_gamma={2}_E2.pdf".format(paths['figures'], nu_x, gamma))
# ==============================

sl_1 = net.populations[0].spiking_activity.id_slice(list(net.populations[0].gids[:500]))
sl_i1 = net.populations[1].spiking_activity.id_slice(list(net.populations[1].gids[:200]))
sl_2 = net.populations[2].spiking_activity.id_slice(list(net.populations[2].gids[:500]))
sl_i2 = net.populations[3].spiking_activity.id_slice(list(net.populations[3].gids[:200]))

spikelist = []
id_list = []
missing = []
offset = 7500
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

fig = pl.figure(figsize=(40, 15))
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

fig.savefig("{0}/raster_nu_x={1}_gamma={2}_P1_P2.pdf".format(paths['figures'], nu_x, gamma))

parameter_set.save(paths['parameters'] + "/Parameters_plot.txt")
