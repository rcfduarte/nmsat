__author__ = 'duarte'
import sys
sys.path.append('../../')
from Modules.parameters import *
from Modules.net_architect import *
from Modules.input_architect import *
from Modules.visualization import *
import numpy as np
import nest


plot = True
display = True
save = True
debug = False

###################################################################################
# Extract parameters from file and build global ParameterSet
# =================================================================================
#params_file = '../ParameterSets/noise_driven_dynamics.py'
params_file = '../ParameterSets/DC_noiseinput'
# params_file = '../ParameterSets/legenstein_maass_spike_template_classification.py'

# set plotting defaults
set_global_rcParams('../../Defaults/matplotlib_rc')

# create parameter set
# parameter_set = ParameterSet(set_params_dict(params_file), label='global')

# parameter_set = ParameterSpace(params_file)[0]
parameter_set = ParameterSet(params_file, label='global')
parameter_set = parameter_set.clean(termination='pars')

if not isinstance(parameter_set, ParameterSet):
	if isinstance(parameter_set, basestring) or isinstance(parameter_set, dict):
		parameter_set = ParameterSet(parameter_set)
	else:
		raise TypeError("parameter_set must be ParameterSet, string with full path to parameter file or dictionary")

if plot:
	import Modules.visualization as vis
	vis.set_global_rcParams(parameter_set.kernel_pars['mpl_path'])
paths = set_storage_locations(parameter_set, save)

np.random.seed(parameter_set.kernel_pars['np_seed'])

analysis_interval = [parameter_set.kernel_pars.transient_t,
                     parameter_set.kernel_pars.sim_time + parameter_set.kernel_pars.transient_t]

########################################################
# Set kernel and simulation parameters
# =======================================================
print '\nRuning ParameterSet {0}'.format(parameter_set.label)
nest.ResetKernel()
nest.SetKernelStatus(extract_nestvalid_dict(parameter_set.kernel_pars.as_dict(), param_type='kernel'))

####################################################
# Build network
# ===================================================
net = Network(parameter_set.net_pars)
# @barni: this is required here, because encoding layer references the 'EI' population
net.merge_subpopulations([net.populations[0], net.populations[1]], name='EI')
#######################################################
# Randomize initial variable values
# =====================================================
for idx, n in enumerate(list(iterate_obj_list(net.populations))):
	if hasattr(parameter_set.net_pars, "randomize_neuron_pars"):
		randomize = parameter_set.net_pars.randomize_neuron_pars[idx]
		for k, v in randomize.items():
			n.randomize_initial_states(k, randomization_function=v[0], **v[1])

########################################################
# Build and connect input
# ======================================================
enc_layer = EncodingLayer(parameter_set.encoding_pars)
enc_layer.connect(parameter_set.encoding_pars, net)

########################################################
# Set-up Analysis
# ======================================================
net.connect_devices()

#######################################################
# Connect Network
# =====================================================
net.connect_populations(parameter_set.connection_pars, progress=True)

if plot and debug:
	net.extract_synaptic_weights()
	topology = vis.TopologyPlots(parameter_set.connection_pars, net)
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

#######################################################
# Simulate
# ======================================================
if parameter_set.kernel_pars.transient_t:
	net.simulate(parameter_set.kernel_pars.transient_t)
	net.flush_records()

net.simulate(parameter_set.kernel_pars.sim_time)  # +.1 to acquire last step...

#######################################################
# Extract and store data
# ======================================================
net.extract_population_activity()
net.extract_network_activity()
# net.flush_records()

#######################################################
# Analyse / plot data
# ======================================================
if save:
	path = parameter_set.kernel_pars.data_path + \
	       parameter_set.kernel_pars.data_prefix + '/'
	save_path = path + parameter_set.label
	if not os.path.exists(path):
		os.mkdir(path)
else:
	save_path = False

net.populations[0].spiking_activity.raster_plot()

results = population_state(net, parameter_set, nPairs=500, time_bin=1., start=analysis_interval[0],
                           stop=analysis_interval[1], plot=plot, display=display, save=paths['figures']+paths['label'])

# ######################################################
# Save data
# ======================================================
if save:
	with open(path + 'Results_' + parameter_set.label, 'w') as f:
		pickle.dump(results, f)
	parameter_set.save(path + 'Parameters_' + parameter_set.label)