__author__ = 'duarte'
import sys
sys.path.append('/home/neuro/Desktop/CODE/NetworkSimulationTestbed/')
from Modules.parameters import *
from Modules.net_architect import *
from Modules.input_architect import *
from Modules.visualization import *
import numpy as np
from scipy import stats
import nest


plot = True
display = True
save = True
analysis_interval = None

###################################################################################
# Extract parameters from file and build global ParameterSet
# =================================================================================
params_file = '../Parameters_files/TimeScales0/single_neuron_dcinput_variability.py'

set_global_rcParams('../../Defaults/matplotlib_rc')

parameter_set = ParameterSet(set_params_dict(params_file), label='global')
parameter_set = parameter_set.clean(termination='pars')

if not isinstance(parameter_set, ParameterSet):
	if isinstance(parameter_set, basestring) or isinstance(parameter_set, dict):
		parameter_set = ParameterSet(parameter_set)
	else:
		raise TypeError("parameter_set must be ParameterSet, string with "
		                "full path to parameter file or dictionary")

if plot:
	import Modules.visualization as vis
	vis.set_global_rcParams(parameter_set.kernel_pars['mpl_path'])

np.random.seed(parameter_set.kernel_pars['np_seed'])

if analysis_interval is None:
	analysis_interval = [parameter_set.kernel_pars.transient_t,
	                     parameter_set.kernel_pars.sim_time]

########################################################
# Set kernel and simulation parameters
# =======================================================
print '\nRuning ParameterSet {0}'.format(parameter_set.label)
nest.ResetKernel()
nest.SetKernelStatus(extract_nestvalid_dict(parameter_set.kernel_pars.as_dict(),
                                            type='kernel'))

####################################################
# Build network
# ===================================================
net = Network(parameter_set.net_pars)
####################################################
# Randomize initial variable values
# ===================================================
for idx, n in enumerate(list(iterate_obj_list(net.populations))):
	if hasattr(parameter_set.net_pars, "randomize_neuron_pars"):
		randomize = parameter_set.net_pars.randomize_neuron_pars[idx]
		for k, v in randomize.items():
			n.randomize_initial_states(k, randomization_function=v[0], **v[1])

########################################################
# Build and connect input
# =======================================================
enc_layer = EncodingLayer(parameter_set.encoding_pars)
enc_layer.connect(parameter_set.encoding_pars, net)
########################################################
# Set-up Analysis
# =======================================================
net.connect_devices()
#######################################################
# Simulate
# ======================================================
if parameter_set.kernel_pars.transient_t:
	net.simulate(parameter_set.kernel_pars.transient_t)
	net.flush_records()

net.simulate(parameter_set.kernel_pars.sim_time + 1.)
#######################################################
# Extract and store data
# ======================================================
net.extract_population_activity()
net.extract_network_activity()
net.flush_records()
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

results = dict()
for idd, nam in enumerate(net.population_names):
	results.update({nam: {}})
	results[nam] = single_neuron_dcresponse(net.populations[idd],
	                                        parameter_set, start=analysis_interval[0],
	                                        stop=analysis_interval[1], plot=plot,
	                                        display=display, save=save_path)
	idx = np.min(np.where(results[nam]['output_rate']))

	print "Rate range for neuron {0} = [{1}, {2}] Hz".format(str(nam), str(np.min(results[nam]['output_rate'][
		                                                     results[nam]['output_rate']>0.])),
	                                                         str(np.max(results[nam]['output_rate'][
		                                                     results[nam]['output_rate']>0.])))
	results[nam].update({'min_rate': np.min(results[nam]['output_rate'][results[nam]['output_rate']>0.]),
	                     'max_rate': np.max(results[nam]['output_rate'][results[nam]['output_rate']>0.])})
	print "Rheobase Current for neuron {0} in [{1}, {2}]".format(str(nam), str(results[nam]['input_amplitudes'][
	                                                    idx - 1]), str(results[nam]['input_amplitudes'][idx]))

	x = np.array(results[nam]['input_amplitudes'])
	x_diff = np.diff(x)
	y = np.array(results[nam]['output_rate'])
	y_diff = np.diff(y)
	# iddxs0 = np.where(y_diff)
	# slope = y_diff[iddxs0] / x_diff[iddxs0]
	# # pl.plot(slope)
	# # pl.show()
	#
	# print "fI Slope for neuron {0} = {1} Hz/nA [diff method]".format(nam, str(np.mean(slope[slope > 0.])*1000.))

	iddxs = np.where(y)
	slope2, intercept, r_value, p_value, std_err = stats.linregress(x[iddxs], y[iddxs])
	print "fI Slope for neuron {0} = {1} Hz/nA [linreg method]".format(nam, str(slope2 * 1000.))

	results[nam].update({'fI_slope': slope2 * 1000., 'I_rh': [results[nam]['input_amplitudes'][idx - 1],
	                                                           results[nam]['input_amplitudes'][idx]]})

#######################################################
# Save data
# ======================================================
if save:
	with open(path + 'Results_' + parameter_set.label, 'w') as f:
		pickle.dump(results, f)
	parameter_set.save(path + 'Parameters_' + parameter_set.label)
