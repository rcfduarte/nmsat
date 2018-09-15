__author__ = 'duarte'
from modules.parameters import ParameterSet, ParameterSpace, extract_nestvalid_dict
from modules.input_architect import EncodingLayer
from modules.net_architect import Network
from modules.io import set_storage_locations
from modules.signals import iterate_obj_list
from modules.analysis import single_neuron_dcresponse
import cPickle as pickle
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as pl
import nest

# ######################################################################################################################
# Experiment options
# ======================================================================================================================
plot = True
display = True
save = True

# ######################################################################################################################
# Extract parameters from file and build global ParameterSet
# ======================================================================================================================
params_file = '../parameters/single_neuron_fI.py'

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
# Simulate
# ======================================================================================================================
if parameter_set.kernel_pars.transient_t:
    net.simulate(parameter_set.kernel_pars.transient_t)
    net.flush_records()

net.simulate(parameter_set.kernel_pars.sim_time + nest.GetKernelStatus()['resolution'])

# ######################################################################################################################
# Extract and store data
# ======================================================================================================================
net.extract_population_activity(t_start=parameter_set.kernel_pars.transient_t + nest.GetKernelStatus()['resolution'],
                                t_stop=parameter_set.kernel_pars.sim_time + parameter_set.kernel_pars.transient_t)
net.extract_network_activity()
net.flush_records()

# ######################################################################################################################
# Analyse / plot data
# ======================================================================================================================
analysis_interval = [parameter_set.kernel_pars.transient_t + nest.GetKernelStatus()['resolution'],
                     parameter_set.kernel_pars.sim_time + parameter_set.kernel_pars.transient_t]

for idd, nam in enumerate(net.population_names):
    results.update({nam: {}})
    results[nam] = single_neuron_dcresponse(net.populations[idd],
                                            parameter_set, start=analysis_interval[0],
                                            stop=analysis_interval[1], plot=plot,
                                            display=display, save=paths['figures'] + paths['label'])
    idx = np.min(np.where(results[nam]['output_rate']))

    print("Rate range for neuron {0} = [{1}, {2}] Hz".format(
        str(nam), str(np.min(results[nam]['output_rate'][results[nam]['output_rate'] > 0.])),
        str(np.max(results[nam]['output_rate'][results[nam]['output_rate'] > 0.]))))

    results[nam].update({'min_rate': np.min(results[nam]['output_rate'][results[nam]['output_rate'] > 0.]),
                         'max_rate': np.max(results[nam]['output_rate'][results[nam]['output_rate'] > 0.])})
    print("Rheobase Current for neuron {0} in [{1}, {2}]".format(
        str(nam), str(results[nam]['input_amplitudes'][idx - 1]), str(results[nam]['input_amplitudes'][idx])))

    x = np.array(results[nam]['input_amplitudes'])
    y = np.array(results[nam]['output_rate'])
    iddxs = np.where(y)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x[iddxs], y[iddxs])

    print("fI Slope for neuron {0} = {1} Hz/nA [linreg method]".format(nam, str(slope * 1000.)))

    results[nam].update({'fI_slope': slope * 1000., 'I_rh': [results[nam]['input_amplitudes'][idx - 1],
                                                             results[nam]['input_amplitudes'][idx]]})

# ######################################################################################################################
# Save data
# ======================================================================================================================
if save:
    with open(paths['results'] + 'Results_' + parameter_set.label, 'w') as f:
        pickle.dump(results, f)
    parameter_set.save(paths['parameters'] + 'Parameters_' + parameter_set.label)
