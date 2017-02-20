__author__ = 'duarte'
from modules.parameters import ParameterSet, ParameterSpace, extract_nestvalid_dict
from modules.input_architect import EncodingLayer, StimulusSet, InputSignalSet
from modules.net_architect import Network
from modules.io import set_storage_locations
from modules.signals import iterate_obj_list, empty
from modules.visualization import set_global_rcParams, InputPlots, extract_encoder_connectivity, TopologyPlots
from modules.analysis import analyse_state_matrix, get_state_rank, readout_train, readout_test
from modules.auxiliary import iterate_input_sequence
import cPickle as pickle
import matplotlib.pyplot as pl
import numpy as np
import time
import itertools
import nest

# ######################################################################################################################
# Experiment options
# ======================================================================================================================
plot = True
display = True
save = True
debug = False
online = True

# ######################################################################################################################
# Extract parameters from file and build global ParameterSet
# ======================================================================================================================
params_file = '../parameters/stimulus_driven.py'

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
results = dict(performance={}, dimensionality={})

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
net.merge_subpopulations([net.populations[0], net.populations[1]], name='EI')  # merge for EI case

# ######################################################################################################################
# Randomize initial variable values
# ======================================================================================================================
for idx, n in enumerate(list(iterate_obj_list(net.populations))):
	if hasattr(parameter_set.net_pars, "randomize_neuron_pars"):
		randomize = parameter_set.net_pars.randomize_neuron_pars[idx]
		for k, v in randomize.items():
			n.randomize_initial_states(k, randomization_function=v[0], **v[1])

########################################################################################################################
# Build Stimulus/Target Sets
# ======================================================================================================================
stim_set_startbuild = time.time()
stim = StimulusSet(parameter_set, unique_set=False)
stim.generate_datasets(parameter_set.stim_pars)

target = StimulusSet(parameter_set, unique_set=False)  # for identity task.
target.generate_datasets(parameter_set.stim_pars)

stim_set_buildtime = time.time()-stim_set_startbuild
print "- Elapsed Time: {0}".format(str(stim_set_buildtime))

########################################################################################################################
# Build Input Signal Set
# ======================================================================================================================
input_set_time = time.time()
inputs = InputSignalSet(parameter_set, stim, online=online)
inputs.generate_datasets(stim)

input_set_buildtime = time.time() - input_set_time
print "- Elapsed Time: {0}".format(str(input_set_buildtime))


