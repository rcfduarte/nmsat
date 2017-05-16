__author__ = 'duarte'
from modules.parameters import ParameterSpace, copy_dict, clean_array
from modules.visualization import *
from modules.io import set_project_paths
from defaults.paths import paths
from modules.signals import smooth
import matplotlib.pyplot as pl
import cPickle as pickle
import scipy.spatial as sp


"""
neuron_RTF
- read and plot rate transfer functions
"""

# data parameters
project = 'heterogeneity_project'
data_type = 'homogeneous'
data_path = '/media/neuro/Data/Heterogeneity_NEW/population_RTF/'
data_label = 'HT_noisedrivendynamics_baseline'
results_path = data_path + data_label + '/Results/'

# set defaults and paths
set_project_paths(project)
set_global_rcParams(paths['local']['matplotlib_rc'])

# re-create ParameterSpace
pars_file = data_path + data_label + '_ParameterSpace.py'
pars = ParameterSpace(pars_file)

# print the full nested structure of the results dictionaries
pars.print_stored_keys(results_path)

# analyse and plot
fig = pl.figure()
fig.suptitle('Rate transfer function (Homogeneous)')
# ax = fig.add_subplot(111)
colors = ['blue', 'red', 'Orange']
neuron_types = ['E', 'I1', 'I2']
results_of_interest = ['rate', 'IE_ratio', 'mean_V', 'cv_isi', 'tau_eff']

d = pars.harvest(results_path, 'spiking_activity/mean_rate')