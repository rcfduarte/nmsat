__author__ = 'duarte'
from modules.parameters import ParameterSpace, copy_dict
from modules.signals import empty
from modules.visualization import *
from modules.io import set_project_paths
from defaults.paths import paths
import matplotlib.pyplot as pl
from matplotlib import cm
import visvis
import scipy as sp
from scipy.interpolate import SmoothBivariateSpline, Rbf, griddata
from os import environ, system
import sys
import pickle

"""
stimulus_processing
- read data recorded from stimulus_processing experiments
"""

# data parameters
project = 'examples'
data_path = '../../../data/'
data_label = 'example4'
results_path = data_path + data_label + '/Results/'

# set defaults and paths
set_project_paths(project)
set_global_rcParams(paths['local']['matplotlib_rc'])

# re-create ParameterSpace
pars_file = data_path + data_label + '_ParameterSpace.py'
pars = ParameterSpace(pars_file)

# print the full nested structure of the results dictionaries (to harvest only specific results)
pars.print_stored_keys(results_path)

# Extract an example result,
with open(results_path + 'Results_' + data_label, 'r') as fp:
    single_result = pickle.load(fp)

# If you ran a parameter scan, you can extract all the values for the corresponding parameter combination,
# providing the sequence of nested keys needed to reach the result of interest
# classification_accuracy = pars.harvest(results_path, key_set='EI/V_m/class0/label/performance')