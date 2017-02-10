from modules.parameters import ParameterSpace, copy_dict
from modules.signals import empty
from modules.visualization import *
from modules.io import set_project_paths
from defaults.paths import paths
import matplotlib.pyplot as pl
from matplotlib import cm
from os import environ, system
import sys

"""
stimulus_processing
- read data recorded from stimulus_processing experiments
"""

# data parameters
project = 'encoding_decoding'
data_type = 'dcinput' # 'spiketemplate'
#population_of_interest = 'E'  # results are provided for only one population (choose Global to get the totals)
data_path = '/home/neuro/Desktop/MANUSCRIPTS/in_preparation/Encoding_Decoding/data/training_parameters/'
data_label = 'ED_{0}_training_parameters'.format(data_type)  # read trial0 to extract data structure
results_path = data_path + data_label + '/Results/'

# set defaults and paths
set_project_paths(project)
set_global_rcParams(paths['local']['matplotlib_rc'])

# re-create ParameterSpace
pars_file = data_path + data_label + '_ParameterSpace.py'
pars = ParameterSpace(pars_file)

# print the full nested structure of the results dictionaries (to harvest only specific results)
pars.print_stored_keys(results_path)

# harvest the whole data set - returns a tuple with the data labels and the complete results dictionaries
# results = pars.harvest(data_path+data_label+'/Results/')

# harvest specific results (specifying the list of keys necessary to reach the data)
analysis = {
	'title': 'Performance',
	'labels': ['Vm ridge', 'Vm pinv'],
	'variable_names': ['vm_ridge', 'vm_pinv'],
	'key_sets': ['performance/EI/V_m0/ridge_classifier/label/performance',
	             'performance/EI/V_m0/pinv_classifier/label/performance',],
	'plot_properties': []}

fig = pl.figure()
ax1 = fig.add_subplot(111, projection='3d')

for var_id, variable_name in enumerate(analysis['variable_names']):
	globals()[variable_name] = pars.harvest(results_path, key_set=analysis['key_sets'][var_id])[1]

	x = pars.parameter_axes['xticks']
	y = pars.parameter_axes['yticks']
	X, Y = np.meshgrid(y, x)
	Z = globals()[variable_name].astype(float)

	ax1.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.3)

	# cset = ax1.contourf(X, Y, Z, zdir='z', cmap=cm.coolwarm)
	# cset = ax1.contourf(X, Y, Z, zdir='x', cmap=cm.coolwarm)
	# cset = ax1.contourf(X, Y, Z, zdir='y', cmap=cm.coolwarm)

	ax1.set_xlabel(pars.parameter_axes['xlabel'])
	ax1.set_zlabel('Performance')

pl.show()

