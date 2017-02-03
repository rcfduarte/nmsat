from modules.parameters import ParameterSpace, copy_dict
from modules.signals import empty
from modules.visualization import *
from modules.io import set_project_paths
from defaults.paths import paths
import matplotlib.pyplot as pl
from os import environ, system
import sys

"""
stimulus_processing
- read data recorded from stimulus_processing experiments
"""

# data parameters
project = 'alzheimers_project'
data_path = '/media/neuro/Data/AD_Project/NEW/'
data_label = 'AD_StimulusDriven_kEE'
results_path = data_path + data_label + '/Results/'

# set defaults and paths
set_project_paths(project)
set_global_rcParams(paths['local']['matplotlib_rc'])

# re-create ParameterSpace
pars_file = data_path + data_label + '_ParameterSpace.py'
pars = ParameterSpace(pars_file)

# print the full nested structure of the results dictionaries
pars.print_stored_keys(results_path)

# harvest the whole data set - returns a tuple with the data labels and the complete results dictionaries
# results = pars.harvest(data_path+data_label+'/Results/')

# harvest specific results (specifying the list of keys necessary to reach the data)
analysis = {
	'title': 'Performance [max]',
	'labels': ['Vm ridge', 'spikes ridge', 'Vm pinv', 'spikes pinv'],
	'variable_names': ['vm_ridge', 'spikes_ridge', 'vm_pinv', 'spikes_pinv'],
	'key_sets': ['performance/EI/V_m/ridge_classifier/max/performance',
	             'performance/EI/spikes/ridge_classifier/max/performance',
	             'performance/EI/V_m/pinv_classifier/max/performance',
	             'performance/EI/spikes/pinv_classifier/max/performance'],
	'plot_properties': []}

fig = pl.figure()
ax1 = fig.add_subplot(111)

for var_id, variable_name in enumerate(analysis['labels']):
	globals()[variable_name] = pars.harvest(results_path, key_set=analysis['key_sets'][var_id])[1]

	ax1.plot(pars.parameter_axes['xticks'], np.mean(globals()[variable_name].astype(float), 1), label=analysis[
		'labels'][var_id])
	ax1.errorbar(pars.parameter_axes['xticks'], np.mean(globals()[variable_name].astype(float), 1), yerr=np.std(
		globals()[variable_name].astype(float), 1))
	ax1.set_xlabel(pars.parameter_axes['xlabel'])
	ax1.set_ylabel('Performance')

pl.show()

