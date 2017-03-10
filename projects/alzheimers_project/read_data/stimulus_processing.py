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
data_label = 'AD_nStimStudy'
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
analysis_1 = {
	'fig_title': 'Performance',
	'ax_titles': [r'Ridge', r'Pseudoinverse'],
	'labels': [[r'Vm', r'spikes (reset)', r'spikes (no reset)', r'parrots'],
	           [r'Vm', r'spikes (reset)', r'spikes (no reset)', r'parrots']],
	'variable_names': [['vm_ridge', 'spikes0_ridge', 'spikes1_ridge', 'parrots_ridge'],
	                   ['vm_pinv', 'spikes0_pinv', 'spikes1_pinv', 'parrots_pinv']],
	'key_sets': [
		['performance/EI/V_m1/ridge_classifier/label/performance',
		 'performance/EI/spikes0/ridge_classifier/label/performance',
		 'performance/EI/spikes2/ridge_classifier/label/performance',
		 'performance/parrots/spikes0/ridge_classifier/label/performance'],
		['performance/EI/V_m1/pinv_classifier/label/performance',
		 'performance/EI/spikes0/pinv_classifier/label/performance',
		 'performance/EI/spikes2/pinv_classifier/label/performance',
		 'performance/parrots/spikes0/pinv_classifier/label/performance'],],

	'plot_properties': []}

fig1 = pl.figure()
fig1.suptitle(analysis_1['fig_title'])

for ax_n, ax_title in enumerate(analysis_1['ax_titles']):
	ax = fig1.add_subplot(2, 1, ax_n + 1)

	for idx_k, keys in enumerate(analysis['key_sets'][ax_n]):
		print "\nHarvesting {0}".format(keys)
		labels, result = pars.harvest(results_path, key_set=keys)
		ax.plot(pars.parameter_axes['xticks'], np.mean(result.astype(float), 1), 'o-', label=analysis['labels'][ax_n][
			idx_k])
		ax.errobar(pars.parameter_axes['xticks'], np.mean(result.astype(float), 1), yerr=np.std(result.astype(float),
		                                                                                        1))
	ax.set_xlabel(r'$' + pars.parameter_axes['xlabel'] + '$')
	ax.legend()
fig1.savefig(data_path + data_label + '_Results_{0}'.format(analysis_1['fig_title']))
pl.show()


########################################################################################################################
analysis = {
	'title': 'MSE [raw]',
	'labels': ['Vm ridge', 'spikes0 ridge', 'spikes1 ridge', 'parrots ridge',
	           'Vm pinv', 'spikes0 pinv', 'spikes1 pinv', 'parrots pinv'],
	'variable_names': ['vm_ridge', 'spikes0_ridge', 'spikes1_ridge', 'parrots_ridge',
	                   'vm_pinv', 'spikes0_pinv', 'spikes1_pinv', 'parrots_pinv'],
	'key_sets': ['performance/EI/V_m1/ridge_classifier/raw/MSE',
	             'performance/EI/spikes0/ridge_classifier/raw/MSE',
				 'performance/EI/spikes2/ridge_classifier/raw/MSE',
				 'performance/parrots/spikes0/ridge_classifier/raw/MSE',
	             'performance/EI/V_m1/pinv_classifier/raw/MSE',
	             'performance/EI/spikes0/pinv_classifier/raw/MSE',
	             'performance/EI/spikes2/pinv_classifier/raw/MSE',
	             'performance/parrots/spikes0/pinv_classifier/raw/MSE',],
	'plot_properties': []}

fig = pl.figure()
ax1 = fig.add_subplot(111)

for var_id, variable_name in enumerate(analysis['variable_names']):
	# globals()[variable_name] = pars.harvest(results_path, key_set=analysis['key_sets'][var_id])[1]

	ax1.plot(pars.parameter_axes['xticks'], np.mean(globals()[variable_name].astype(float), 1), label=analysis[
		'labels'][var_id])
	ax1.errorbar(pars.parameter_axes['xticks'], np.mean(globals()[variable_name].astype(float), 1), yerr=np.std(
		globals()[variable_name].astype(float), 1))

	ax1.plot()
	ax1.set_xlabel(r'$' + pars.parameter_axes['xlabel'] + '$')
	ax1.set_ylabel('MSE')
	ax1.legend()

pl.show()