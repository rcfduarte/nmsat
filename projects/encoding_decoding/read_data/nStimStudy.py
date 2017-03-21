__author__ = 'duarte'
from modules.parameters import ParameterSpace, copy_dict, clean_array
from modules.signals import empty
from modules.visualization import *
from modules.io import set_project_paths
from defaults.paths import paths
import matplotlib.pyplot as pl
from os import environ, system
from auxiliary_functions import harvest_results
import sys

"""
stimulus_processing
- read data recorded from stimulus_processing experiments
"""

# data parameters
project = 'encoding_decoding'
data_type = 'spikepatterninput' # 'dcinput'# # spikepatterninput
data_path = '/media/neuro/Data/EncodingDecoding_NEW/nStimStudy/'
data_label = 'ED_{0}_nStimStudy2'.format(data_type)
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

processed_data = []

# harvest specific results (specifying the list of keys necessary to reach the data)
analysis_1 = {
	'fig_title': 'Performance',
	'ax_titles': [r'Ridge', r'Pseudoinverse'],
	'labels': [[r'Vm', r'spikes'],
	           [r'Vm', r'spikes']],
	'variable_names': [['vm_ridge', 'spikes_ridge'],
	                   ['vm_pinv', 'spikes_pinv']],
	'key_sets': [
		['performance/EI/V_m0/ridge_classifier/label/performance',
		 'performance/EI/spikes1/ridge_classifier/label/performance',],
		['performance/EI/V_m0/pinv_classifier/label/performance',
		 'performance/EI/spikes1/pinv_classifier/label/performance',],],
	'plot_properties': [],
	'plot_colors': [['k', 'g', 'r'], []]}


processed_data.append(harvest_results(pars, analysis_1, results_path, display=True, save=data_path+data_label))

########################################################################################################################
analysis_2 = {
	'fig_title': 'MSE',
	'ax_titles': [r'Ridge', r'Pseudoinverse'],
	'labels': [[r'Vm', r'spikes'],
	           [r'Vm', r'spikes']],
	'variable_names': [['vm_ridge', 'spikes_ridge'],
	                   ['vm_pinv', 'spikes_pinv']],
	'key_sets': [
		['performance/EI/V_m0/ridge_classifier/raw/MSE',
		 'performance/EI/spikes1/ridge_classifier/raw/MSE',],
		['performance/EI/V_m0/pinv_classifier/raw/MSE',
		 'performance/EI/spikes1/pinv_classifier/raw/MSE',],],
	'plot_properties': []}

processed_data.append(harvest_results(pars, analysis_2, results_path, display=True, save=data_path+data_label))

########################################################################################################################
analysis_3 = {
	'fig_title': 'Readout stability',
	'ax_titles': [r'Ridge', r'Pseudoinverse'],
	'labels': [[r'Vm', r'spikes'],
	           [r'Vm', r'spikes']],
	'variable_names': [['vm_ridge', 'spikes_ridge'],
	                   ['vm_pinv', 'spikes_pinv']],
	'key_sets': [
		['performance/EI/V_m0/ridge_classifier/norm_wOut',
		 'performance/EI/spikes1/ridge_classifier/norm_wOut',],
		['performance/EI/V_m0/pinv_classifier/norm_wOut',
		 'performance/EI/spikes1/pinv_classifier/norm_wOut',],],
	'plot_properties': []}

processed_data.append(harvest_results(pars, analysis_3, results_path, display=True, save=data_path+data_label))

########################################################################################################################
analysis_4 = {
	'fig_title': 'Dimensionality',
	'ax_titles': [r'V_m', r'Spikes'],
	'labels': [[r'Vm'], [r'spikes']],
	'variable_names': [['vm'], ['spikes']],
	'key_sets': [['dimensionality/EI/V_m0'],
		 ['dimensionality/EI/spikes1']],
	'plot_properties': []}

processed_data.append(harvest_results(pars, analysis_4, results_path, display=True, save=data_path+data_label))

########################################################################################################################
all_keys = list(itertools.chain(*[n[0].keys() for n in processed_data]))
all_results = [clean_array(n[0][k]) for n in processed_data for k in n[0].keys()]
all_figures = [n[1] for n in processed_data]
all_axes = list(itertools.chain(*[n[2] for n in processed_data]))


fig = pl.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()
cm = get_cmap(2, 'RdBu')

keys = ['performance/EI/V_m0/ridge_classifier/label/performance',
        'performance/EI/spikes1/ridge_classifier/label/performance']
labels = [r'$\mathrm{V_{m}}$', r'$\mathrm{x}$']
for idd, k in enumerate(keys):
	idx = all_keys.index(k)
	result = all_results[idx]
	if len(result.shape) > 1:
		ax1.plot(pars.parameter_axes['xticks'], np.nanmean(all_results[idx], 1), '-', c=cm(idd), label=labels[idd])
	else:
		ax1.plot(pars.parameter_axes['xticks'], all_results[idx], '-', c=cm(idd), label=labels[idd])

keys = ['performance/EI/V_m0/ridge_classifier/norm_wOut',
        'performance/EI/spikes1/ridge_classifier/norm_wOut']
# labels = []
for idd, k in enumerate(keys):
	idx = all_keys.index(k)
	result = all_results[idx]
	if len(result.shape) > 1:
		ax1.plot(pars.parameter_axes['xticks'], np.nanmean(all_results[idx], 1), '-', c=cm(idd), label=labels[idd])
	else:
		ax1.plot(pars.parameter_axes['xticks'], all_results[idx], '-', c=cm(idd), label=labels[idd])

ax1.legend()
ax1.grid(False)
ax1.set_xlabel(r'$\mathrm{N_{u}}$')
ax1.set_ylabel(r'$\mathrm{Accuracy}$')
ax2.set_ylabel(r'$|\mathrm{W^{out}}|$')

pl.show()

##########################################
fig = pl.figure()
ax1 = fig.add_subplot(111)
cm = get_cmap(2, 'RdBu')

keys = ['dimensionality/EI/spikes1',
        'dimensionality/EI/V_m0']
labels = [r'$\mathrm{V_{m}}$', r'$\mathrm{x}$']
for idd, k in enumerate(keys):
	idx = all_keys.index(k)
	result = all_results[idx]
	if len(result.shape) > 1:
		ax1.plot(pars.parameter_axes['xticks'], np.nanmean(all_results[idx], 1), '-', c=cm(idd), label=labels[idd])
	else:
		ax1.plot(pars.parameter_axes['xticks'], all_results[idx], '-', c=cm(idd), label=labels[idd])

ax1.plot(np.insert(pars.parameter_axes['xticks'], 0, 0), np.insert(pars.parameter_axes['xticks'], 0, 0), 'r--')
ax1.legend()
ax1.grid(True)
ax1.set_xlabel(r'$\mathrm{N_{u}}$')
ax1.set_ylabel(r'$\lambda_{\mathrm{eff}}$')

pl.show()