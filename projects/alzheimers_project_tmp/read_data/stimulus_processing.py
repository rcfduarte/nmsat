from modules.parameters import ParameterSpace, copy_dict
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

display = False

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


processed_data = harvest_results(pars, analysis_1, results_path, display=display, save=data_path+data_label)

########################################################################################################################
analysis_2 = {
	'fig_title': 'Raw MSE',
	'ax_titles': [r'Ridge', r'Pseudoinverse'],
	'labels': [[r'Vm', r'spikes (reset)', r'spikes (no reset)', r'parrots'],
	           [r'Vm', r'spikes (reset)', r'spikes (no reset)', r'parrots']],
	'variable_names': [['vm_ridge', 'spikes0_ridge', 'spikes1_ridge', 'parrots_ridge'],
	                   ['vm_pinv', 'spikes0_pinv', 'spikes1_pinv', 'parrots_pinv']],
	'key_sets': [
		['performance/EI/V_m1/ridge_classifier/raw/MSE',
		 'performance/EI/spikes0/ridge_classifier/raw/MSE',
		 'performance/EI/spikes2/ridge_classifier/raw/MSE',
		 'performance/parrots/spikes0/ridge_classifier/raw/MSE'],
		['performance/EI/V_m1/pinv_classifier/raw/MSE',
		 'performance/EI/spikes0/pinv_classifier/raw/MSE',
		 'performance/EI/spikes2/pinv_classifier/raw/MSE',
		 'performance/parrots/spikes0/pinv_classifier/raw/MSE'],],
	'plot_properties': []}

harvest_results(pars, analysis_2, results_path, display=display, save=data_path+data_label)

########################################################################################################################
analysis_3 = {
	'fig_title': 'Readout stability',
	'ax_titles': [r'Ridge', r'Pseudoinverse'],
	'labels': [[r'Vm', r'spikes (reset)', r'spikes (no reset)', r'parrots'],
	           [r'Vm', r'spikes (reset)', r'spikes (no reset)', r'parrots']],
	'variable_names': [['vm_ridge', 'spikes0_ridge', 'spikes1_ridge', 'parrots_ridge'],
	                   ['vm_pinv', 'spikes0_pinv', 'spikes1_pinv', 'parrots_pinv']],
	'key_sets': [
		['performance/EI/V_m1/ridge_classifier/norm_wOut',
		 'performance/EI/spikes0/ridge_classifier/norm_wOut',
		 'performance/EI/spikes2/ridge_classifier/norm_wOut',
		 'performance/parrots/spikes0/ridge_classifier/norm_wOut'],
		['performance/EI/V_m1/pinv_classifier/norm_wOut',
		 'performance/EI/spikes0/pinv_classifier/norm_wOut',
		 'performance/EI/spikes2/pinv_classifier/norm_wOut',
		 'performance/parrots/spikes0/pinv_classifier/norm_wOut'],],
	'plot_properties': []}

harvest_results(pars, analysis_3, results_path, display=display, save=data_path+data_label)

########################################################################################################################
analysis_4 = {
	'fig_title': 'Raw MAE',
	'ax_titles': [r'Ridge', r'Pseudoinverse'],
	'labels': [[r'Vm', r'spikes (reset)', r'spikes (no reset)', r'parrots'],
	           [r'Vm', r'spikes (reset)', r'spikes (no reset)', r'parrots']],
	'variable_names': [['vm_ridge', 'spikes0_ridge', 'spikes1_ridge', 'parrots_ridge'],
	                   ['vm_pinv', 'spikes0_pinv', 'spikes1_pinv', 'parrots_pinv']],
	'key_sets': [
		['performance/EI/V_m1/ridge_classifier/raw/MAE',
		 'performance/EI/spikes0/ridge_classifier/raw/MAE',
		 'performance/EI/spikes2/ridge_classifier/raw/MAE',
		 'performance/parrots/spikes0/ridge_classifier/raw/MAE'],
		['performance/EI/V_m1/pinv_classifier/raw/MAE',
		 'performance/EI/spikes0/pinv_classifier/raw/MAE',
		 'performance/EI/spikes2/pinv_classifier/raw/MAE',
		 'performance/parrots/spikes0/pinv_classifier/raw/MAE'],],
	'plot_properties': []}

harvest_results(pars, analysis_4, results_path, display=display, save=data_path+data_label)

########################################################################################################################
analysis_6 = {
	'fig_title': 'max MAE',
	'ax_titles': [r'Ridge', r'Pseudoinverse'],
	'labels': [[r'Vm', r'spikes (reset)', r'spikes (no reset)', r'parrots'],
	           [r'Vm', r'spikes (reset)', r'spikes (no reset)', r'parrots']],
	'variable_names': [['vm_ridge', 'spikes0_ridge', 'spikes1_ridge', 'parrots_ridge'],
	                   ['vm_pinv', 'spikes0_pinv', 'spikes1_pinv', 'parrots_pinv']],
	'key_sets': [
		['performance/EI/V_m1/ridge_classifier/max/MAE',
		 'performance/EI/spikes0/ridge_classifier/max/MAE',
		 'performance/EI/spikes2/ridge_classifier/max/MAE',
		 'performance/parrots/spikes0/ridge_classifier/max/MAE'],
		['performance/EI/V_m1/pinv_classifier/max/MAE',
		 'performance/EI/spikes0/pinv_classifier/max/MAE',
		 'performance/EI/spikes2/pinv_classifier/max/MAE',
		 'performance/parrots/spikes0/pinv_classifier/max/MAE'],],
	'plot_properties': []}

harvest_results(pars, analysis_6, results_path, display=display, save=data_path+data_label)

########################################################################################################################
analysis_5 = {
	'fig_title': 'Dimensionality',
	'ax_titles': [''],
	'labels': [[r'Vm', r'spikes (reset)', r'spikes (no reset)', r'parrots']],
	'variable_names': [['vm_ridge', 'spikes0_ridge', 'spikes1_ridge', 'parrots_ridge']],
	'key_sets': [
		['dimensionality/EI/V_m1',
		 'dimensionality/EI/spikes0',
		 'dimensionality/EI/spikes2',
		 'dimensionality/parrots/spikes0']],
	'plot_properties': []}

harvest_results(pars, analysis_5, results_path, display=display, save=data_path+data_label)