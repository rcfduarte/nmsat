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
data_type = 'spikepatterninput' #'dcinput' #
#population_of_interest = 'E'  # results are provided for only one population (choose Global to get the totals)
#data_path = '/home/neuro/Desktop/MANUSCRIPTS/in_preparation/Encoding_Decoding/data/training_parameters/'
data_path = '/media/neuro/Data/EncodingDecoding_NEW/jitterStudy/'
data_label = 'ED_{0}_jitter'.format(data_type)  # read trial0 to extract data structure
results_path = data_path + data_label + '/Results/'

# set defaults and paths
set_project_paths(project)
set_global_rcParams(paths['local']['matplotlib_rc'])

# re-create ParameterSpace
pars_file = data_path + data_label + '_ParameterSpace.py'
pars = ParameterSpace(pars_file)

# print the full nested structure of the results dictionaries (to know where are the results of interest and how to
# harvest them)
pars.print_stored_keys(results_path)

# harvest the whole data set - returns a tuple with the data labels and the complete results dictionaries
# results = pars.harvest(data_path+data_label+'/Results/')

#################
# Analyse and plot, harvesting only the specified result (specifying the nested sequence of dictionary keys necessary
#  to reach the data - key_sets)
#################
# global plot properties
pl_props = copy_dict(pars.parameter_axes, {'xlabel': r'$' + pars.parameter_axes['ylabel'] + '$',
                                           'ylabel': r'$' + pars.parameter_axes['xlabel'] + '$',
                                           'xticklabels': pars.parameter_axes['yticklabels'],
                                           'yticklabels': pars.parameter_axes['xticklabels'][::4],
                                           'xticks': np.arange(0., len(pars.parameter_axes['yticks']), 1.),
                                           'yticks': np.arange(0., len(pars.parameter_axes['xticks']), 4.),})
analysis = {
	'title': 'Pinv classifier',
	'variable_names': [r'$Accuracy$', r'$Precision$',  r'$Recall$', r'$rawMSE$',
	                   r'$maxMSE$', r'$|W_{\mathrm{out}}|$'],
	'key_sets': ['performance/EI/V_m0/pinv_classifier/label/performance',
				 'performance/EI/V_m0/pinv_classifier/label/precision',
				 'performance/EI/V_m0/pinv_classifier/label/recall',
				 'performance/EI/V_m0/pinv_classifier/raw/MSE',
	             'performance/EI/V_m0/pinv_classifier/max/MSE',
	             'performance/EI/V_m0/pinv_classifier/norm_wOut',],}
##
fig = pl.figure()
fig.suptitle(analysis['title'])
n_subplots = len(analysis['variable_names'])
axes = []
arrays = []
remove_indices = [(1, 0), (3, 1), (5, 2), (7, 3), (9, 4), (15, 7)] # incorrect results
for idx, var in enumerate(analysis['variable_names']):
	globals()['ax{0}'.format(idx)] = fig.add_subplot(int(np.floor(np.sqrt(n_subplots))), int(np.ceil(np.sqrt(
		n_subplots))), idx+1)
	array = pars.harvest(results_path, key_set=analysis['key_sets'][idx])[1]
	for indice in remove_indices:
		array[indice] = np.nan

	axes.append(globals()['ax{0}'.format(idx)])
	arrays.append(array.astype(float))

plot_2d_parscans(arrays, axis=axes, fig_handle=fig, labels=analysis['variable_names'], **pl_props)

analysis2 = {
	'title': 'Ridge classifier',
	'variable_names': [r'$Accuracy$', r'$Precision$',  r'$Recall$', r'$rawMSE$',
	                   r'$maxMSE$', r'$|W_{\mathrm{out}}|$'],
	'key_sets': ['performance/EI/V_m0/ridge_classifier/label/performance',
				 'performance/EI/V_m0/ridge_classifier/label/precision',
				 'performance/EI/V_m0/ridge_classifier/label/recall',
				 'performance/EI/V_m0/ridge_classifier/raw/MSE',
	             'performance/EI/V_m0/ridge_classifier/max/MSE',
	             'performance/EI/V_m0/ridge_classifier/norm_wOut',],}
##
fig2 = pl.figure()
fig2.suptitle(analysis2['title'])
n_subplots = len(analysis2['variable_names'])
axes = []
arrays = []
for idx, var in enumerate(analysis2['variable_names']):
	globals()['ax{0}'.format(idx)] = fig2.add_subplot(int(np.floor(np.sqrt(n_subplots))), int(np.ceil(np.sqrt(
		n_subplots))), idx+1)
	array = pars.harvest(results_path, key_set=analysis2['key_sets'][idx])[1]

	axes.append(globals()['ax{0}'.format(idx)])
	arrays.append(array.astype(float))

plot_2d_parscans(arrays, axis=axes, fig_handle=fig2, labels=analysis2['variable_names'], **pl_props)

analysis3 = {
	'title': 'Dimensionality',
	'variable_names': [r'$Dimensionality$'],
	'key_sets': ['dimensionality/EI/V_m0'],}
##
fig3 = pl.figure()
fig3.suptitle(analysis3['title'])
n_subplots = len(analysis3['variable_names'])
axes = []
arrays = []
for idx, var in enumerate(analysis3['variable_names']):
	globals()['ax{0}'.format(idx)] = fig3.add_subplot(int(np.floor(np.sqrt(n_subplots))), int(np.ceil(np.sqrt(
		n_subplots))), idx+1)
	array = pars.harvest(results_path, key_set=analysis3['key_sets'][idx])[1]

	axes.append(globals()['ax{0}'.format(idx)])
	arrays.append(array.astype(float))

plot_2d_parscans(arrays, axis=axes, fig_handle=fig3, labels=analysis3['variable_names'], **pl_props)


pl.show()