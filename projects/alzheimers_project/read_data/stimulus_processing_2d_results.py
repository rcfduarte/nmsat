__author__ = 'duarte'
from modules.parameters import ParameterSpace, copy_dict, clean_array
from modules.signals import empty
from modules.visualization import *
from modules.io import set_project_paths
from defaults.paths import paths
import matplotlib.pyplot as pl
# from matplotlib.mlab import griddata
from matplotlib import cm
import visvis
import scipy as sp
from scipy.interpolate import SmoothBivariateSpline, Rbf, griddata
from os import environ, system
import sys

"""
stimulus_processing
- read data recorded from stimulus_processing experiments
"""

# data parameters
project = 'alzheimers_project'
data_path = '/media/neuro/Data/AD_Project/NEW/'
data_label = 'AD_nStim_kEE'
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
# analysis = {
# 	'title': 'Performance',
# 	'labels': ['Vm pinv'],
# 	'variable_names': ['vm_pinv'],
# 	'key_sets': ['performance/EI/spikes0/ridge_classifier/label/performance',],
# 	'plot_properties': []}
#
# # plot performance results surface
# fig2 = pl.figure()
# fig2.suptitle(analysis['title'])
# ax = Axes3D(fig2)
#
# remove_indices = []#(1, 0), (3, 1), (5, 2), (7, 3), (9, 4), (15, 7)] # incorrect results
# for var_id, variable_name in enumerate(analysis['variable_names']):
# 	array = pars.harvest(results_path, key_set=analysis['key_sets'][var_id])[1]
#
# 	for indice in remove_indices:
# 		array[indice] = np.nan
#
# 	# try interpolation (to fill the missing data)
# 	x = pars.parameter_axes['xticks']
# 	y = pars.parameter_axes['yticks']
# 	z = array.astype(float)
#
# 	nx, ny = len(x), len(y)
# 	xi, yi = np.meshgrid(np.linspace(x.min(), x.max(), nx),
# 	                     np.linspace(y.min(), y.max(), ny))
#
# 	point_coordinates = [(k, v) for k in x for v in y]
# 	x_vals = np.array([xx[0] for xx in point_coordinates])
# 	y_vals = np.array([xx[1] for xx in point_coordinates])
# 	z_vals = z.flatten()
#
# 	points = np.array([(k, v) for k in x for v in y if not np.isnan(z[np.where(x==k)[0][0], np.where(y==v)[0][0]])])
# 	values = np.array([z[np.where(x==k)[0][0], np.where(y==v)[0][0]] for k in x for v in y if not np.isnan(z[np.where(x==k)[0][0], np.where(y==v)[0][0]])])
#
# 	grid_z = griddata(points, values, (xi, yi), method='cubic')
#
# 	ax.scatter3D(x_vals, y_vals, z_vals, c=z_vals, cmap=cm.coolwarm)
# 	surf = ax.plot_surface(xi, yi, grid_z, cmap='coolwarm', linewidth=1, alpha=0.6, rstride=1, cstride=1,
# 	                                              antialiased=True)
# 	fig2.colorbar(surf, shrink=0.5, aspect=5)
# 	ax.set_xlabel(pars.parameter_axes['xlabel'])
# 	ax.set_ylabel(pars.parameter_axes['ylabel'])
# 	ax.set_zlabel('Performance')
#
# pl.show()
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
	'title': 'Pinv classifier (spikes0 - reset)',
	'variable_names': [r'$Accuracy$', #r'$Precision$',  r'$Recall$', r'$rawMSE$',
	                   #r'$maxMSE$',
	                   r'$|W_{\mathrm{out}}|$'],
	'key_sets': ['performance/EI/spikes0/pinv_classifier/label/performance',
				 #'performance/EI/spikes0/pinv_classifier/label/precision',
				 #'performance/EI/spikes0/pinv_classifier/label/recall',
				 #'performance/EI/spikes0/pinv_classifier/raw/MSE',
	             #'performance/EI/spikes0/pinv_classifier/max/MSE',
	             'performance/EI/spikes0/pinv_classifier/norm_wOut',],}
##
fig = pl.figure()
fig.suptitle(analysis['title'])
n_subplots = len(analysis['variable_names'])
axes = []
arrays = []
remove_indices = []#(1, 0), (3, 1), (5, 2), (7, 3), (9, 4), (15, 7)] # incorrect results
for idx, var in enumerate(analysis['variable_names']):
	globals()['ax{0}'.format(idx)] = fig.add_subplot(int(np.floor(np.sqrt(n_subplots))), int(np.ceil(np.sqrt(
		n_subplots))), idx+1)
	array = pars.harvest(results_path, key_set=analysis['key_sets'][idx])[1]
	array = clean_array(array)
	for indice in remove_indices:
		array[indice] = np.nan

	axes.append(globals()['ax{0}'.format(idx)])
	arrays.append(array.astype(float))

plot_2d_parscans(arrays, axis=axes, fig_handle=fig, labels=analysis['variable_names'], **pl_props)

analysis2 = {
	'title': 'Ridge classifier (spikes0 - reset)',
	'variable_names': [r'$Accuracy$', #r'$Precision$',  r'$Recall$', r'$rawMSE$', r'$maxMSE$',
	                   r'$|W_{\mathrm{out}}|$'],
	'key_sets': ['performance/EI/spikes0/ridge_classifier/label/performance',
				 #'performance/EI/spikes0/ridge_classifier/label/precision',
				 #'performance/EI/spikes0/ridge_classifier/label/recall',
				 #'performance/EI/spikes0/ridge_classifier/raw/MSE',
	             #'performance/EI/spikes0/ridge_classifier/max/MSE',
	             'performance/EI/spikes0/ridge_classifier/norm_wOut',],}
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
	'title': 'Dimensionality (spikes0 - reset)',
	'variable_names': [r'$Dimensionality$'],
	'key_sets': ['dimensionality/EI/spikes0'],}
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