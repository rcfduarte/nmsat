from modules.parameters import ParameterSpace, copy_dict
from modules.signals import empty
from modules.visualization import *
from modules.io import set_project_paths
from defaults.paths import paths
import matplotlib.pyplot as pl
from matplotlib.mlab import griddata
from matplotlib import cm
from os import environ, system
import sys

"""
stimulus_processing
- read data recorded from stimulus_processing experiments
"""

# data parameters
project = 'encoding_decoding'
data_type = 'dcinput' # 'spikepatterninput' #
#population_of_interest = 'E'  # results are provided for only one population (choose Global to get the totals)
data_path = '/home/neuro/Desktop/MANUSCRIPTS/in_preparation/Encoding_Decoding/data/training_parameters/'
# data_path = '/media/neuro/Data/EncodingDecoding_NEW/training_parameters/'
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
	'labels': ['Vm pinv'],
	'variable_names': ['vm_pinv'],
	'key_sets': [#'performance/EI/V_m0/ridge_classifier/label/performance',
	             'performance/EI/V_m0/pinv_classifier/label/performance',],
	'plot_properties': []}

# plot performance results surface
# fig1 = pl.figure()
# fig1.suptitle(analysis['title'])
# ax1 = fig1.add_subplot(111, projection='3d')
#
for var_id, variable_name in enumerate(analysis['variable_names']):
	globals()[variable_name] = pars.harvest(results_path, key_set=analysis['key_sets'][var_id])[1]

	x = pars.parameter_axes['xticks']
	y = pars.parameter_axes['yticks']
	X, Y = np.meshgrid(y, x)
	Z = globals()[variable_name].astype(float)

# 	ax1.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.3)
# 	# cset = ax1.contourf(X, Y, Z, zdir='z', cmap=cm.coolwarm)
# 	# cset = ax1.contourf(X, Y, Z, zdir='x', cmap=cm.coolwarm)
# 	# cset = ax1.contourf(X, Y, Z, zdir='y', cmap=cm.coolwarm)
# 	ax1.set_xlabel(pars.parameter_axes['ylabel'])
# 	ax1.set_ylabel(pars.parameter_axes['xlabel'])
# 	ax1.set_zlabel('Performance')
#
# pl.show()


# plot 2d arrays
fig2 = pl.figure()
fig2.suptitle(analysis['title'])
ax = Axes3D(fig2)
# try the interpolation


x = pars.parameter_axes['xticks']
y = pars.parameter_axes['yticks']
z = vm_pinv.astype(float)

point_coordinates = [(k, v) for k in x for v in y]
x_vals = np.array([xx[0] for xx in point_coordinates])
y_vals = np.array([xx[1] for xx in point_coordinates])
z_vals = z.flatten()


xi = np.linspace(min(x), max(x))
yi = np.linspace(min(y), max(y))
X, Y = np.meshgrid(xi, yi)
z = vm_pinv.astype(float)
Z = griddata(x_vals, y_vals, z_vals, xi, yi)

# ax.scatter3D(x_vals, y_vals, z_vals, c=z_vals, cmap=cm.jet)
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet', linewidth=1, alpha=0.8, antialiased=False)

ax.set_zlim3d(np.min(Z), np.max(Z))
fig2.colorbar(surf)
# m = cm.ScalarMappable(cmap=cm.jet)
# m.set_array(Z)
# fig2.colorbar(m)

ax.set_xlabel(pars.parameter_axes['xlabel'])
ax.set_ylabel(pars.parameter_axes['ylabel'])
ax.set_zlabel('Performance')

########################################################################################
### Plot dimensionality
analysis = {
	'title': 'Dimensionality',
	'labels': [r'$\lambda_{\mathrm{eff}}$'],
	'variable_names': ['dim'],
	'key_sets': ['dimensionality/EI/V_m0',],
	'plot_properties': []}
dim = pars.harvest(results_path, key_set=analysis['key_sets'][0])[1]
# plot 2d arrays
fig3 = pl.figure()
fig3.suptitle(analysis['title'])
ax = Axes3D(fig3)
# try the interpolation
x = pars.parameter_axes['xticks']
y = pars.parameter_axes['yticks']
z = dim.astype(float)

point_coordinates = [(k, v) for k in x for v in y]
x_vals = np.array([xx[0] for xx in point_coordinates])
y_vals = np.array([xx[1] for xx in point_coordinates])
z_vals = z.flatten()

xi = np.linspace(min(x), max(x))
yi = np.linspace(min(y), max(y))
X, Y = np.meshgrid(xi, yi)
Z = griddata(x_vals, y_vals, z_vals, xi, yi)

# ax.scatter3D(x_vals, y_vals, z_vals, c=z_vals, cmap=cm.jet)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet, linewidth=1, antialiased=True)

ax.set_xlabel(pars.parameter_axes['xlabel'])
ax.set_ylabel(pars.parameter_axes['ylabel'])
ax.set_zlabel(analysis['labels'][0])

pl.show()
