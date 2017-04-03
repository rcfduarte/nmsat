from modules.parameters import ParameterSpace, copy_dict
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
project = 'encoding_decoding'
data_type = 'spikepatterninput' # 'dcinput' #s
data_path = '/media/neuro/Data/EncodingDecoding_NEW/training_parameters/'
data_label = 'ED_{0}_training_parameters'.format(data_type)
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
	'key_sets': [#'performance/EI/V_m0/ridge_classifier/label/performance'],
	            'performance/EI/V_m0/pinv_classifier/label/performance',],
	'plot_properties': []}

# plot performance results surface
fig2 = pl.figure()
fig2.suptitle(analysis['title'])
ax = Axes3D(fig2)

if data_type == 'dcinput':
	remove_indices = [(1, 0), (3, 1), (5, 2), (7, 3), (9, 4), (15, 7)] # incorrect results
else:
	remove_indices = [(1, 0), (5, 2), (7, 3), (11, 5), (17, 8)]
for var_id, variable_name in enumerate(analysis['variable_names']):
	array = pars.harvest(results_path, key_set=analysis['key_sets'][var_id])[1]

	for indice in remove_indices:
		array[indice] = np.nan

	# try interpolation (to fill the missing data)
	x = pars.parameter_axes['xticks']
	y = pars.parameter_axes['yticks']
	z = array.astype(float)

	nx, ny = len(x), len(y)
	xi, yi = np.meshgrid(np.linspace(x.min(), x.max(), nx),
	                     np.linspace(y.min(), y.max(), ny))

	point_coordinates = [(k, v) for k in x for v in y]
	x_vals = np.array([xx[0] for xx in point_coordinates])
	y_vals = np.array([xx[1] for xx in point_coordinates])
	z_vals = z.flatten()

	points = np.array([(k, v) for k in x for v in y if not np.isnan(z[np.where(x==k)[0][0], np.where(y==v)[0][0]])])
	values = np.array([z[np.where(x==k)[0][0], np.where(y==v)[0][0]] for k in x for v in y if not np.isnan(z[
		                                                                                                       np.where(x==k)[0][0], np.where(y==v)[0][0]])])

	grid_z = griddata(points, values, (xi, yi), method='nearest')

	# ax.scatter3D(x_vals, y_vals, z_vals, c=z_vals, cmap=cm.winter)
	surf = ax.plot_surface(xi, yi, grid_z, cmap='winter', linewidth=1, alpha=0.8, rstride=1, cstride=1,
	                                              antialiased=True)

	# plot results @ T=10000 (see nStimStudy)
	ax.plot(x, np.ones_like(x)+1100, np.ones_like(x), 'ro-')

	fig2.colorbar(surf, shrink=0.2, aspect=5)

	ax.plot_wireframe(xi, yi, grid_z, color='k', linewidth=0.5, rstride=1, cstride=1)
	ax.set_xlabel(r'$\mathrm{N_{u}}$')
	ax.set_ylabel(r'$\mathrm{T}$')
	ax.set_zlabel(r'$\mathrm{Accuracy}$')

pl.show()


## Explorations to improve the surface plot
# f = visvis.gca()
# # m = visvis.grid(xi,yi,grid_z)
# f.daspect = 1, 1, 200 # z x 10
# # draped colors
# m = visvis.surf(xi,yi,grid_z)
# m.colormap = visvis.CM_JET


# from mayavi import mlab
# # create a figure with white background
# mlab.figure(bgcolor=(1, 1, 1))
# # create surface and passes it to variable surf
# surf=mlab.surf(grid_z, warp_scale=0.2)
# # import palette
# # surf.module_manager.scalar_lut_manager.lut.table = RGBA255
# # push updates to the figure
# mlab.draw()
# mlab.show()


########################################################################################
### Plot dimensionality
analysis = {
	'title': 'Dimensionality',
	'labels': [r'$\lambda_{\mathrm{eff}}$'],
	'variable_names': ['dim'],
	'key_sets': ['dimensionality/EI/V_m0',],
	'plot_properties': []}

# plot dim results surface
fig3 = pl.figure()
fig3.suptitle(analysis['title'])
ax = Axes3D(fig3)

if data_type == 'dcinput':
	remove_indices = [(1, 0), (3, 1), (5, 2), (7, 3), (9, 4), (15, 7)] # incorrect results
else:
	remove_indices = [(11, 5), (17, 8)]

for var_id, variable_name in enumerate(analysis['variable_names']):
	array = pars.harvest(results_path, key_set=analysis['key_sets'][var_id])[1]

	for indice in remove_indices:
		array[indice] = np.nan

	# try interpolation (to fill the missing data)
	x = pars.parameter_axes['xticks']
	y = pars.parameter_axes['yticks']
	y1 = y
	# y1 = np.insert(y, 0, 0) # add data point 0 - for plotting
	z = array.astype(float)
	z1 = z
	# z1 = np.zeros((50, 11))
	# z1[:, 1:] = z

	nx, ny = len(x), len(y)
	xi, yi = np.meshgrid(np.linspace(x.min(), x.max(), nx),
	                     np.linspace(y1.min(), y1.max(), ny))

	point_coordinates = [(k, v) for k in x for v in y1]
	x_vals = np.array([xx[0] for xx in point_coordinates])
	y_vals = np.array([xx[1] for xx in point_coordinates])
	z_vals = z1.flatten()

	points = np.array([(k, v) for k in x for v in y if not np.isnan(z1[np.where(x==k)[0][0], np.where(y==v)[0][0]])])
	values = np.array([z1[np.where(x==k)[0][0], np.where(y==v)[0][0]] for k in x for v in y if not np.isnan(z1[
		                                                                                                       np.where(x==k)[0][0], np.where(y==v)[0][0]])])

	grid_z = griddata(points, values, (xi, yi), method='cubic')

	# ax.scatter3D(x_vals, y_vals, z_vals, c=z_vals, cmap=cm.coolwarm)
	surf = ax.plot_surface(xi, yi, grid_z, cmap='coolwarm', linewidth=1, alpha=0.8, rstride=1, cstride=1,
	                                              antialiased=True)
	fig3.colorbar(surf, shrink=0.5, aspect=5)
	# surf = ax.plot_wireframe(xi, yi, grid_z, linewidth=1, rstride=1, cstride=1)#,
	#                        #antialiased=True)
	ax.set_xlabel(r'$\mathrm{N_{u}}$')
	ax.set_ylabel(r'$\mathrm{T}$')
	ax.set_zlabel(r'$\lambda_{\mathrm{eff}}$')

pl.show()
