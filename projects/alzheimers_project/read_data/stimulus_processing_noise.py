__author__ = 'duarte'
from modules.parameters import ParameterSpace, copy_dict, clean_array
from modules.signals import empty
from modules.visualization import *
from modules.io import set_project_paths
from defaults.paths import paths
import matplotlib.pyplot as pl
from auxiliary_functions import harvest_results
from scipy.interpolate import griddata
from scipy.stats import sem

"""
stimulus_processing
- read data recorded from stimulus_processing experiments
"""

# data parameters
project = 'alzheimers_project'
data_path = '/media/neuro/Data/AD_Project/NEW/'
data_label = 'AD_IntervalLagStudy'
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

processed_data = []

# harvest specific results (specifying the list of keys necessary to reach the data)
analysis_1 = {
	'fig_title': 'Performance',
	'ax_titles': [r'Ridge', r'Pseudoinverse'],
	'labels': [[r'Vm', r'spikes', r'parrots'],
	           [r'Vm', r'spikes', r'parrots']],
	'variable_names': [['vm_ridge', 'spikes0_ridge', 'spikes0_parrots'],
	                   ['vm_pinv', 'spikes0_pinv', 'spikes0_parrots']],
	'key_sets': [
		['performance/EI/V_m1/ridge_classifier/label/performance',
		 'performance/EI/spikes0/ridge_classifier/label/performance',
		 'performance/parrots/spikes0/ridge_classifier/label/performance'],
		['performance/EI/V_m1/pinv_classifier/label/performance',
		 'performance/EI/spikes0/pinv_classifier/label/performance',
		 'performance/parrots/spikes0/pinv_classifier/label/performance'],],
	'plot_properties': [],
	'plot_colors': [['k', 'g', 'r'], []]}

processed_data.append(harvest_results(pars, analysis_1, results_path, plot=False, display=False,
                                      save=data_path+data_label))
########################################################################################################################
analysis_2 = {
	'fig_title': 'Raw MSE',
	'ax_titles': [r'Ridge', r'Pseudoinverse'],
	'labels': [[r'Vm', r'spikes', r'parrots'],
	           [r'Vm', r'spikes', r'parrots']],
	'variable_names': [['vm_ridge', 'spikes0_ridge', 'parrots_ridge'],
	                   ['vm_pinv', 'spikes0_pinv', 'spikes1_pinv', 'parrots_pinv']],
	'key_sets': [
		['performance/EI/V_m1/ridge_classifier/raw/MSE',
		 'performance/EI/spikes0/ridge_classifier/raw/MSE',
		 'performance/parrots/spikes0/ridge_classifier/raw/MSE'],
		['performance/EI/V_m1/pinv_classifier/raw/MSE',
		 'performance/EI/spikes0/pinv_classifier/raw/MSE',
		 'performance/parrots/spikes0/pinv_classifier/raw/MSE'],],
	'plot_properties': []}

processed_data.append(harvest_results(pars, analysis_2, results_path, plot=False, display=False,
                                      save=data_path+data_label))
########################################################################################################################
all_keys = list(itertools.chain(*[n[0].keys() for n in processed_data]))
all_results = [clean_array(n[0][k].astype(float)) for n in processed_data for k in n[0].keys()]
all_labels = [n[1] for n in processed_data]

########################################################################################################################
keys = ['performance/EI/V_m1/ridge_classifier/raw/MSE',
		'performance/EI/spikes0/ridge_classifier/raw/MSE',
		'performance/parrots/spikes0/ridge_classifier/raw/MSE']

for k in keys:
	k_idx = all_keys.index(k)
	array = all_results[k_idx].astype(float)

	fig1 = pl.figure()
	fig1.suptitle(k)

	# print all_labels[0][k]
	ax_main = pl.subplot2grid((4, 4), (0, 1), colspan=3, rowspan=3)
	ax_meanx = pl.subplot2grid((4, 4), (3, 1), colspan=3, rowspan=1)
	ax_meany = pl.subplot2grid((4, 4), (0, 0), colspan=1, rowspan=3)

	# interpolate missing data
	# try interpolation (to fill the missing data)
	x = np.array(pars.parameter_axes['xticks'])
	y = np.array(pars.parameter_axes['yticks'])
	z = array.astype(float)

	nx, ny = len(x), len(y)
	xi, yi = np.meshgrid(np.linspace(x.min(), x.max(), nx),
	                     np.linspace(y.min(), y.max(), ny))

	point_coordinates = [(k, v) for k in x for v in y]
	x_vals = np.array([xx[0] for xx in point_coordinates])
	y_vals = np.array([xx[1] for xx in point_coordinates])
	z_vals = z.flatten()

	points = np.array([(k, v) for k in x for v in y if not np.isnan(z[np.where(x == k)[0][0], np.where(y == v)[0][0]])])
	values = np.array([z[np.where(x == k)[0][0], np.where(y == v)[0][0]] for k in x for v in y if not np.isnan(z[np.where(
			                    x == k)[0][0], np.where(y == v)[0][0]])])

	grid_z = griddata(points, values, (xi, yi), method='nearest')

	# plot_2d_parscans([all_results[0]], [ax_main])
	ax_main.imshow(grid_z.T) # !!!
	ax_meanx.plot(pars.parameter_axes['yticks'], np.nanmean(array, 0))
	ax_meanx.fill_between(pars.parameter_axes['yticks'], np.nanmean(array, 0) - sem(array, 0,
	                nan_policy='omit'), np.nanmean(array, 0) + sem(array, 0, nan_policy='omit'),
	                      alpha=0.5)
	ax_meany.plot(np.nanmean(array, 1), pars.parameter_axes['xticks'])
	ax_meany.fill_betweenx(pars.parameter_axes['xticks'], np.nanmean(array, 1) - sem(array, 1,
	                nan_policy='omit'), np.nanmean(array, 1) + sem(array, 1, nan_policy='omit'),
	                       alpha=0.5)
	ax_main.set_xticks([])
	ax_main.set_yticks([])
	ax_meanx.set_xticks(pars.parameter_axes['yticks'])
	ax_meanx.set_xlabel(pars.parameter_axes['ylabel'])
	ax_meanx.set_xticklabels(pars.parameter_axes['yticklabels'])
	ax_meanx.set_xlim([np.min(pars.parameter_axes['yticks']), np.max(pars.parameter_axes['yticks'])])

	ax_meany.set_yticks(pars.parameter_axes['xticks'])
	ax_meany.set_ylabel(pars.parameter_axes['xlabel'])
	ax_meany.set_yticklabels(pars.parameter_axes['xticklabels'])
	ax_meany.set_ylim([np.min(pars.parameter_axes['xticks']), np.max(pars.parameter_axes['xticks'])])
	ax_meany.set_ylim(ax_meany.get_ylim()[::-1])

	pl.show()

########################################################################################################################

# k = 'performance/EI/V_m1/ridge_classifier/raw/MSE'
# k = 'performance/parrots/spikes0/ridge_classifier/raw/MSE'
# k = 'performance/EI/spikes0/ridge_classifier/label/performance'
k = 'performance/parrots/spikes0/ridge_classifier/label/performance'
array = all_results[all_keys.index(k)]


fig2 = pl.figure()
fig2.suptitle(k)
ax = fig2.add_subplot(111)

for xx in range(array.shape[0]):
	ax.plot(y, array[xx, :], 'o-', label=r'k={0}'.format(x[xx]))
ax.set_xticks(pars.parameter_axes['yticks'])
ax.set_xlabel(pars.parameter_axes['ylabel'])
ax.set_xticklabels(pars.parameter_axes['yticklabels'])
ax.set_xlim([np.min(pars.parameter_axes['yticks']), np.max(pars.parameter_axes['yticks'])])
pl.legend()
pl.show()
