__author__ = 'duarte'
from modules.parameters import ParameterSpace, copy_dict, clean_array
from modules.visualization import *
from modules.io import set_project_paths
from defaults.paths import paths
from modules.signals import smooth
import matplotlib.pyplot as pl
import cPickle as pickle
import scipy.spatial as sp


"""
neuron_fI
- plot fI curves for the multiple heterogeneous neurons or single homogeneous
"""

# data parameters
project = 'heterogeneity_project'
data_type = 'heterogeneous' # 'dcinput' #
data_path = '/media/neuro/Data/Heterogeneity_OUT/SingleNeuronDC/'
data_label = 'SingleNeuron_DC_{0}_ab_multiTrial'.format(data_type)
results_path = data_path + data_label + '/Results/'

# set defaults and paths
set_project_paths(project)
set_global_rcParams(paths['local']['matplotlib_rc'])

# re-create ParameterSpace
pars_file = data_path + data_label + '_ParameterSpace.py'
pars = ParameterSpace(pars_file)

# print the full nested structure of the results dictionaries
# pars.print_stored_keys(results_path)

with open(results_path + '/Results_{0}_trial=1'.format(data_label), 'r') as fp:
	data = pickle.load(fp)

parameter_set = pars[0]

input_amplitudes = parameter_set.encoding_pars.generator.model_pars[0]['amplitude_values'][:-1]
input_times = parameter_set.encoding_pars.generator.model_pars[0]['amplitude_times']

results = {}
for k1 in data.keys():
	results.update({k1: {}})
	results[k1].update({
		'I_rh': pars.harvest(results_path, key_set=k1+'/I_rh', operation=np.mean)[1],
		'output_rate': pars.harvest(results_path, key_set=k1+'/output_rate')[1],
		'fI_slope': pars.harvest(results_path, key_set=k1+'/fI_slope')[1],
		'min_rate': pars.harvest(results_path, key_set=k1+'/min_rate')[1],
		'max_rate': pars.harvest(results_path, key_set=k1+'/max_rate')[1]})

# #### PLOT #####
keys = ['E', 'I1', 'I2']
colors = [(0., 0., 1.), (1., 0., 0.), (231. / 255., 101. / 255., 0.)]  # 'b', 'tomato', 'orangered']
ylims = [[0., 200.], [0., 500.], [0., 400.]]
sm = [10, 80, 40]

for idx, k1 in enumerate(keys):
	plot_props = {'xlabel': 'Rates', 'ylabel': 'Freq.', 'histtype': 'stepfilled', 'alpha': 0.8}
	fig = pl.figure(figsize=(8, 8))
	ax1 = fig.add_subplot(111)

	# create new axes on the right and on the top of the current axes
	# The first argument of the new_vertical(new_horizontal) method is
	# the height (width) of the axes to be created in inches.
	divider = make_axes_locatable(ax1)
	ax2 = divider.append_axes("bottom", 1.0, pad=0.5)  # , sharex=ax1)
	ax3 = divider.append_axes("top", 1.0, pad=0.5)  # , sharex=ax1)
	ax4 = divider.append_axes("left", 1.0, pad=0.5)  # , sharey=ax1)
	ax5 = divider.append_axes("right", 1.0, pad=0.5)  # , sharey=ax1)

	# ax2 = fig.add_subplot(232)
	plot_props.update({'xlabel': r'$\mathrm{I_{rh}}$'})
	plot_histogram(results[k1]['I_rh'].astype(float), nbins=50, norm=True, ax=ax2, color=colors[idx], display=True,
	               save=False, mark_mean=False, **plot_props)
	# ax2.hist(results[k1]['I_rh'], 100, c=colors[idx])

	# ax3 = fig.add_subplot(233)
	plot_props.update({'xlabel': r'$\mathrm{Slope [Hz/nA]}$'})
	plot_histogram(results[k1]['fI_slope'].astype(float), nbins=50, norm=True, ax=ax3, color=colors[idx],
	               display=True, save=False, mark_mean=False, **plot_props)
	# ax3.hist(data[k1]['fI_slope'], 100, c=colors[idx])

	# ax4 = fig.add_subplot(234)
	plot_props.update({'ylabel': r'$\nu_{\mathrm{max}} \mathrm{[Hz]}$', 'xlabel': 'Freq.', 'orientation': 'horizontal'})
	plot_histogram(results[k1]['max_rate'].astype(float), nbins=50, norm=True, ax=ax5, color=colors[idx],
	               display=True, save=False, mark_mean=False, **plot_props)
	# ax4.hist(data[k1]['max_rate'], 100, c=colors[idx])

	# ax5 = fig.add_subplot(235)
	plot_props.update({'ylabel': r'$\nu_{\mathrm{min}} \mathrm{[Hz]}$', 'xlabel': 'Freq.', 'orientation': 'horizontal'})
	plot_histogram(results[k1]['min_rate'].astype(float), nbins=10, norm=True, ax=ax4, color=colors[idx],
	               display=True, save=False, mark_mean=False, **plot_props)
	# ax5.hist(data[k1]['min_rate'], 100, c=colors[idx])

	rate = np.array(results[k1]['output_rate'])
	# ax1.plot(input_amplitudes, np.mean(rate, 0), '-', lw=3, c=colors[idx], alpha=1.)

	dist = []
	for r in rate:
		dist.append(sp.distance.euclidean(r, np.mean(rate, 0)))
	ax1.plot(input_amplitudes, smooth(rate[np.argmin(dist)], window_len=sm[idx]), '-', lw=3, c=colors[idx], alpha=1.)
	mean_idx = np.argmin(dist)
	print k1, mean_idx

	for nn in results[k1]['output_rate']:
		# rate_plot = np.array(nn[:100])
		ax1.plot(input_amplitudes, smooth(nn, window_len=sm[idx]), '-', lw=0.1, c=colors[idx], alpha=0.1)
	ax1.set_xlabel(r'$\mathrm{I [pA]}$')
	ax1.set_ylabel(r'$\nu_{\mathrm{i}} \mathrm{[spikes/s]}$')
	ax1.set_xlim((0., 800.))
	ax1.set_ylim(ylims[idx])

	pl.show()