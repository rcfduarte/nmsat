from modules.parameters import ParameterSpace, copy_dict
from modules.signals import empty, smooth
from modules.visualization import set_global_rcParams, get_cmap, violin_plot, box_plot, plot_2d_parscans
from modules.io import set_project_paths
from defaults.paths import paths
import matplotlib.pyplot as pl
from auxiliary_functions import plot_distribution
from matplotlib.mlab import griddata
from matplotlib import cm
from os import environ, system
import numpy as np
import sys

"""
noise_vs_stimulus
- read data on the transition from noise to stimulus-driven dynamics
"""

# data parameters
project = 'encoding_decoding'
data_type = 'spikeinput'
data_path = '/media/neuro/Data/EncodingDecoding_NEW/noise_vs_stimulus/'
data_label = 'ED_{0}_noise_vs_stimulus'.format(data_type)
results_path = data_path + data_label + '/Results/'

# set defaults and paths
set_project_paths(project)
set_global_rcParams(paths['local']['matplotlib_rc'])

# re-create ParameterSpace
pars_file = data_path + data_label + '_ParameterSpace.py'
pars = ParameterSpace(pars_file)

# print the full nested structure of the results dictionaries
# pars.print_stored_keys(results_path)  # takes a lot of time due to the size of the data


########################################################################################################################
population = 'E'
metrics_of_interest = ['mean_rates', 'ffs', 'corrected_rates', 'lvs', 'ents'] #'ISI_distance_matrix',
                       # 'SPIKE_distance_matrix',
                       #'SPIKE_sync_matrix', 'ents']

# harvest the noise data
k = 'ongoing/spiking_activity/{0}'.format(population)
noise = pars.harvest(results_path, key_set=k)[1]

# harvest stimulus data
k = 'evoked/spiking_activity/{0}'.format(population)
stim = pars.harvest(results_path, key_set=k)[1]

########################################################################################################################
# analyse and plot distributions
########################################################################################################################
# fig = pl.figure()
# fig.suptitle('All metrics')
# positions = np.arange(len(metrics_of_interest))
# cm = get_cmap(len(metrics_of_interest), 'Accent')
#
# for idx, key in enumerate(metrics_of_interest):
# 	ax = pl.subplot2grid((1, len(metrics_of_interest) * 2), (0, positions[idx] * 2), rowspan=1, colspan=2)
#
# 	data = noise[key]
# 	ax = plot_distribution(data, pos=-0.2, cmap=cm, positions=positions, idx=idx, ax=ax)
#
# 	data = stim[key]
# 	ax = plot_distribution(data, pos=0.2, cmap=cm, positions=positions, idx=idx, ax=ax)
#
# 	ax.set_title(key)
# 	ax.set_xticks([-0.2, 0.2])
# 	ax.set_xticklabels(['Noise', 'Stimulus'])
# 	ax.set_xlim([-0.5, 0.5])
# 	ax.grid()

# #######################################################################################################################
# plot distance matrices (memory!!)
# fig1 = pl.figure()
# ax11 = fig1.add_subplot(221)
# k = ['ISI_distance_matrix', 'SPIKE_distance_matrix', 'SPIKE_sync_matrix']
# for idx, key in enumerate(k):
# 	fig = pl.figure()
# 	fig.suptitle(key)
# 	ax1 = fig.add_subplot(221)
# 	ax2 = fig.add_subplot(222)
#
# 	results_arrays = [noise[key], stim[key]]
# 	axes = [ax1, ax2]
#
# 	plot_2d_parscans(results_arrays, axes, fig_handle=fig, labels=['Noise', 'Stimulus'], cmap='Accent', display=True)

########################################################################################################################
# fig = pl.figure()
# fig.suptitle('Synchrony')
# positions = np.arange(3)
# cm = get_cmap(3, 'Accent')
#
# k = ['ISI_distance_matrix', 'SPIKE_distance_matrix', 'SPIKE_sync_matrix']
# for idx, key in enumerate(k):
# 	ax = pl.subplot2grid((1, 3 * 2), (0, positions[idx] * 2), rowspan=1, colspan=2)
# 	data = np.triu(noise[key])
# 	values = data[np.nonzero(data)]
# 	ax = plot_distribution(values, pos=-0.2, cmap=cm, positions=positions, idx=idx, ax=ax)
#
# 	data = np.triu(stim[key])
# 	values = data[np.nonzero(data)]
# 	ax = plot_distribution(values, pos=0.2, cmap=cm, positions=positions, idx=idx, ax=ax)
#
# 	ax.set_title(key)
# 	ax.set_xticks([-0.2, 0.2])
# 	ax.set_xticklabels(['Noise', 'Stimulus'])
# 	ax.set_xlim([-0.5, 0.5])
# 	ax.grid()
# pl.show()

########################################################################################################################
fig3 = pl.figure()
fig3.suptitle('Summary Indices')
positions = np.arange(2)
cm = get_cmap(2, 'Accent')

poisson_expectations = {
	'synchrony': {
		'ISI_distance': 0.5,
		'SPIKE_distance': 0.3,
		#'SPIKE_sync': 0.25
	},
	'regularity': {
		'lvs': 1.
	}}
#k = ['ISI_distance_matrix', 'SPIKE_distance_matrix']#, 'SPIKE_sync_matrix']

labels = [r'$\mathrm{I^{reg}}$', r'$\mathrm{I^{sync}}$']
result = {}
for idx, metric_set in enumerate(poisson_expectations.keys()):
	ax = pl.subplot2grid((1, 2 * len(labels)), (0, positions[idx] * 2), rowspan=1, colspan=2)

	totals_nz = np.zeros((10000, 10000, 2))
	totals_st = np.zeros((10000, 10000, 2))
	ctr = 0
	for key, value in poisson_expectations[metric_set].items():
		if metric_set == 'synchrony':
			# noise data
			# data_nz = noise[key + '_matrix']
			data_nz = np.triu(noise[key + '_matrix'])
			index_data_nz = np.zeros_like(data_nz)

			for iid, val in np.ndenumerate(data_nz):
				if val:
					index_data_nz[iid] = np.abs(val - value)
			totals_nz[:, :, ctr] = index_data_nz

			# stim data
			# data_st = stim[key + '_matrix']
			data_st = np.triu(stim[key + '_matrix'])
			index_data_st = np.zeros_like(data_st)
			for iid, val in np.ndenumerate(data_st):
				if val:
					index_data_st[iid] = np.abs(val - value)
			totals_st[:, :, ctr] = index_data_st

			ctr += 1

		else:
			values_nz = np.abs(noise[key] - value)
			values_st = np.abs(stim[key] - value)

	if metric_set == 'synchrony':
		result.update({metric_set: {'noise': np.mean(totals_nz, 2)[np.nonzero(np.mean(totals_nz, 2))],
                              'stim': np.mean(totals_st, 2)[np.nonzero(np.mean(totals_st, 2))]}})
	else:
		result.update({metric_set: {'noise': values_nz, 'stim': values_st}})

	ax = plot_distribution(result[metric_set]['noise'], pos=-0.2, cmap=cm, positions=positions, idx=idx, ax=ax)
	ax = plot_distribution(result[metric_set]['stim'], pos=0.2, cmap=cm, positions=positions, idx=idx, ax=ax)
	ax.set_title(labels[idx])
	ax.set_xticks([-0.2, 0.2])
	ax.set_xticklabels(['Noise', 'Stimulus'])
	ax.set_xlim([-0.5, 0.5])
	ax.grid()
pl.show()


########################################################################################################################
# harvest transition data
# k = 'transition/spiking_activity/{0}'.format(population)
# transition = pars.harvest(results_path, key_set=k)[1]
#
#
# color_map = 'coolwarm'
# label = data_type
# analysis_interval = [pars[0].kernel_pars.transient_t - 2000., pars[0].kernel_pars.transient_t + 2000.,]
#
# fig5 = pl.figure()
# fig5.suptitle('{0} - Time-resolved regularity'.format(str(label)))
# stats = ['isi_5p_profile', 'cvs_profile', 'cvs_log_profile', 'lvs_profile', 'iR_profile', 'ents_profile']
# cm = get_cmap(len(stats), color_map)
#
# for idx, n in enumerate(stats):
# 	ax = fig5.add_subplot(len(stats), 1, idx + 1)
# 	data_mean = np.array([transition[n][i][0] for i in range(len(transition[n]))])
# 	data_std = np.array([transition[n][i][1] for i in range(len(transition[n]))])
# 	t_axis = np.linspace(analysis_interval[0], analysis_interval[1], len(data_mean))
#
# 	ax.plot(t_axis, data_mean, c=cm(idx), lw=2.5)
# 	ax.fill_between(t_axis, data_mean - data_std, data_mean + data_std, facecolor=cm(idx), alpha=0.2)
# 	ax.set_ylabel(n)
# 	ax.set_xlabel('Time [ms]')
# 	ax.set_xlim(analysis_interval)
#
# fig4 = pl.figure()
# fig4.suptitle('{0} - Time-resolved synchrony'.format(str(label)))
# ax1 = fig4.add_subplot(311)
# ax2 = fig4.add_subplot(312, sharex=ax1)
# ax3 = fig4.add_subplot(313, sharex=ax1)
# # if epochs is not None:
# # 	mark_epochs(ax1, epochs, color_map)
#
# x, y = transition['SPIKE_sync_profile'].get_plottable_data()
# ax1.plot(x, y, '-g', alpha=0.4)
# ax1.set_ylabel(r'$S_{\mathrm{SPIKE_{s}}}(t)$')
# ax1.plot(x, smooth(y, window_len=100, window='hamming'), '-g', lw=2.5)
# ax1.set_xlim([min(x), max(x)])
#
# x3, y3 = transition['ISI_profile'].get_plottable_data()
# ax2.plot(x3, y3, '-b', alpha=0.4)
# ax2.plot(x3, smooth(y3, window_len=100, window='hamming'), '-b', lw=2.5)
# ax2.set_ylabel(r'$d_{\mathrm{ISI}}(t)$')
# ax2.set_xlim([min(x3), max(x3)])
#
# x5, y5 = transition['SPIKE_profile'].get_plottable_data()
# ax3.plot(x5, y5, '-k', alpha=0.4)
# ax3.plot(x5, smooth(y5, window_len=100, window='hamming'), '-k', lw=2.5)
# ax3.set_ylabel(r'$d_{\mathrm{SPIKE}}(t)$')
# ax3.set_xlim([min(x5), max(x5)])
#
# pl.show()
