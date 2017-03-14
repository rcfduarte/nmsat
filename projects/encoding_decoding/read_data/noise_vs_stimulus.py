from modules.parameters import ParameterSpace, copy_dict
from modules.signals import empty
from modules.visualization import set_global_rcParams, get_cmap, violin_plot, box_plot
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
data_type = 'dcinput'
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
population = 'Global'
metrics_of_interest = ['mean_rates', 'ffs', 'corrected_rates', 'lvs', 'ISI_distance_matrix', 'SPIKE_distance_matrix',
                       'SPIKE_sync_matrix', 'ents']

# harvest the noise data
k = 'ongoing/spiking_activity/{0}'.format(population)
noise = pars.harvest(results_path, key_set=k)[1]

# harvest stimulus data
k = 'evoked/spiking_activity/{0}'.format(population)
stim = pars.harvest(results_path, key_set=k)[1]


# analyse and plot distributions
fig = pl.figure()
fig.suptitle('All metrics')
positions = np.arange(len(metrics_of_interest))
cm = get_cmap(len(metrics_of_interest), 'Accent')

for idx, key in enumerate(metrics_of_interest):
	ax = pl.subplot2grid((1, len(metrics_of_interest) * 2), (0, positions[idx] * 2), rowspan=1, colspan=2)

	data = noise[key]
	ax = plot_distribution(data, pos=-0.2, cmap=cm, positions=positions, ax=ax)

	data = stim[key]
	ax = plot_distribution(data, pos=0.2, cmap=cm, positions=positions, ax=ax)

	ax.set_title(key)
	ax.set_xticks([-0.2, 0.2])
	ax.set_xticklabels(['Noise', 'Stimulus'])
	ax.set_xlim([-0.5, 0.5])
	ax.grid()


fig = pl.figure()
fig.suptitle('Summary indices')
positions = np.arange(len(metrics_of_interest))
cm = get_cmap(len(metrics_of_interest), 'Accent')

poisson_expectations = {
	'synchrony': {
		'ISI_distance': 0.5,
		'SPIKE_distance': 0.3,
		'SPIKE_sync_distance': 0.25},
	'regularity': {
		'lvs': 1.
	}}



# analysis_1 = {
# 	'fig_title': 'Performance',
# 	'ax_titles': [r'Ridge', r'Pseudoinverse'],
# 	'labels': [[r'Vm', r'spikes (reset)', r'spikes (no reset)', r'parrots'],
# 	           [r'Vm', r'spikes (reset)', r'spikes (no reset)', r'parrots']],
# 	'variable_names': [['vm_ridge', 'spikes0_ridge', 'spikes1_ridge', 'parrots_ridge'],
# 	                   ['vm_pinv', 'spikes0_pinv', 'spikes1_pinv', 'parrots_pinv']],
# 	'key_sets': [
# 		['spiking_activity/{0}/'.format(population),
# 		 'performance/EI/spikes0/ridge_classifier/label/performance',
# 		 'performance/EI/spikes2/ridge_classifier/label/performance',
# 		 'performance/parrots/spikes0/ridge_classifier/label/performance'],
# 		['performance/EI/V_m1/pinv_classifier/label/performance',
# 		 'performance/EI/spikes0/pinv_classifier/label/performance',
# 		 'performance/EI/spikes2/pinv_classifier/label/performance',
# 		 'performance/parrots/spikes0/pinv_classifier/label/performance'],],
# 	'plot_properties': []}
#
# keys = ['mean_rates', 'ffs', 'lvs', 'iR', 'ccs_pearson', 'd_vr', 'isi_5p', 'ents']
# labels = [r'$\nu_{C}$', r'$\mathrm{FF}$', r'$\mathrm{LV}_{\mathrm{ISI}}$', r'$\mathrm{IR}$', r'$\mathrm{CC}$',
#           r'$\mathrm{d}_{\mathrm{vR}}$', r'$\mathrm{ISI}_{\mathrm{5p}}$', r'$\mathrm{H}_{\mathrm{ISI}}$']
# positions = np.arange(len(keys))
# cm = get_cmap(len(keys), 'Accent')

