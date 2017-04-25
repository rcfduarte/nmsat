from IPython.core.pylabtools import figsize

from modules.parameters import ParameterSpace
from modules import signals
from modules.visualization import *
from modules.io import set_project_paths
from defaults.paths import paths
import matplotlib.pyplot as pl
import matplotlib.cm as cm
import re

"""
stimulus_processing
- read data recorded from stimulus_processing experiments
"""

# data parameters
project = 'state_transfer'
data_type = 'inh_poisson'
data_path = '../../../data/'
data_label = 'ST_twopool_stimulusdriven_ud_bgnoise_025'
results_path = data_path + data_label + '/Results/'
figures_path = data_path + data_label + '/Figures/'

# set defaults and paths
set_project_paths(project)
set_global_rcParams(paths['local']['matplotlib_rc'])

# re-create ParameterSpace
pars_file = data_path + data_label + '_ParameterSpace.py'
pars = ParameterSpace(pars_file)

# print the full nested structure of the results dictionaries (to harvest only specific results)
# pars.print_stored_keys(results_path)

# harvest the whole data set - returns a tuple with the data labels and the complete results dictionaries
# results = pars.harvest(data_path+data_label+'/Results/')
# print results

for pop_name in ['E1I1', 'E2I2']:
	analysis_spikes = {
		'title': 'Ridge classifier population {0}'.format(pop_name),
		'variable_names': [r'$Accuracy$', r'$Precision$',  r'$Recall$', r'$rawMSE$',
						   r'$maxMSE$', r'$|W_{\mathrm{out}}|$'],
		'key_sets': ['performance/{0}/spikes0/ridge_classifier/label/performance'.format(pop_name),
					 'performance/{0}/spikes0/ridge_classifier/label/precision'.format(pop_name),
					 'performance/{0}/spikes0/ridge_classifier/label/recall'.format(pop_name),
					 'performance/{0}/spikes0/ridge_classifier/raw/MSE'.format(pop_name),
					 'performance/{0}/spikes0/ridge_classifier/max/MSE'.format(pop_name),
					 'performance/{0}/spikes0/ridge_classifier/norm_wOut'.format(pop_name),],}
	analysis_Vm = {
		'title': 'Ridge classifier {0}'.format(pop_name),
		'variable_names': [r'$Accuracy$', r'$Precision$',  r'$Recall$', r'$rawMSE$',
						   r'$maxMSE$', r'$|W_{\mathrm{out}}|$'],
		'key_sets': ['performance/{0}/V_m1/ridge_classifier/label/performance'.format(pop_name),
					 'performance/{0}/V_m1/ridge_classifier/label/precision'.format(pop_name),
					 'performance/{0}/V_m1/ridge_classifier/label/recall'.format(pop_name),
					 'performance/{0}/V_m1/ridge_classifier/raw/MSE'.format(pop_name),
					 'performance/{0}/V_m1/ridge_classifier/max/MSE'.format(pop_name),
					 'performance/{0}/V_m1/ridge_classifier/norm_wOut'.format(pop_name),],}
	##############################################
	fig2 = pl.figure(figsize=(10, 7))
	fig2.suptitle(analysis_spikes['title'])
	n_subplots = len(analysis_spikes['variable_names'])

	axes = []
	arrays = []
	n_diff_stim = 8
	x = np.arange(n_diff_stim)
	ys = [i + x + (i * x) ** 2 for i in range(n_diff_stim)]
	colors = cm.rainbow(np.linspace(0, 1, len(ys)))
	lines = []

	for idx, var in enumerate(analysis_spikes['variable_names']):
		ax = fig2.add_subplot(int(np.floor(np.sqrt(n_subplots))), int(np.ceil(np.sqrt(n_subplots))), idx+1)
		labels, val_spikes = pars.harvest(results_path, key_set=analysis_spikes['key_sets'][idx])
		val_V_m = pars.harvest(results_path, key_set=analysis_Vm['key_sets'][idx])[1]

		for i in range(len(val_spikes)):
			line, = ax.plot([1, 2], [val_spikes[i], val_V_m[i]], 'o', c=colors[i])
			if idx == 0:
				lines.append(line)

		ax.set_xlabel('')
		ax.set_ylabel('')
		ax.set_xticks([0, 1, 2, 3])
		ax.set_xticklabels(['', r'$spikes$', r'$V_m$', ''])
		ax.set_title(var)

	fig2.tight_layout()
	fig2.legend(lines, [re.search('n_stim=\d+', l).group(0) for l in labels[:n_diff_stim]], loc='upper right',
				bbox_to_anchor=(1.3, 0.8))
	pl.subplots_adjust(top=0.85)
	fig2.savefig(figures_path + data_label + "_{0}_readout_performance.pdf".format(pop_name))
	# pl.show()
	# exit(0)