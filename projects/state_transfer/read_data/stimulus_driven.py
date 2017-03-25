from modules.parameters import ParameterSpace
from modules import signals
from modules.visualization import *
from modules.io import set_project_paths
from defaults.paths import paths
import matplotlib.pyplot as pl

"""
stimulus_processing
- read data recorded from stimulus_processing experiments
"""

# data parameters
project = 'state_transfer'
data_type = 'inh_poisson'
data_path = '/home/barni/code/fzj/nst/data/ST_onepool/'
data_label = 'ST_onepool_stimulus_driven_nstim=100'
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

analysis_spikes = {
	'title': 'Ridge classifier',
	'variable_names': [r'$Accuracy$', r'$Precision$',  r'$Recall$', r'$rawMSE$',
	                   r'$maxMSE$', r'$|W_{\mathrm{out}}|$'],
	'key_sets': ['performance/EI/spikes0/ridge_classifier/label/performance',
				 'performance/EI/spikes0/ridge_classifier/label/precision',
				 'performance/EI/spikes0/ridge_classifier/label/recall',
				 'performance/EI/spikes0/ridge_classifier/raw/MSE',
	             'performance/EI/spikes0/ridge_classifier/max/MSE',
	             'performance/EI/spikes0/ridge_classifier/norm_wOut',],}
analysis_Vm = {
	'title': 'Ridge classifier',
	'variable_names': [r'$Accuracy$', r'$Precision$',  r'$Recall$', r'$rawMSE$',
	                   r'$maxMSE$', r'$|W_{\mathrm{out}}|$'],
	'key_sets': ['performance/EI/V_m1/ridge_classifier/label/performance',
				 'performance/EI/V_m1/ridge_classifier/label/precision',
				 'performance/EI/V_m1/ridge_classifier/label/recall',
				 'performance/EI/V_m1/ridge_classifier/raw/MSE',
	             'performance/EI/V_m1/ridge_classifier/max/MSE',
	             'performance/EI/V_m1/ridge_classifier/norm_wOut',],}
##############################################
fig2 = pl.figure()
fig2.suptitle(analysis_spikes['title'])
n_subplots = len(analysis_spikes['variable_names'])
axes = []
arrays = []
for idx, var in enumerate(analysis_spikes['variable_names']):
	ax = fig2.add_subplot(int(np.floor(np.sqrt(n_subplots))), int(np.ceil(np.sqrt(n_subplots))), idx+1)
	val_spikes = pars.harvest(results_path, key_set=analysis_spikes['key_sets'][idx])[1]
	val_V_m = pars.harvest(results_path, key_set=analysis_Vm['key_sets'][idx])[1]

	ax.plot([1, 2], [val_spikes, val_V_m], 'o', c='b')
	ax.set_xlabel('')
	ax.set_ylabel('')
	ax.set_xticks([0, 1, 2, 3])
	ax.set_xticklabels(['', r'$spikes$', r'$V_m$', ''])
	ax.set_title(var)
fig2.savefig(figures_path + data_label + "_EI_readout_performance.pdf")
# pl.show(block=True)
