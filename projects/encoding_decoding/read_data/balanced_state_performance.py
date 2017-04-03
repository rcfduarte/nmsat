__author__ = 'duarte'
from modules.parameters import ParameterSpace, copy_dict, clean_array
from modules.signals import empty
from modules.visualization import *
from modules.io import set_project_paths
from defaults.paths import paths
import matplotlib.pyplot as pl
from os import environ, system
from auxiliary_functions import harvest_results
import sys
from scipy.stats import sem
from scipy.optimize import curve_fit
import pickle

"""
balanced_state_performance
- relate population state with classification performance / MSE
"""

# data parameters
project = 'encoding_decoding'
data_type = 'spikeinput' # 'dcinput'
data_path = '/media/neuro/Data/EncodingDecoding_NEW/activeState/'
data_label = 'ED_{0}_activeState'.format(data_type)
results_path = data_path + data_label + '/Results/'

# set defaults and paths
set_project_paths(project)
set_global_rcParams(paths['local']['matplotlib_rc'])

# re-create ParameterSpace
pars_file = data_path + data_label + '_ParameterSpace.py'
pars = ParameterSpace(pars_file)

# print the full nested structure of the results dictionaries
pars.print_stored_keys(results_path)

# look at an example
# with open(results_path + 'Results_' + pars[30].label, 'r') as f:
# 	example_results = pickle.load(f)

# harvest specific results (specifying the list of keys necessary to reach the data)
results = {
	'regularity': {
		'lvs': []},
	'synchrony': {
		'ISI_distance': [],
		'SPIKE_distance': [],
		'SPIKE_sync_distance': []},
	'analogs': {#'EI_CC': [],
	            'IE_ratio': []} #_corrected
}
expected_values = {
	'regularity': {
		'lvs': 1.},
	'synchrony': {
		'ISI_distance': 0.5,
		'SPIKE_distance': 0.3,
		'SPIKE_sync_distance': 0.25},
	'analogs': {
		#'EI_CC': -1.,
		'IE_ratio': 0.}
}

# Regularity data
for k in results['regularity'].keys():
	_, result = pars.harvest(results_path, key_set='spiking_activity/E/{0}'.format(k))
	results['regularity'][k] = clean_array(result)
	I_reg = np.empty(np.shape(result))
	I_reg[:] = np.nan
	for idx, val in np.ndenumerate(clean_array(result)):
		if isinstance(val, tuple):
			value = val[0]
		else:
			value = val
		if not np.isnan(value):
			I_reg[idx] = value#abs(value - expected_values['regularity'][k])

	reg_values = I_reg.flatten()

# Synchrony data
for ii, k in enumerate(results['synchrony'].keys()):
	_, result = pars.harvest(results_path, key_set='spiking_activity/E/{0}'.format(k))
	results['synchrony'][k] = clean_array(result)

	I_sync = np.empty((np.shape(result)[0], np.shape(result)[1], 3))
	I_sync[:] = np.nan

	for idx, val in np.ndenumerate(clean_array(result)):
		if isinstance(val, tuple):
			value = val[0]
		else:
			value = val
		I_sync[idx[0], idx[1], ii] = abs(value - expected_values['synchrony'][k])

	sync_values = np.nanmean(I_sync, 2).flatten()

# AIness
ai_ness = np.empty((np.shape(result)[0], np.shape(result)[1], 2))
ai_ness[:, :, 0] = I_reg
ai_ness[:, :, 1] = np.nanmean(I_sync, 2)

ainess_values = np.nanmean(ai_ness, 2).flatten()

# EI ratio
for k in results['analogs'].keys():
	_, result = pars.harvest(results_path, key_set='analog_activity/E/{0}'.format(k))
	results['analogs'][k] = clean_array(result)

	ei_values = clean_array(result).astype(float).flatten()

# Error data
_, result = pars.harvest(results_path, key_set='performance/EI/V_m0/ridge_classifier/raw/MAE')
# _, result = pars.harvest(results_path, key_set='performance/EI/V_m0/ridge_classifier/label/performance')
error_values = clean_array(result).astype(float).flatten()


# # Plot data
fig = pl.figure()
ax1 = fig.add_subplot(121, projection='3d')

ax1.scatter3D(ei_values, ainess_values, error_values)
# ax1.plot(ei_values, error_values, 'k+', zdir='y', zs=1, alpha=0.8)
# ax1.plot(ainess_values, error_values, 'k+', zdir='x', zs=1, alpha=0.8)

ax1.set_xlabel("EI ratio")
ax1.set_ylabel("AIness")
ax1.set_zlabel("MAE")

ax2 = fig.add_subplot(122, projection='3d')

ax2.scatter3D(reg_values, sync_values, error_values)
# ax2.plot(reg_values, error_values, 'k+', zdir='y', zs=1, alpha=0.8)
# ax2.plot(sync_values, error_values, 'k+', zdir='x', zs=1, alpha=0.8)

ax2.set_xlabel(r'$\mathrm{I^{reg}}$')
ax2.set_ylabel(r'$\mathrm{I^{sync}}$')
ax2.set_zlabel("MAE")

pl.show()