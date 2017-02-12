from modules.parameters import ParameterSpace, copy_dict
from modules.signals import empty
from modules.visualization import *
from modules.io import set_project_paths
from defaults.paths import paths
import matplotlib.pyplot as pl

"""
noise_driven_dynamics_singletrial
- read and plot data recorded from noise_driven_dynamics experiment with 2 parameters (2d ParameterSpace)
- one trial per condition
- data stored with summary_only=True (only the means are read out here)
"""

# data parameters
project = 'state_transfer'
data_type = 'SpikeNoise'  # 'SpikeNoise'
trial = 0
population_of_interest = 'Global'  # results are provided for only one population (choose Global to get the totals)
data_path = "/home/zajzon/code/nst/network_simulation_testbed/data/"
data_label = 'state_transfer_onepool_noisedriven'

# set defaults and paths
set_project_paths(project)
set_global_rcParams(paths['local']['matplotlib_rc'])

# re-create ParameterSpace
pars_file = data_path + data_label + '_ParameterSpace.py'
pars = ParameterSpace(pars_file)

# harvest data
results = pars.harvest(data_path+data_label+'/Results/')

# initialize data arrays
results_arrays = {'activity': {'mean_rates': np.zeros_like(results[0]), 'ffs': np.zeros_like(results[0])},
               'regularity': {'cvs': np.zeros_like(results[0]), 'lvs': np.zeros_like(results[0]),
                        'lvRs': np.zeros_like(results[0]), 'iR': np.zeros_like(results[0]),
                        'cvs_log': np.zeros_like(results[0]), 'ents': np.zeros_like(results[0]),
                        'isi_5p': np.zeros_like(results[0]), 'ai': np.zeros_like(results[0])},
               'synchrony': {'ccs': np.zeros_like(results[0]), 'ccs_pearson': np.zeros_like(results[0]),
                        'd_vp': np.zeros_like(results[0]), 'd_vr': np.zeros_like(results[0]),
                        'ISI_distance': np.zeros_like(results[0]), 'SPIKE_distance': np.zeros_like(results[0]),
                        'SPIKE_sync_distance': np.zeros_like(results[0])},
               'analogs': {'EI_CC': np.zeros_like(results[0]), 'mean_V_m': np.zeros_like(results[0]),
                           'std_V_m': np.zeros_like(results[0]), 'mean_I_ex': np.zeros_like(results[0]),
                           'std_I_ex': np.zeros_like(results[0]), 'mean_I_in': np.zeros_like(results[0]),
                           'std_I_in': np.zeros_like(results[0]), 'IE_ratio': np.zeros_like(results[0])}}
expected_values = {'activity': {'ffs': 1., 'mean_rates': 5.},
                  'regularity': {'cvs': 1., 'lvs': 1., 'cvs_log': 0.25},
                  'synchrony': {'ISI_distance': 0.5, 'SPIKE_distance': 0.3, 'SPIKE_sync_distance': 0.25},
                  'analogs': {'EI_CC': -1., 'IE_ratio': 0.}}

for x_value in pars.parameter_axes['xticks']:
	for y_value in pars.parameter_axes['yticks']:
		label = data_label + '_' + pars.parameter_axes['xlabel'] + '={0}_'.format(str(x_value)) + \
		        pars.parameter_axes['ylabel'] + '={0}'.format(str(y_value))
		idx = np.where(results[0] == label)
		d = results[1][idx][0]
		print(label, idx)

		if d is not None and bool(d['spiking_activity']):
			metrics = d['spiking_activity'][population_of_interest].keys()
			for x in results_arrays['activity'].keys():
				if x in metrics:
					if not empty(d['spiking_activity'][population_of_interest][x]):
						if not isinstance(d['spiking_activity'][population_of_interest][x], float):
							results_arrays['activity'][x][idx] = d['spiking_activity'][population_of_interest][x][0]
						else:
							results_arrays['activity'][x][idx] = d['spiking_activity'][population_of_interest][x]

			for x in results_arrays['regularity'].keys():
				if x in metrics:
					if not empty(d['spiking_activity'][population_of_interest][x]):
						if not isinstance(d['spiking_activity'][population_of_interest][x], float):
							results_arrays['regularity'][x][idx] = d['spiking_activity'][
								population_of_interest][x][0]
						else:
							results_arrays['regularity'][x][idx] = d['spiking_activity'][population_of_interest][x]

			for x in results_arrays['synchrony'].keys():
				if x in metrics:
					if not empty(d['spiking_activity'][population_of_interest][x]):
						if not isinstance(d['spiking_activity'][population_of_interest][x], float):
							results_arrays['synchrony'][x][idx] = d['spiking_activity'][population_of_interest][x][0]
						else:
							results_arrays['synchrony'][x][idx] = d['spiking_activity'][population_of_interest][x]

		if d is not None and bool(d['analog_activity']) and population_of_interest in d['analog_activity'].keys():
				metrics = d['analog_activity'][population_of_interest].keys()
				for x in results_arrays['analogs'].keys():
					if x in metrics:
						if not empty(d['analog_activity'][population_of_interest][x]):
							results_arrays['analogs'][x][idx] = np.mean(d['analog_activity'][population_of_interest][x])

########################################################################################################################
# Plot
pl_props = copy_dict(pars.parameter_axes, {'xlabel': r'$' + pars.parameter_axes['ylabel'] + '$',
                                           'ylabel': r'$' + pars.parameter_axes['xlabel'] + '$',
                                           'xticklabels': pars.parameter_axes['yticklabels'][::2],
                                           'yticklabels': pars.parameter_axes['xticklabels'][::2],
                                           'xticks': np.arange(0., len(pars.parameter_axes['yticks']), 2.),
                                           'yticks': np.arange(0., len(pars.parameter_axes['xticks']), 2.),})

fig1 = pl.figure()
fig1.suptitle('Regularity metrics')
ax11 = fig1.add_subplot(241)
ax12 = fig1.add_subplot(242)
ax13 = fig1.add_subplot(243)
ax14 = fig1.add_subplot(244)
ax15 = fig1.add_subplot(245)
ax16 = fig1.add_subplot(246)
ax17 = fig1.add_subplot(247)
ax18 = fig1.add_subplot(248)

image_arrays = [x.astype(float) for x in results_arrays['regularity'].values()]
boundaries = []
for x in results_arrays['regularity'].keys():
	if x in expected_values['regularity'].keys():
		boundaries.append([expected_values['regularity'][x]])
	else:
		boundaries.append([None])
labels = [r'$'+x+'$' for x in results_arrays['regularity'].keys()]
plot_2d_parscans(image_arrays=image_arrays, axis=[ax11, ax12, ax13, ax14, ax15, ax16, ax17, ax18],
                 fig_handle=fig1, labels=labels, cmap='jet', boundaries=boundaries, **pl_props)

fig2 = pl.figure()
fig2.suptitle('Synchrony metrics')
ax21 = fig2.add_subplot(241)
ax22 = fig2.add_subplot(242)
ax23 = fig2.add_subplot(243)
ax24 = fig2.add_subplot(244)
ax25 = fig2.add_subplot(245)
ax26 = fig2.add_subplot(246)
ax27 = fig2.add_subplot(247)

image_arrays = [x.astype(float) for x in results_arrays['synchrony'].values()]
boundaries = []
for x in results_arrays['synchrony'].keys():
	if x in expected_values['synchrony'].keys():
		boundaries.append([expected_values['synchrony'][x]])
	else:
		boundaries.append([None])
labels = [r'$'+x+'$' for x in results_arrays['synchrony'].keys()]
plot_2d_parscans(image_arrays=image_arrays, axis=[ax21, ax22, ax23, ax24, ax25, ax26, ax27],#, ax18],
                 fig_handle=fig2, labels=labels, cmap='jet', boundaries=boundaries, **pl_props)

fig3 = pl.figure()
fig3.suptitle('Activity metrics')
ax31 = fig3.add_subplot(121)
ax32 = fig3.add_subplot(122)
image_arrays = [x.astype(float) for x in results_arrays['activity'].values()]
boundaries = []
for x in results_arrays['activity'].keys():
	if x in expected_values['activity'].keys():
		boundaries.append([expected_values['activity'][x]])
	else:
		boundaries.append([None])
labels = [r'$'+x+'$' for x in results_arrays['activity'].keys()]
plot_2d_parscans(image_arrays=image_arrays, axis=[ax31, ax32],
                 fig_handle=fig3, labels=labels, cmap='jet', boundaries=boundaries, **pl_props)


fig4 = pl.figure()
fig4.suptitle('Analog signal metrics')
ax41 = fig4.add_subplot(231)
ax42 = fig4.add_subplot(232)
ax43 = fig4.add_subplot(233)
ax44 = fig4.add_subplot(234)
ax45 = fig4.add_subplot(235)
use_keys = ['mean_V_m', 'mean_I_ex', 'mean_I_in', 'IE_ratio', 'EI_CC']
labels = ['$\langle V_{m} \rangle$', '$\langle I_{E} \rangle$', '$\langle I_{I} \rangle$',
          '$\langle I_{I}-I_{E} \rangle$', '$CC_{EI}$']
image_arrays = [x.astype(float) for k, x in results_arrays['analogs'].items() if k in use_keys]
boundaries = []
for x in results_arrays['analogs'].keys():
	if x in expected_values['analogs'].keys():
		boundaries.append([expected_values['analogs'][x]])
	else:
		boundaries.append([None])
plot_2d_parscans(image_arrays=image_arrays, axis=[ax41, ax42, ax43, ax44, ax45],
                 fig_handle=fig4, labels=labels, cmap='jet', boundaries=boundaries, **pl_props)


fig5 = pl.figure()
fig5.suptitle('Summary')
ax51 = fig5.add_subplot(221)
ax52 = fig5.add_subplot(222)
ax53 = fig5.add_subplot(223)
ax54 = fig5.add_subplot(224)
image_arrays = []
labels = []
for k, v in expected_values.items():
	summary_result = np.zeros_like(results[0])
	labels.append(k)
	for idx, _ in np.ndenumerate(np.zeros_like(results[0])):
		result_vector = [results_arrays[k][k1][idx] for k1, v1 in v.items()]
		target_vector = [v1 for k1, v1 in v.items()]
		dist = [((x - target_vector[idd])**2) for idd, x in enumerate(result_vector)]
		summary_result[idx] = np.mean(dist)
	image_arrays.append(summary_result.astype(float))

plot_2d_parscans(image_arrays=image_arrays, axis=[ax51, ax52, ax53, ax54],
                 fig_handle=fig5, labels=labels, cmap='jet', boundaries=[], **pl_props)

##########################################
# # main figure ###
# # fig6 = pl.figure()
# # fig6.suptitle('AIness')
# # ax61 = fig6.add_subplot(111)
# #
# # pl_props = copy_dict(pars.parameter_axes, {'xlabel': r'$\rho_{\mathrm{u}}$',
# #                                            'ylabel': r'$\mathrm{g}$',
# #                                            'xticklabels': pars.parameter_axes['yticklabels'][::2],
# #                                            'yticklabels': pars.parameter_axes['xticklabels'][::4],
# #                                            'xticks': np.arange(0., len(pars.parameter_axes['yticks']), 2.),
# #                                            'yticks': np.arange(0., len(pars.parameter_axes['xticks']), 4.),})
# #
# # sync_summary = image_arrays[labels.index('synchrony')]
# # reg_summary = image_arrays[labels.index('regularity')]
# # activity_summary = image_arrays[labels.index('activity')]
# # analogs_summary = image_arrays[labels.index('analogs')]
# #
# # # normalize
# # sync_summary = (sync_summary - np.min(sync_summary)) / (np.max(sync_summary) - np.min(sync_summary))
# # reg_summary = (reg_summary - np.min(reg_summary)) / (np.max(reg_summary) - np.min(reg_summary))
# # activity_summary = (activity_summary - np.min(activity_summary)) / (np.max(activity_summary) - np.min(activity_summary))
# # analogs_summary = (analogs_summary - np.min(analogs_summary)) / (np.max(analogs_summary) - np.min(analogs_summary))
# #
# # ai_ness = ((sync_summary + reg_summary) / 2.)
# # plot_2d_parscans(image_arrays=[ai_ness.astype(float)], axis=[ax61],
# #                  fig_handle=fig6, labels=[], cmap='jet', boundaries=[], **{}) # coolwarm, rainbow
# #
# # ax61.scatter(np.where(ai_ness == ai_ness.min())[1][0], np.where(ai_ness == ai_ness.min())[0][0], s=20, c='red',
# #              marker='o')
# # ax61.set(**pl_props)
# # ax61.grid(False)
#
#
#
# fig7 = pl.figure()
# fig7.suptitle("All constraints")
# ax71 = fig7.add_subplot(111)
#
# values = ((sync_summary + reg_summary + activity_summary + analogs_summary) / 4.)
# plot_2d_parscans(image_arrays=[values.astype(float)], axis=[ax71],
#                  fig_handle=fig7, labels=[], cmap='jet', boundaries=[], **{}) # coolwarm, rainbow
#
# ax71.scatter(np.where(values == values.min())[1][0], np.where(values == values.min())[0][0], s=20, c='red',
#              marker='o')
# ax71.set(**pl_props)
# ax71.grid(False)


pl.show()
