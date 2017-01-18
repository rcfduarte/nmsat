from modules.parameters import ParameterSpace
from modules.visualization import *
import modules.analysis as analysis
from modules.io import process_template
from defaults.paths import paths
import matplotlib.pyplot as pl
from os import environ, system
import sys

"""
noise_driven_dynamics_singletrial
- read data recorded from noise_driven_dynamics experiment with 2 parameters
- one trial per condition
"""

# data parameters
project = 'encoding_decoding'
data_path = '/home/neuro/Desktop/MANUSCRIPTS/in_preparation/Encoding_Decoding/data/noise_driven_dynamics/DCNoiseInput/'
data_label = 'ED_DCNoiseInput_ParSpace_trial0'

# set defaults and paths
set_project_paths(project)
set_global_rcParams(paths['local']['matplotlib_rc'])

# re-create ParameterSpace
pars_file = data_path + data_label + '_ParameterSpace.py'
pars = ParameterSpace(pars_file)

# harvest data
results = pars.harvest(data_path+data_label+'/')

# initialize data arrays
full_arrays = {'activity': {'mean_rates': np.zeros_like(results[0]), 'ffs': np.zeros_like(results[0])},
               'isis': {'cvs': np.zeros_like(results[0]), 'lvs': np.zeros_like(results[0]),
                        'lvRs': np.zeros_like(results[0]), 'iR': np.zeros_like(results[0]),
                        'cvs_log': np.zeros_like(results[0]), 'ents': np.zeros_like(results[0]),
                        'isi_5p': np.zeros_like(results[0]), 'ai': np.zeros_like(results[0])},
               'sync': {'ccs': np.zeros_like(results[0]), 'ccs_pearson': np.zeros_like(results[0]),
                        'd_vp': np.zeros_like(results[0]), 'd_vr': np.zeros_like(results[0]),
                        'ISI_distance': np.zeros_like(results[0]), 'SPIKE_distance': np.zeros_like(results[0]),
                        'SPIKE_synch_distance': np.zeros_like(results[0])}}
poisson_values = {'activity': {'mean_rates': 5., 'ffs': 1.},
                  'isis': {'cvs': 1., 'lvs': 1., 'lvRs': 1., 'iR': 1., 'cvs_log': 0.25, 'ents': 5., 'isi_5p': 15.,
                           'ai': 0.},
                  'sync': {'ccs': 0., 'ccs_pearson': 0., 'd_vp': 100., 'd_vr': 10., 'ISI_distance': 0.5,
                           'SPIKE_distance': 0., 'SPIKE_synch_distance': 0.25}}
analog_arrays = {'EI_CC': np.zeros_like(results[0]),
                 'mean_V_m': np.zeros_like(results[0]),
                 'std_V_m': np.zeros_like(results[0]),
                 'mean_I_ex': np.zeros_like(results[0]),
                 'std_I_ex': np.zeros_like(results[0]),
                 'mean_I_in': np.zeros_like(results[0]),
                 'std_I_in': np.zeros_like(results[0]),
                 'IE_ratio': np.zeros_like(results[0])}

keys2 = ['E']

ainess = np.zeros_like(results[0])
extra_analysis_parameters = {'time_bin': 1.,
                             'n_pairs': 500,
                             'tau': 20.,
                             'window_len': 100,
                             'summary_only': True,
                             'complete': True,
                             'time_resolved': False}
main_metrics = ['ISI_distance', 'SPIKE_distance', 'ccs_pearson', 'cvs', 'cvs_log', 'd_vp', 'd_vr', 'ents', 'ffs']


for x_value in pars.parameter_axes['xticks']:
	for y_value in pars.parameter_axes['yticks']:
		label = data_label + '_' + pars.parameter_axes['xlabel'] + '={0}_'.format(str(x_value)) + \
		        pars.parameter_axes['ylabel'] + '={0}'.format(str(y_value))
		idx = np.where(results[0] == label)
		d = results[1][idx][0]

		print label, idx

		ai = analysis.compute_ainess(d, main_metrics, pop='E', template_duration=10000.,
               template_resolution=0.1, **extra_analysis_parameters)
		ainess[idx] = ai['E']

		if d is not None and bool(d['spiking_activity']):

			for n_pop in keys2:
				globals()['{0}_full'.format(n_pop)] = copy_dict(full_arrays)

				metrics = d['spiking_activity'][n_pop].keys()
				for x in full_arrays['activity'].keys():
					if x in metrics:
						if not empty(d['spiking_activity'][n_pop][x]):
							if not isinstance(d['spiking_activity'][n_pop][x], float):
								globals()['{0}_full'.format(n_pop)]['activity'][x][idx] = d['spiking_activity'][n_pop][x][0]
							else:
								globals()['{0}_full'.format(n_pop)]['activity'][x][idx] = d['spiking_activity'][n_pop][x]

				for x in full_arrays['isis'].keys():
					if x in metrics:
						if not empty(d['spiking_activity'][n_pop][x]):
							if not isinstance(d['spiking_activity'][n_pop][x], float):
								globals()['{0}_full'.format(n_pop)]['isis'][x][idx] = d['spiking_activity'][n_pop][x][0]
							else:
								globals()['{0}_full'.format(n_pop)]['isis'][x][idx] = d['spiking_activity'][n_pop][x]

				for x in full_arrays['sync'].keys():
					if x in metrics:
						if not empty(d['spiking_activity'][n_pop][x]):
							if not isinstance(d['spiking_activity'][n_pop][x], float):
								globals()['{0}_full'.format(n_pop)]['sync'][x][idx] = d['spiking_activity'][n_pop][x][0]
							else:
								globals()['{0}_full'.format(n_pop)]['sync'][x][idx] = d['spiking_activity'][n_pop][x]

		if d is not None and bool(d['analog_activity']):
			for n_pop in keys2:
				globals()['{0}_analogs'.format(n_pop)] = copy_dict(analog_arrays)

				metrics = d['analog_activity'][n_pop].keys()
				for x in analog_arrays.keys():
					if x in metrics:
						if not empty(d['analog_activity'][n_pop][x]):
							globals()['{0}_analogs'.format(n_pop)][x][idx] = np.mean(d['analog_activity'][n_pop][x])


# plot
fig1 = pl.figure()
fig1.suptitle('ISI metrics')
ax11 = fig1.add_subplot(241)
ax12 = fig1.add_subplot(242)
ax13 = fig1.add_subplot(243)
ax14 = fig1.add_subplot(244)
ax15 = fig1.add_subplot(245)
ax16 = fig1.add_subplot(246)
ax17 = fig1.add_subplot(247)
ax18 = fig1.add_subplot(248)

image_arrays = [x.astype(float) for x in E_full['isis'].values()]
labels = ['$'+x+'$' for x in E_full['isis'].keys()]
boundaries = [[poisson_values['isis'][v]] for v in E_full['isis'].keys()]
pl_props = copy_dict(pars.parameter_axes, {'xlabel': r'$' + pars.parameter_axes['ylabel'] + '$',
                                           'ylabel': r'$' + pars.parameter_axes['xlabel'] + '$',
                                           'xticklabels': pars.parameter_axes['yticklabels'],
                                           'yticklabels': pars.parameter_axes['xticklabels'],
                                           'xticks': np.arange(0., len(pars.parameter_axes['yticks']), 1.),
                                           'yticks': np.arange(0., len(pars.parameter_axes['xticks']), 1.)})
boundaries = [[0.25], [0.8, 1.0], [1.], [0.8, 1.0], [0.1], [0.8, 1.0], [0.8, 1.], [1., 3., 4.]]
plot_2d_parscans(image_arrays=image_arrays, axis=[ax11, ax12, ax13, ax14, ax15, ax16, ax17, ax18],
                 fig_handle=fig1, labels=labels, cmap='jet', boundaries=boundaries, **pl_props)


fig2 = pl.figure()
fig2.suptitle('Activity')
ax21 = fig2.add_subplot(121)
ax22 = fig2.add_subplot(122)
image_arrays = [x.astype(float) for x in E_full['activity'].values()]
labels = ['$'+x+'$' for x in E_full['activity'].keys()]
boundaries = [[poisson_values['activity'][v]] for v in E_full['activity'].keys()]
pl_props = copy_dict(pars.parameter_axes, {'xlabel': r'$' + pars.parameter_axes['ylabel'] + '$',
                                           'ylabel': r'$' + pars.parameter_axes['xlabel'] + '$',
                                           'xticklabels': pars.parameter_axes['yticklabels'],
                                           'yticklabels': pars.parameter_axes['xticklabels'],
                                           'xticks': np.arange(0., len(pars.parameter_axes['yticks']), 1.),
                                           'yticks': np.arange(0., len(pars.parameter_axes['xticks']), 1.)})
boundaries = [[0., 1.0, 3., 5., 10.], [0.9, 1.0]]
plot_2d_parscans(image_arrays=image_arrays, axis=[ax21, ax22], fig_handle=fig2,
                 labels=labels, cmap='jet', boundaries=boundaries, **pl_props)


fig3 = pl.figure()
fig3.suptitle('Synchrony')
ax31 = fig3.add_subplot(241)
ax32 = fig3.add_subplot(242)
ax33 = fig3.add_subplot(243)
ax34 = fig3.add_subplot(244)
ax35 = fig3.add_subplot(245)
ax36 = fig3.add_subplot(246)
ax37 = fig3.add_subplot(247)
#ax38 = fig3.add_subplot(248)

image_arrays = [x.astype(float) for x in E_full['sync'].values()]
labels = ['$'+x+'$' for x in E_full['sync'].keys()]
boundaries = [[poisson_values['sync'][v]] for v in E_full['sync'].keys()]
pl_props = copy_dict(pars.parameter_axes, {'xlabel': r'$' + pars.parameter_axes['ylabel'] + '$',
                                           'ylabel': r'$' + pars.parameter_axes['xlabel'] + '$',
                                           'xticklabels': pars.parameter_axes['yticklabels'],
                                           'yticklabels': pars.parameter_axes['xticklabels'],
                                           'xticks': np.arange(0., len(pars.parameter_axes['yticks']), 1.),
                                           'yticks': np.arange(0., len(pars.parameter_axes['xticks']), 1.)})
#boundaries = [[0.], [0], [100.], [10.], [0.5], [0.], [0.25]]
plot_2d_parscans(image_arrays=image_arrays, axis=[ax31, ax32, ax33, ax34, ax35, ax36, ax37],
                 fig_handle=fig1, labels=labels, cmap='jet', boundaries=boundaries, **pl_props)


# analogs
fig4 = pl.figure()
ax41 = fig4.add_subplot(221)
ax42 = fig4.add_subplot(222)
ax43 = fig4.add_subplot(223)
ax44 = fig4.add_subplot(224)
use_keys = ['mean_V_m', 'IE_ratio', 'EI_CC']
image_arrays = [E_analogs[v].astype(float) for v in use_keys]
for im_ar in image_arrays:
	im_ar[np.where(im_ar==0.)] = np.nan
labels = ['$\langle V_{m} \rangle$', '$\langle I_{I}-I_{E} \rangle$', '$CC_{EI}$']
boundaries = [[-60., -55.], [-0.1, 0., 0.1], [-1., -0.8, -0.6, -0.4]]
plot_2d_parscans(image_arrays=image_arrays, axis=[ax41, ax43, ax44], fig_handle=fig4, labels=labels,
                 cmap='jet', boundaries=boundaries, **pl_props)
pl.show()

# np.mean(np.array([trial0, trial1]),0).shape
