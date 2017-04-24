__author__ = 'duarte'
from defaults.paths import paths
from modules.parameters import ParameterSpace
from modules.io import set_project_paths
from modules.visualization import set_global_rcParams, plot_2d_parscans
import matplotlib.pyplot as pl
import numpy as np

"""
receptor_conductances
- retrieve and plot paramter scans for receptor conductances
"""

# data parameters
neuron_type = 'I2' #'I1' 'E'
project = 'heterogeneity_project'
data_path = '/media/neuro/Data/Heterogeneity_OUT/ReceptorParameters/'
data_label = '{0}_GluRs'.format(neuron_type)
# data_label = '{0}_GABARs'.format(neuron_type)
results_path = data_path + data_label + '/Results/'

# set defaults and paths
set_project_paths(project)
set_global_rcParams(paths['local']['matplotlib_rc'])

# re-create ParameterSpace
pars_file = data_path + data_label + '_ParameterSpace.py'
pars = ParameterSpace(pars_file)

# harvest data
try:
    psc_ratios = pars.harvest(results_path, key_set=neuron_type+'/psc_ratio')
    q_ratios = pars.harvest(results_path, key_set=neuron_type+'/q_ratio')
    PSP_rise = pars.harvest(results_path, key_set=neuron_type+'/PSP_mean_fit_rise')
    PSP_decay = pars.harvest(results_path, key_set=neuron_type+'/PSP_mean_fit_decay')
    PSP_amp = pars.harvest(results_path, key_set=neuron_type+'/PSP_mean_amplitude')
except: # for older data
    psc_ratios = pars.harvest(results_path, key_set='psc_ratio')
    q_ratios = pars.harvest(results_path, key_set='q_ratio')
    PSP_rise = ([], np.zeros_like(q_ratios[1]))
    PSP_decay = ([], np.zeros_like(q_ratios[1]))
    PSP_amp = ([], np.zeros_like(q_ratios[1]))

# plot
fig = pl.figure()
ax1 = fig.add_subplot(231)
ax2 = fig.add_subplot(232)
ax3 = fig.add_subplot(233)
ax4 = fig.add_subplot(234)
ax5 = fig.add_subplot(235)
ax6 = fig.add_subplot(236)

pars.parameter_axes['xticks'] = np.arange(0, len(pars.parameter_axes['yticklabels']))
pars.parameter_axes['yticks'] = np.arange(0, len(pars.parameter_axes['xticklabels']))
pars.parameter_axes['xticklabels'], pars.parameter_axes['yticklabels'] = (pars.parameter_axes['yticklabels'],
                                                                          pars.parameter_axes['xticklabels'])
pars.parameter_axes['xlabel'], pars.parameter_axes['ylabel'] = (pars.parameter_axes['ylabel'], pars.parameter_axes[
	'xlabel'])
ax1.set(**pars.parameter_axes)
ax2.set(**pars.parameter_axes)
ax3.set(**pars.parameter_axes)
ax4.set(**pars.parameter_axes)
ax5.set(**pars.parameter_axes)
ax6.set(**pars.parameter_axes)

plot_2d_parscans(image_arrays=[psc_ratios[1].astype(float), q_ratios[1].astype(float), PSP_rise[1].astype(float),
                               PSP_decay[1].astype(float), PSP_amp[1].astype(float)],
                 axis=[ax1, ax2, ax4, ax5, ax6],
                 fig_handle=fig,
                 labels=['PSC ratio',
                'Charge ratio', 'PSP rise', 'PSP decay', 'PSP amplitude'],
                 boundaries=[[0.05], [0.20, 0.32], [0.7, 3.1], [10., 12.], [0.7, 3.]])