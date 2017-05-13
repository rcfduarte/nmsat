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
neuron_RTF
- read and plot rate transfer functions
"""

# data parameters
project = 'heterogeneity_project'
data_type = 'homogeneous'
data_path = '/media/neuro/Data/Heterogeneity_NEW/singleneuron_RTF/'
data_label = 'HT_singleneuron_RTF_{0}'.format(data_type)
results_path = data_path + data_label + '/Results/'

# set defaults and paths
set_project_paths(project)
set_global_rcParams(paths['local']['matplotlib_rc'])

# re-create ParameterSpace
pars_file = data_path + data_label + '_ParameterSpace.py'
pars = ParameterSpace(pars_file)

# print the full nested structure of the results dictionaries
pars.print_stored_keys(results_path)


# analyse and plot
fig = pl.figure()
fig.suptitle('Rate transfer function (Homogeneous)')
# ax = fig.add_subplot(111)
colors = ['blue', 'red', 'Orange']
neuron_types = ['E', 'I1', 'I2']
results_of_interest = ['rate', 'IE_ratio', 'mean_V', 'cv_isi', 'tau_eff']
axes_labels = [r'$\nu_{\mathrm{out}} [sps]$', r'$\langle | \mathrm{I_{in}} - \mathrm{I_{ex}} | \rangle$',
               r'$\langle \mathrm{V_{m}} \rangle$', r'$\mathrm{CV_{ISI}}$', r'\tau_{\mathrm{eff}}']
is_tuple = [False, False, True, False, True]

for ctr, result in enumerate(results_of_interest):
	ax = fig.add_subplot(2, 3, ctr+1)

	for idx, neuron in enumerate(neuron_types):
		print "- Harvesting {0} [{1}]".format(result, neuron)
		if is_tuple[ctr]:
			d = pars.harvest(results_path, key_set='{0}/{1}'.format(neuron, result))[1]
			d_mean = np.array([x[0] for x in d])
			d_std = np.array([x[1] for x in d])

			ax.plot(pars.parameter_axes['xticks'], d_mean, '-', c=colors[idx], lw=3, label=neuron)
			ax.fill_between(pars.parameter_axes['xticks'], d_mean - d_std, d_mean + d_std, facecolor=colors[idx],
			                alpha=0.2)
		else:
			d = pars.harvest(results_path, key_set='{0}/{1}'.format(neuron, result))[1].astype(float)
			ax.plot(pars.parameter_axes['xticks'], d, '-', c=colors[idx], lw=3, label=neuron)
		ax.set_xlabel(r'$\nu_{\mathrm{in}} [sps]$')
		ax.set_ylabel(axes_labels[ctr])
		ax.set_title(r'${0}$'.format(result))
		ax.set_xlim([ax.get_xlim()[0], 100.])
pl.legend()
pl.show()