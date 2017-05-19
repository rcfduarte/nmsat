from modules.parameters import ParameterSpace, copy_dict
from modules.signals import empty
from modules.visualization import *
from modules.io import set_project_paths
from defaults.paths import paths
import matplotlib.pyplot as pl

"""

"""

# data parameters
project = 'legenstein_rsc'
data_type = 'SpikeNoise'  # 'SpikeNoise'
trial = 0
population_of_interest = 'EI'  # results are provided for only one population (choose Global to get the totals)
# data_path = "/home/barni/code/fzj/nst/data/"
data_path = "/home/zajzon/code/nst/network_simulation_testbed/data/legenstein/"
data_label = 'Legenstein_kernel_quality_T18'

# set defaults and paths
set_project_paths(project)
set_global_rcParams(paths['local']['matplotlib_rc'])

# re-create ParameterSpace
pars_file = data_path + data_label + '_ParameterSpace.py'
pars = ParameterSpace(pars_file)

# harvest data
results = pars.harvest(data_path+data_label+'/Results/')

# initialize data arrays
results_arrays = {'rank': np.zeros_like(results[0]).T}
ymax = len(pars.parameter_axes['yticks']) - 1

for x_value in pars.parameter_axes['xticks']:
	for y_value in pars.parameter_axes['yticks']:
		label = data_label + '_' + pars.parameter_axes['xlabel'] + '={0}_'.format(str(x_value)) + \
		        pars.parameter_axes['ylabel'] + '={0}'.format(str(y_value))

		idx = np.where(results[0] == label)
		if idx[0].size == 0:
			continue
		d = results[1][idx][0]

		row = ymax - idx[1][0]
		col = idx[0][0]
		if d is not None and bool(d['rank']):
			results_arrays['rank'][row][col] = d['rank'][population_of_interest]['spikes0']

		# print label, d['rank'][population_of_interest]['spikes0']


cmap = "coolwarm"
########################################################################################################################
# Plot
pl_props = copy_dict(pars.parameter_axes, {
										   'xlabel': pars.parameter_axes['xlabel'],
                                           'ylabel': pars.parameter_axes['ylabel'],
                                           'xticklabels': pars.parameter_axes['xticklabels'][::1],
                                           'yticklabels': pars.parameter_axes['yticklabels'][::-1],
                                           'xticks': np.arange(0., len(pars.parameter_axes['xticks']), 1.),
                                           'yticks': np.arange(0., len(pars.parameter_axes['yticks']), 1.),
})

fig1 = pl.figure(1, figsize=(8, 5))
fig1.suptitle('Kernel Quality')
ax11 = fig1.add_subplot(111)

image_arrays = [results_arrays['rank'].astype(float)]
labels = ['rank']
plot_2d_parscans(image_arrays=image_arrays, axis=[ax11],
                 fig_handle=fig1, labels=labels, cmap=cmap, **pl_props)
fig1.tight_layout()
pl.subplots_adjust(top=0.85)
fig1.savefig("kernel_quality_inv{0}.pdf".format(population_of_interest), format='pdf')

