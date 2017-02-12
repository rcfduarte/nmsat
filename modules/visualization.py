__author__ = 'duarte'
"""
============================================================================================
Visualization Module
============================================================================================

Provides all the relevant classes, methods and functions for plotting routines

Classes:
----------
	SpikePlots - wrapper class for all plotting routines applied to population or network spiking data
	AnalogSignalPlots - wrapper class for all plotting routines associated with continuous, analog recorded data
	TopologyPlots - wrapper class for all plotting routines related to network structure and connectivity

Functions:
----------
"""
import matplotlib.pyplot as pl
import matplotlib as mpl
import numpy as np
from matplotlib import rc
import itertools
import matplotlib.cm as cmx
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset, inset_axes
from matplotlib import animation
from matplotlib.patches import Polygon
import scipy.integrate as integ
import scipy.stats as st
from scipy import linalg
import sklearn.decomposition as sk
from sklearn.cluster.bicluster import SpectralBiclustering
import signals
import net_architect
import sys
import os
from modules import check_dependency
import io
import analysis
import nest
import nest.topology as tp
has_networkx = check_dependency('networkx')
if has_networkx:
	import networkx as nx
has_sklearn = check_dependency('sklearn')
has_mayavi = check_dependency('mayavi.mlab')
if has_mayavi:
	from mayavi import mlab
import math
import copy


def set_axes_properties(ax, **kwargs):
	"""
	Set axes properties...
	:param ax: axes handle
	:param kwargs: key-word arguments (dictionary of properties)
	:return:
	"""
	ax.set(kwargs)


def set_plot_properties(pl_handle, **kwargs):
	"""
	Modify and set properties of plot function
	:param pl_handle: plot handle
	:param kwargs: key-word argument, properties dictionary
	"""
	pl_handle.setp(kwargs)


def set_global_rcParams(rc_config_file):
	"""
	Replace the rcParams configuration of matplotlib, with predefined values in the dictionary rc_dict. To restore
	the default values, call mpl.rcdefaults()

	For more details, consult the documentation of rcParams

	:param rc_config_file: path to file containing detailed mpl configuration
	"""
	assert os.path.isfile(rc_config_file), 'input must be path to config file'
	mpl.rc_file(rc_config_file)


def plot_kernel(src_gid, kernel=None, ax=None, color='r'):
	"""
	* adapted from nest.topology
	:param src_gid:
	:param mask:
	:param ax:
	:param color:
	:return:
	"""
	if kernel is not None and isinstance(kernel, dict):
		if 'gaussian' in kernel:
			sigma = kernel['gaussian']['sigma']
			for r in range(3):
				ax.add_patch(pl.Circle(src_gid, radius=(r + 1) * sigma, zorder=-1000,
				                        fc='none', ec=color, lw=3, ls='dashed'))
		else:
			raise ValueError('Kernel type cannot be plotted with this version of PyTopology')


def plot_mask(src_gid, mask=None, ax=None, color='r'):
	"""
	*** adapted from nest.topology.PlotKernel

	Add indication of mask and kernel to axes.

    Adds solid red line for mask. For doughnut mask show inner and outer line.
    If kern is Gaussian, add blue dashed lines marking 1, 2, 3 sigma.
    This function ignores periodic boundary conditions.
    Usually, this function is invoked by PlotTargets.

    Note: You should not use this function in distributed simulations.

    Parameters
    ----------
    src_gid   GID of source neuron  (as single element list), mask and kernel plotted relative to it.
    ax        Axes returned by PlotTargets
    mask      Mask used in creating connections.
    kernel    Kernel used in creating connections.

    mask_color   Color used for line marking mask [default: 'red']
    kernel_color Color used for lines marking kernel [default: 'red']

	"""
	if (ax is not None) and (not isinstance(ax, mpl.axes.Axes)):
		raise ValueError('ax must be matplotlib.axes.Axes instance.')
	assert mask is not None, "Mask must be provided"

	srcpos = np.array(tp.GetPosition([src_gid])[0])

	if 'anchor' in mask:
		offs = np.array(mask['anchor'])
	else:
		offs = np.array([0., 0.])

	if 'circular' in mask:
		r = mask['circular']['radius']
		ax.add_patch(pl.Circle(srcpos + offs, radius=r, zorder=-1000,
		                        fc='none', ec=color, lw=3))
	elif 'doughnut' in mask:
		r_in = mask['doughnut']['inner_radius']
		r_out = mask['doughnut']['outer_radius']
		ax.add_patch(pl.Circle(srcpos + offs, radius=r_in, zorder=-1000,
		                        fc='none', ec=color, lw=3))
		ax.add_patch(pl.Circle(srcpos + offs, radius=r_out, zorder=-1000,
		                        fc='none', ec=color, lw=3))
	elif 'rectangular' in mask:
		ll = mask['rectangular']['lower_left']
		ur = mask['rectangular']['upper_right']
		ax.add_patch(pl.Rectangle(srcpos + ll + offs, ur[0] - ll[0], ur[1] - ll[1],
		                           zorder=-1000, fc='none', ec=color, lw=3))
	else:
		raise ValueError('Mask type cannot be plotted with this version of PyTopology.')

	pl.draw()


# def dot_display(spikelist, gids=None, colors=None, with_rate=True, dt=1.0, start=None, stop=None, display=True,
#                 ax=None, fig=None, save=False, **kwargs):
# 	"""
# 	Simplest case, dot display
# 	:param gids: [list] if some ids should be highlighted in a different color, this should be specified by
# 	providing a list of gids and a list of corresponding colors, if None, no ids are differentiated
# 	:param colors: [list] - list of colors corresponding to the specified gids, if None all neurons are plotted
# 	in the same color (blue)
# 	:param with_rate: [bool] - whether to display psth or not
# 	:param dt: [float] - delta t for the psth
# 	:param display: [bool] - display the figure
# 	:param ax: [axes handle] - axes on which to display the figure
# 	:param save: [bool] - save the figure
# 	:param kwargs: [key=value pairs] axes properties
# 	"""
# 	# TODO: add event highlight...
# 	if (ax is not None) and (not isinstance(ax, mpl.axes.Axes) or not isinstance(ax[0], mpl.axes.Axes)):
# 		raise ValueError('ax must be matplotlib.axes.Axes instance.')
#
# 	if ax is None:
# 		fig = pl.figure()
# 		if with_rate:
# 			ax = pl.subplot2grid((30, 1), (0, 0), rowspan=23, colspan=1)
# 			ax2 = pl.subplot2grid((30, 1), (24, 0), rowspan=5, colspan=1, sharex=ax)
# 			ax2.set(xlabel='Time [ms]', ylabel='Rate')
# 			ax.set(ylabel='Neuron')
# 		else:
# 			ax = fig.add_subplot(111)
# 	else:
# 		if with_rate:
# 			assert isinstance(ax, list), "Incompatible properties... (with_rate requires two axes provided or None)"
# 			ax = ax[0]
# 			ax2 = ax[1]
#
# 	if 'suptitle' in kwargs and fig is not None:
# 		fig.suptitle(kwargs['suptitle'])
# 		kwargs.pop('suptitle')
#
# 	if colors is None:
# 		colors = 'b'
# 	# extract properties from kwargs and divide them into axes properties and others
# 	ax_props = {k: v for k, v in kwargs.iteritems() if k in ax.properties()}
# 	pl_props = {k: v for k, v in kwargs.iteritems() if k not in ax.properties()}  # TODO: improve
#
# 	tt = spikelist.time_slice(start, stop)
#
# 	if gids is None:
# 		times = tt.raw_data()[:, 0]
# 		neurons = tt.raw_data()[:, 1]
# 		ax.plot(times, neurons, '.', color=colors)
# 	else:
# 		assert isinstance(gids, list), "Gids should be a list"
# 		for n, ids in enumerate(gids):
# 			tt1 = spikelist.time_slice(start, stop).id_slice(list(ids))
# 			times = tt1.raw_data()[:, 0]
# 			neurons = tt1.raw_data()[:, 1]
# 			ax.plot(times, neurons, '.', color=colors[n])
# 	if with_rate:
# 		time = tt.time_axis(dt)[:-1]
# 		rate = tt.firing_rate(dt, average=True)
# 		ax2.plot(time, rate, **pl_props)
# 		ax.set(**ax_props)
# 		ax.set(ylim=[min(spikelist.id_list), max(spikelist.id_list)], xlim=[start, stop])
# 		ax2.set(xlim=[start, stop])
# 	else:
# 		ax.set(**ax_props)
# 		ax.set(ylim=[min(spikelist.id_list), max(spikelist.id_list)], xlim=[start, stop])
#
# 	if save:
# 		assert isinstance(save, str), "Please provide filename"
# 		pl.savefig(save)
#
# 	if display:
# 		pl.show(False)


# def simple_raster(spk_times, spk_ids, neuron_ids, ax, **kwargs):
# 	"""
#     Create a simple line raster plot
# 	"""
# 	if min(neuron_ids) == 0:
# 		new_spk_ids = spk_ids - min(neuron_ids)
# 	else:
# 		new_spk_ids = spk_ids - min(neuron_ids)
# 		new_spk_ids -= min(new_spk_ids)
# 		for idx, n in enumerate(spk_times):
# 			ax.vlines(n, new_spk_ids[idx]-0.5, new_spk_ids[idx]+0.5, **kwargs)
# 		ax.set_ylim(min(new_spk_ids)-0.5, max(new_spk_ids)+0.5)
#
# 	return new_spk_ids, ax


def plot_histogram(tmpa, nbins, norm=True, mark_mean=True, ax=None, color='b', display=True, save=False, **kwargs):
	"""

	:return:
	"""
	tmpa = np.array(tmpa)
	if (ax is not None) and (not isinstance(ax, mpl.axes.Axes)):
		raise ValueError('ax must be matplotlib.axes.Axes instance.')

	if ax is None:
		fig = pl.figure()
		ax = fig.add_subplot(111)

	if 'suptitle' in kwargs:
		fig.suptitle(kwargs['suptitle'])
		kwargs.pop('suptitle')

	# extract properties from kwargs and divide them into axes properties and others
	in_pl = ['label', 'alpha', 'orientation']
	ax_props = {k: v for k, v in kwargs.iteritems() if k in ax.properties() and k not in in_pl}
	pl_props = {k: v for k, v in kwargs.iteritems() if k not in ax.properties() or k in in_pl}

	# if len(tmpa) > 1:
	# 	tmp = list(itertools.chain(*tmpa))
	tmpa = tmpa[tmpa != 0]
	if tmpa[np.isnan(tmpa)]:
		tmp = list(tmpa)
		print "Removing {0}".format(str(tmp.pop(np.where(np.isnan(tmpa))[0][0])))
		tmpa = np.array(tmp)
	if tmpa[np.isinf(tmpa)]:
		tmp = list(tmpa)
		print "Removing {0}".format(str(tmp.pop(np.where(np.isinf(tmpa))[0][0])))
		tmpa = np.array(tmp)

	n = 0
	bins = 0
	if norm and list(tmpa):
		weights = np.ones_like(tmpa) / len(tmpa)
		n, bins, patches = ax.hist(tmpa, nbins, weights=weights, **pl_props)#histtype='stepfilled',
		# alpha=0.8)
		pl.setp(patches, 'facecolor', color)
	elif list(tmpa):
		n, bins, patches = ax.hist(tmpa, nbins, **pl_props)
		pl.setp(patches, 'facecolor', color)

	if 'label' in pl_props.keys():
		pl.legend()

	if mark_mean:
		ax.axvline(tmpa.mean(), color=color, linestyle='dashed')

	ax.set(**ax_props)

	if save:
		assert isinstance(save, str), "Please provide filename"
		pl.savefig(save)

	if display:
		pl.show(False)

	return n, bins


def violin_plot(ax, data, pos, location=-1, color='y'):
	"""
    create violin plots on an axis

    :param ax:
    :param data:
    :param pos:
    :param location: location on the axis (-1 left,1 right or 0 both)
    :param bp:
    :return:
    """
	dist = max(pos)-min(pos)
	w = min(0.15*max(dist, 1.0), 0.5)
	for d, p, c in zip(data, pos, color):
		k = st.gaussian_kde(d)     #calculates the kernel density
		m = k.dataset.min()     #lower bound of violin
		M = k.dataset.max()     #upper bound of violin
		x = np.arange(m, M, (M-m)/100.) # support for violin
		v = k.evaluate(x) #violin profile (density curve)
		v = v/v.max() * w #scaling the violin to the available space
		if location:
			ax.fill_betweenx(x, p, (location*v)+p, facecolor=c, alpha=0.3)
		else:
			ax.fill_betweenx(x, p, v + p, facecolor=c, alpha=0.3)
			ax.fill_betweenx(x, p, -v + p, facecolor=c, alpha=0.3)


def box_plot(ax, data, pos):
	"""
	creates one or a set of boxplots on the axis provided
	:param ax: axis handle
	:param data: list of data points
	:param pos: list of x positions
	:return:
	"""
	ax.boxplot(data, notch=1, positions=pos, vert=1)


def summary_statistics(data_list, labels, loc=0, fig=None, cmap='jet'):
	"""

	:return:
	"""
	n_axes = len(data_list)
	if fig is None:
		fig = pl.figure()
	cm = get_cmap(n_axes, cmap)
	for i in range(n_axes):
		globals()['ax_{0}'.format(str(i))] = pl.subplot2grid((1, n_axes*2), (0, i*2), rowspan=1, colspan=2)
		violin_plot(globals()['ax_{0}'.format(str(i))], [data_list[i]], pos=[0], location=loc, color=[cm(i)])
		globals()['ax_{0}'.format(str(i))].set_title(labels[i])
		box_plot(globals()['ax_{0}'.format(str(i))], [data_list[i]], pos=[0])


def isi_analysis_histogram_axes(label=''):
	"""
	Returns the standard axes for the isi histogram plots
	:return:
	"""
	fig1 = pl.figure()
	fig1.suptitle('ISI Metrics - {0}'.format(str(label)))
	ax11 = pl.subplot2grid((3, 11), (0, 0), rowspan=1, colspan=11)
	ax11.set_xscale('log')
	ax12 = pl.subplot2grid((3, 11), (1, 0), rowspan=1, colspan=2)
	ax13 = pl.subplot2grid((3, 11), (1, 3), rowspan=1, colspan=2)
	ax14 = pl.subplot2grid((3, 11), (1, 6), rowspan=1, colspan=2)
	ax15 = pl.subplot2grid((3, 11), (1, 9), rowspan=1, colspan=2)
	ax16 = pl.subplot2grid((3, 11), (2, 0), rowspan=1, colspan=2)
	ax17 = pl.subplot2grid((3, 11), (2, 3), rowspan=1, colspan=2)
	ax18 = pl.subplot2grid((3, 11), (2, 6), rowspan=1, colspan=2)
	ax19 = pl.subplot2grid((3, 11), (2, 9), rowspan=1, colspan=2)
	return fig1, [ax11, ax12, ax13, ax14, ax15, ax16, ax17, ax18, ax19]


def connectivity_axes(synapse_types, equal=True):
	"""
	Returns the axes handles for connectivity plots (one axis per synapse type)
	:return:
	"""
	if equal:
		fig = pl.figure()
		axes = []
	return fig, axes


def plot_histograms(ax_list, data_list, n_bins, args_list, cmap='hsv'):
	assert(len(ax_list) == len(data_list)), "Data dimension mismatch"
	colors = get_cmap(len(ax_list), cmap)
	counter = range(len(ax_list))
	for ax, data, c in zip(ax_list, data_list, counter):
		n, bins = plot_histogram(data, n_bins[c], ax=ax, color=colors(c), **{'histtype': 'stepfilled', 'alpha': 0.6})
		approximate_pdf_isi = st.kde.gaussian_kde(data)
		x = np.linspace(np.min(data), np.max(data), n_bins[c])
		y = approximate_pdf_isi(x)
		y /= np.sum(y)
		ax.plot(x, y, color=colors(c), lw=2)
		ax.set(**args_list[c])
		ax.set_ylim([0., np.max(n)])


def plot_state_analysis(parameter_set, results, summary_only=False, start=None, stop=None, display=True, save=False):
	"""
	"""
	fig1 = []
	fig2 = []
	fig3 = []
	fig1 = pl.figure()
	fig1.suptitle(r'Population ${0}$ - Global Activity $[{1}, {2}]$'.format(
				str(parameter_set.kernel_pars.data_prefix + results[
				'metadata']['population_name']), str(start), str(stop)))
	if bool(results['analog_activity']):
		ax1 = pl.subplot2grid((23, 1), loc=(0, 0), rowspan=11, colspan=1)
		ax2 = pl.subplot2grid((23, 1), loc=(11, 0), rowspan=3, colspan=1, sharex=ax1)
		ax3 = pl.subplot2grid((23, 1), loc=(17, 0), rowspan=3, colspan=1, sharex=ax1)
		ax4 = pl.subplot2grid((23, 1), loc=(20, 0), rowspan=3, colspan=1, sharex=ax1)
	else:
		ax1 = pl.subplot2grid((25, 1), loc=(0, 0), rowspan=20, colspan=1)
		ax2 = pl.subplot2grid((25, 1), loc=(20, 0), rowspan=5, colspan=1, sharex=ax1)

	colors = ['b', 'r', 'Orange', 'gray', 'g'] # color sequence for the different populations (TODO automate)

	if bool(results['spiking_activity']):
		pop_names = results['spiking_activity'].keys()

		spiking_activity = results['metadata']['spike_list']
		rp = SpikePlots(spiking_activity, start, stop)
		if display:
			rp.print_activity_report(label=results['metadata']['population_name'],
			                         results=results['spiking_activity'][results['metadata']['population_name']])
		plot_props = {'xlabel': 'Time [ms]', 'ylabel': 'Neuron', 'color': 'b', 'linewidth': 1.0,
		              'linestyle': '-'}
		if len(pop_names) > 1 or 'sub_population_names' in results['metadata'].keys():
			gids = results['metadata']['sub_population_gids']
			rp.dot_display(gids=gids, colors=colors[:len(gids)], ax=[ax1, ax2], with_rate=True, display=False,
			               save=False, **plot_props)
		else:
			rp.dot_display(ax=[ax1, ax2], with_rate=True, display=False, save=False, **plot_props)
		ax2.set_ylabel(r'$\mathrm{\bar{r}(t)} \mathrm{[sps]}$')
		ax1.set_ylabel(r'$\mathrm{Neuron}$')
		#################
		if not summary_only:
			fig2 = pl.figure()
			fig2.suptitle(r'Population ${0}$ - Spiking Statistics $[{1}, {2}]$'.format(
						str(parameter_set.kernel_pars.data_prefix + results[
							'metadata']['population_name']), str(start), str(stop)))
			ax21 = pl.subplot2grid((9, 14), loc=(0, 0), rowspan=4, colspan=4)
			ax22 = pl.subplot2grid((9, 14), loc=(0, 5), rowspan=4, colspan=4)
			ax23 = pl.subplot2grid((9, 14), loc=(0, 10), rowspan=4, colspan=4)
			ax24 = pl.subplot2grid((9, 14), loc=(5, 3), rowspan=4, colspan=4)
			ax25 = pl.subplot2grid((9, 14), loc=(5, 8), rowspan=4, colspan=4)

			for indice, name in enumerate(pop_names):
				plot_props = {'xlabel': 'Rates', 'ylabel': 'Count', 'histtype': 'stepfilled', 'alpha': 0.4}

				plot_histogram(results['spiking_activity'][name]['mean_rates'], nbins=100, norm=True, ax=ax21, color=colors[indice],
				                   display=False, save=False, **plot_props)
				plot_props.update({'xlabel': 'ISI'})  # , 'yscale': 'log'}) #, 'xscale': 'log'})##
				ax22.set_yscale('log')
				plot_histogram(results['spiking_activity'][name]['isi'], nbins=100, norm=True, ax=ax22, color=colors[indice],
				                   display=False, save=False, **plot_props)
				plot_props['xlabel'] = 'CC'
				tmp = results['spiking_activity'][name]['ccs']
				if not isinstance(tmp, np.ndarray):
					tmp = np.array(tmp)
				ccs = tmp[~np.isnan(tmp)] #tmp
				plot_histogram(ccs, nbins=100, norm=True, ax=ax23, color=colors[indice],
				                   display=False, save=False, **plot_props)
				plot_props['xlabel'] = 'FF'
				tmp = results['spiking_activity'][name]['ffs']
				if not isinstance(tmp, np.ndarray):
					tmp = np.array(tmp)
				ffs = tmp[~np.isnan(tmp)]
				plot_histogram(ffs, nbins=100, norm=True, ax=ax24, color=colors[indice],
				                   display=False, save=False, **plot_props)
				plot_props['xlabel'] = '$CV_{ISI}$'
				tmp = results['spiking_activity'][name]['cvs']
				cvs = tmp[~np.isnan(tmp)]
				if not isinstance(tmp, np.ndarray):
					tmp = np.array(tmp)
				plot_histogram(cvs, nbins=100, norm=True, ax=ax25, color=colors[indice],
				                   display=False, save=False, **plot_props)

	if bool(results['analog_activity']):
		pop_names = results['analog_activity'].keys()
		
		for indice, name in enumerate(pop_names):
			if len(results['analog_activity'][name]['recorded_neurons']) > 1:
				fig3 = pl.figure()
				fig3.suptitle(r'Population ${0}$ - Analog Signal Statistics [${1}, {2}$]'.format(
						str(parameter_set.kernel_pars.data_prefix + results[
						'metadata']['population_name']), str(start), str(stop)))
				ax31 = pl.subplot2grid((6, 3), loc=(2, 0), rowspan=3, colspan=1)
				ax32 = pl.subplot2grid((6, 3), loc=(2, 1), rowspan=3, colspan=1)
				ax33 = pl.subplot2grid((6, 3), loc=(2, 2), rowspan=3, colspan=1)

				plot_props = {'xlabel': r'$\langle V_{m} \rangle$', 'ylabel': 'Count', 'histtype': 'stepfilled',
				              'alpha': 0.8}
				plot_histogram(results['analog_activity'][name]['mean_V_m'], nbins=20, norm=True, ax=ax31,
				               color=colors[indice],
				                   display=False, save=False, **plot_props)
				plot_props.update({'xlabel': r'$\langle I_{Syn}^{Total} \rangle$'}) #, 'label': r'\langle I_{Exc}
				# \rangle'})
				plot_histogram(results['analog_activity'][name]['mean_I_ex'], nbins=20, norm=True, ax=ax32, color='b',
				                   display=False, save=False, **plot_props)
				#plot_props.update({'label': r'\langle I_{Inh} \rangle'})
				plot_histogram(results['analog_activity'][name]['mean_I_in'], nbins=20, norm=True, ax=ax32, color='r',
				                   display=False, save=False, **plot_props)
				#plot_props.update({'label': r'\langle I_{Total} \rangle'})
				plot_histogram(np.array(results['analog_activity'][name]['mean_I_in']) + np.array(results[
					'analog_activity'][name]['mean_I_ex']), nbins=20, norm=True, ax=ax32, color='gray',
				                   display=False, save=False, **plot_props)
				plot_props.update({'xlabel': r'$CC_{I_{E}/I_{I}}$'})
				#plot_props.pop('label')
				plot_histogram(results['analog_activity'][name]['EI_CC'], nbins=20, norm=True, ax=ax33, color=colors[
					indice], display=False, save=False, **plot_props)
			elif results['analog_activity'][name]['recorded_neurons']:
				pop_idx = parameter_set.net_pars.pop_names.index(name)
				###
				times = results['analog_activity'][name]['time_axis']
				vm = results['analog_activity'][name]['single_Vm']
				idx = results['analog_activity'][name]['single_idx']
				
				if len(vm) != len(times):
					times = times[:-1]

				ax4.plot(times, vm, 'k', lw=1)
				idxs = vm.argsort()
				if 'V_reset' in parameter_set.net_pars.neuron_pars[pop_idx].keys() and 'V_th' in parameter_set.net_pars.neuron_pars[pop_idx].keys():
					v_reset = parameter_set.net_pars.neuron_pars[pop_idx]['V_reset']
					v_th = parameter_set.net_pars.neuron_pars[pop_idx]['V_th']
					possible_spike_times = [t for t in idxs if (t < len(vm) - 1) and (vm[t + 1] == v_reset) and (vm[t] != v_reset)]
					ax4.vlines(times[possible_spike_times], v_th, 50., lw=1)

				ax4.set_ylim(min(vm) - 5., 10.)
				ax4.set_ylabel(r'$\mathrm{V_{m} [mV]}$')
				ax3.set_title('Neuron {0}'.format(str(idx)))
				currents = [x for x in results['analog_activity'][name].keys() if x[0] == 'I']

				if not signals.empty(currents):
					cl = ['r', 'b', 'gray']
					for iiddxx, nn_curr in enumerate(currents):
						ax3.plot(times, results['analog_activity'][name][nn_curr], c=cl[iiddxx], lw=1)
					ax3.set_xlim(min(times), max(times))
					ax3.set_ylabel(r'$\mathrm{I^{syn}} \mathrm{[nA]}$')
				else:
					irrelevant_keys = ['single_Vm', 'single_idx']
					other_variables = [x for x in results['analog_activity'][name].keys() if x[:6] == 'single' and x
					not in irrelevant_keys]
					cl = ['g', 'k', 'gray']
					for iiddxx, nn_curr in enumerate(other_variables):
						ax3.plot(times, results['analog_activity'][name][nn_curr], c=cl[iiddxx], lw=1)
						ax3.set_xlim(min(times), max(times))
						ax3.set_ylabel('{0}'.format(nn_curr))

	if display:
		pl.show(False)
	if save:
		assert isinstance(save, str), "Please provide filename"
		if isinstance(fig1, mpl.figure.Figure):
			fig1.savefig(save + results['metadata']['population_name'] + '_Figure1.pdf')
		if isinstance(fig2, mpl.figure.Figure):
			fig2.savefig(save + results['metadata']['population_name'] + '_Figure2.pdf')
		if isinstance(fig3, mpl.figure.Figure):
			fig3.savefig(save + results['metadata']['population_name'] + '_Figure3.pdf')


def plot_fI_curve(input_amplitudes, output_rates, ax=None, save=False, display=False, **kwargs):
	"""

	:param input_amplitudes:
	:param output_rates:
	:return:
	"""
	if (ax is not None) and (not isinstance(ax, mpl.axes.Axes)):
		raise ValueError('ax must be matplotlib.axes.Axes instance.')

	if ax is None:
		fig = pl.figure()
		if 'suptitle' in kwargs:
			fig.suptitle(kwargs['suptitle'])
			kwargs.pop('suptitle')
		else:
			ax = fig.add_subplot(111)
	in_pl = []
	ax_props = {k: v for k, v in kwargs.iteritems() if k in ax.properties() and k not in in_pl}
	pl_props = {k: v for k, v in kwargs.iteritems() if k not in ax.properties() or k in in_pl}

	ax.plot(input_amplitudes, output_rates, **pl_props)
	ax.set(**ax_props)

	if save:
		assert isinstance(save, str), "Please provide filename"
		pl.savefig(save)

	if display:
		pl.show(False)


def plot_singleneuron_isis(isis, ax=None, save=False, display=False, **kwargs):
	"""

	:param isis:
	:return:
	"""
	if (ax is not None) and (not isinstance(ax, mpl.axes.Axes)):
		raise ValueError('ax must be matplotlib.axes.Axes instance.')

	if ax is None:
		fig = pl.figure()
		if 'suptitle' in kwargs:
			fig.suptitle(kwargs['suptitle'])
			kwargs.pop('suptitle')
		else:
			ax = fig.add_subplot(111)

	if 'inset' in kwargs.keys():
		inset = kwargs['inset']
		kwargs.pop('inset')
		ax2 = inset_axes(ax, width="60%", height=1.5, loc=1)
	else:
		inset = None
	in_pl = []
	ax_props = {k: v for k, v in kwargs.iteritems() if k in ax.properties() and k not in in_pl}
	pl_props = {k: v for k, v in kwargs.iteritems() if k not in ax.properties() or k in in_pl}

	ax.plot(range(len(isis)), isis, '.', **pl_props)
	ax.set(**ax_props)

	if inset is not None:
		ax2.plot(range(len(inset['isi'])), inset['isi'], '.')
		inset.pop('isi')

	if save:
		assert isinstance(save, str), "Please provide filename"
		pl.savefig(save+'single_neuron_isis.pdf')

	if display:
		pl.show(False)


def test_offline_filtering(spike_list, N, dt, tau):
	"""

	:param spike_list:
	:param N:
	:param dt:
	:return:
	"""
	start = spike_list.t_start
	stop = spike_list.t_stop

	idx = np.random.permutation(spike_list.id_list)[0]
	mat_idx = idx - min(spike_list.id_list)
	activity_matrix = spike_list.filter_spiketrains(dt, tau, start=start, stop=stop, N=N, display=True)
	spk_train = spike_list.spiketrains[idx]
	t = np.arange(spike_list.t_start, spike_list.t_stop, dt)
	resp = spk_train.exponential_filter(dt, tau, start, stop)

	fig, ax = pl.subplots()
	ax.plot(spk_train.spike_times, 20*np.ones_like(spk_train.spike_times), 'ro')
	ax.plot(t, resp, 'b')
	ax.plot(t, activity_matrix[mat_idx, :], 'g')

	pl.show()


def plot_acc(t, accs, fit_params, acc_function, title='', ax=None, display=True, save=False):
	"""

	:param t:
	:param accs:
	:param fit_params:
	:param acc_function:
	:param title:
	:param ax:
	:param display:
	:param save:
	:return:
	"""
	if (ax is not None) and (not isinstance(ax, mpl.axes.Axes)):
		raise ValueError('ax must be matplotlib.axes.Axes instance.')

	if ax is None:
		fig = pl.figure()
		fig.suptitle(title)
		ax = fig.add_subplot(111)
	else:
		ax.set_title(title)

	for n in range(accs.shape[0]):
		ax.plot(t, accs[n, :], alpha=0.1, lw=0.1, color='k')

	error = np.sum((np.mean(accs, 0) - acc_function(t, *fit_params)) ** 2)
	label = r'$a = {0}, b = {1}, {2}={3}, MSE = {4}$'.format(str(np.round(fit_params[0], 2)), str(np.round(fit_params[
		                                    1], 2)), r'\tau_{int}', str(np.round(fit_params[2], 2)), str(error))
	ax.errorbar(t, np.mean(accs, 0), yerr=st.sem(accs), fmt='', color='k', alpha=0.3)
	ax.plot(t, np.mean(accs, 0), '--')
	ax.plot(t, acc_function(t, *fit_params), 'r', label=label)
	ax.legend()

	ax.set_ylabel(r'Autocorrelation')
	ax.set_xlabel(r'Lag [ms]')
	ax.set_xlim(min(t), max(t))
	#ax.set_ylim(0., 1.)

	if save:
		assert isinstance(save, str), "Please provide filename"
		ax.figure.savefig(save + 'acc_fit.pdf')

	if display:
		pl.show(False)


def scatter_variability(variable, ax):
	"""
	scatter the variance vs mean of the individual neuron's isis
	:param spike_list:
	:return:
	"""
	variable = np.array(variable)
	vars = []
	means = []
	if len(np.shape(variable)) == 2:
		for n in range(np.shape(variable)[0]):
			vars.append(np.var(variable[n, :]))
			means.append(np.mean(variable[n, :]))
	else:
		for n in range(len(variable)):
			vars.append(np.var(variable[n]))
			means.append(np.mean(variable[n]))

	ax.scatter(means, vars, color='k', lw=0.5, alpha=0.3)
	x_range = np.linspace(min(means), max(means), 100)
	ax.plot(x_range, x_range, '--r', lw=2)
	ax.set_xlabel('Means')
	ax.set_ylabel('Variance')


def plot_2d_parscans(image_arrays=[], axis=[], fig_handle=None, labels=[], cmap='jet', boundaries=[], **kwargs):
	"""
	Plots a list of arrays as images in the corresponding axis with the corresponding colorbar

	:return:
	"""
	assert len(image_arrays) == len(axis), "Number of provided arrays must match number of axes"

	origin = 'upper'
	for idx, ax in enumerate(axis):
		if not isinstance(ax, mpl.axes.Axes):
			raise ValueError('ax must be matplotlib.axes.Axes instance.')
		else:
			plt1 = ax.imshow(image_arrays[idx], aspect='auto', origin=origin, cmap=cmap) #  interpolation='nearest',

			if boundaries:
				cont = ax.contour(image_arrays[idx], boundaries[idx], origin='lower', colors='k', linewidths=2)
				pl.clabel(cont, fmt='%2.1f', colors='k', fontsize=12)
			if labels:
				ax.set_title(labels[idx])
			divider = make_axes_locatable(ax)
			cax = divider.append_axes("right", "10%", pad="4%")
			if fig_handle is not None:
				cbar = fig_handle.colorbar(plt1, cax=cax)

			ax.set(**kwargs)
			pl.draw()
	pl.show(block=False)


def plot_3d_volume(X):
	"""

	:return:
	"""
	assert (has_mayavi), "mayavi required"
	b1 = np.percentile(X, 20)
	b2 = np.percentile(X, 80)
	mlab.pipeline.volume(mlab.pipeline.scalar_field(X), vmin=b1, vmax=b2)
	mlab.axes()

	arr = mlab.screenshot()
	pl.imshow(arr)


def plot_3d_parscans(image_arrays=[], axis=[], dimensions=[10, 10, 10], fig_handle=None, labels=[], cmap='jet',
                     boundaries=[],
                     **kwargs):
	"""

	:return:
	"""
	assert(has_mayavi), "mayavi required"
	assert len(image_arrays) == len(axis), "Number of provided arrays mus match number of axes"
	x = np.linspace(0, dimensions[0], 1)
	y = np.linspace(0, dimensions[1], 1)
	z = np.linspace(0, dimensions[2], 1)

	X1, Y1, Z1 = np.meshgrid(x, y, z)

	origin = 'upper'
	for idx, ax in enumerate(axis):
		if not isinstance(ax, mpl.axes.Axes):
			raise ValueError('ax must be matplotlib.axes.Axes instance.')
		else:
			plt1 = ax.imshow(image_arrays[idx], aspect='auto', interpolation='nearest', origin=origin, cmap=cmap)

			if boundaries:
				cont = ax.contour(image_arrays[idx], boundaries[idx], origin='lower', colors='k', linewidths=2)
				pl.clabel(cont, fmt='%2.1f', colors='k', fontsize=12)
			if labels:
				ax.set_title(labels[idx])
			divider = make_axes_locatable(ax)
			cax = divider.append_axes("right", "10%", pad="4%")
			if fig_handle is not None:
				cbar = fig_handle.colorbar(plt1, cax=cax)

			ax.set(**kwargs)
			pl.draw()
	pl.show(block=False)


def recurrence_plot(time_series, dt=1, ax=None, color='k', type='.', display=True, save=False, **kwargs):
	"""
	Plot a general recurrence plot of a 1D time series
	:param time_series:
	:param ax:
	:param kwargs:
	:return:
	"""
	if (ax is not None) and (not isinstance(ax, mpl.axes.Axes)):
		raise ValueError('ax must be matplotlib.axes.Axes instance.')

	if ax is None:
		fig = pl.figure()
		if 'suptitle' in kwargs:
			fig.suptitle(kwargs['suptitle'])
			kwargs.pop('suptitle')
		else:
			ax = fig.add_subplot(111)
	in_pl = []
	ax_props = {k: v for k, v in kwargs.iteritems() if k in ax.properties() and k not in in_pl}
	pl_props = {k: v for k, v in kwargs.iteritems() if k not in ax.properties() or k in in_pl}

	for ii, isi_val in enumerate(time_series):
		if ii < len(time_series) - int(dt):
			ax.plot(isi_val, time_series[ii + int(dt)], type, c=color, **pl_props)
	ax.set(**ax_props)
	ax.set_xlabel(r'$x(t)$')
	ax.set_ylabel(r'$x(t-{0})$'.format(str(dt)))

	if save:
		assert isinstance(save, str), "Please provide filename"
		pl.savefig(save+'recurrence.pdf')

	if display:
		pl.show(False)


def plot_w_out(w_out, label, display=True, save=False):
	"""
	Creates a histogram of the readout weights
	"""
	fig1, ax1 = pl.subplots()
	fig1.suptitle(r"${0}$ - Biclustering W out".format(str(label)))
	n_clusters = np.min(w_out.shape)
	n_bars = np.max(w_out.shape)
	model = SpectralBiclustering(n_clusters=n_clusters, method='log',
	                             random_state=0)
	model.fit(w_out)
	fit_data = w_out[np.argsort(model.row_labels_)]
	fit_data = fit_data[:, np.argsort(model.column_labels_)]
	ax1.matshow(fit_data, cmap=pl.cm.Blues, aspect='auto')
	ax1.set_yticks(range(len(model.row_labels_)))
	ax1.set_yticklabels(np.argsort(model.row_labels_))
	ax1.set_ylabel("Out")
	ax1.set_xlabel("Neuron")

	if np.argmin(w_out.shape) == 0:
		w_out = w_out.copy().T
	fig = pl.figure()
	for n in range(n_clusters):
		locals()['ax_{0}'.format(str(n))] = fig.add_subplot(1, n_clusters, n+1)
		locals()['ax_{0}'.format(str(n))].barh(range(n_bars), w_out[:, n], height=1.0, linewidth=0, alpha=0.8)
		locals()['ax_{0}'.format(str(n))].set_ylim([0, w_out.shape[0]])
		locals()['ax_{0}'.format(str(n))].set_xticklabels([])
		locals()['ax_{0}'.format(str(n))].set_yticklabels([])
	if save:
		assert isinstance(save, str), "Please provide filename"
		fig1.savefig(save+'W_out_Biclustering.pdf')
		fig.savefig(save+'w_out.pdf')
	if display:
		pl.show(False)


def plot_confusion_matrix(matrix, label='', display=True, save=False):
	"""
	"""
	fig1, ax1 = pl.subplots()
	fig1.suptitle("{0} - Confusion Matrix".format(str(label)))
	ax1.matshow(matrix, cmap=pl.cm.YlGn, aspect='auto')

	if save:
		assert isinstance(save, str), "Please provide filename"
		fig1.savefig(save+'confusion_matrix.pdf')
	if display:
		pl.show(False)


def plot_2d_regression_fit(fit_obj, state_matrix, target_labels, readout, display=True, save=False):
	"""
	Take a 2D random sample from the ND state space ...
	"""
	fig1, ax = pl.subplots()
	pca_obj = sk.PCA(n_components=2)
	states = pca_obj.fit(state_matrix).transform(state_matrix)

	fig1.suptitle(r'{0} [{1}] - 2PC [{2}] example'.format(str(readout.name), str(readout.rule),
	              str(pca_obj.explained_variance_ratio_)))
	# dimensions = np.random.permutation(state_matrix.shape[0])[:2]
	# states = state_matrix[:, dimensions]
	fit_obj.fit(states, target_labels)
	# Plot the decision boundary. For that, we will assign a color to each
	# point in the mesh [x_min, m_max]x[y_min, y_max].
	h = 0.01
	x_min, x_max = states[:, 0].min() - .01, states[:, 0].max() + .01
	y_min, y_max = states[:, 1].min() - .01, states[:, 1].max() + .01
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	Z = fit_obj.predict(np.c_[xx.ravel(), yy.ravel()])
	# Put the result into a color plot
	Z = Z.reshape(xx.shape)
	ax.pcolormesh(xx, yy, Z, cmap=pl.cm.Paired)

	# Plot also the training points
	ax.scatter(state_matrix[:, 0], state_matrix[:, 1], c=target_labels, edgecolors='k', cmap=pl.cm.ocean)
	ax.set_xlim(xx.min(), xx.max())
	ax.set_ylim(yy.min(), yy.max())
	ax.set_xticks(())
	ax.set_yticks(())

	if save:
		assert isinstance(save, str), "Please provide filename"
		fig1.savefig(save+'2D_decision_boundaries.pdf')
	if display:
		pl.show(False)


def plot_raster(spike_list, dt, ax, **kwargs):
	ax1a = pl.twinx(ax)
	spike_list.raster_plot(ax=ax, display=False, **kwargs)
	ax.grid(False)
	ax.set_ylabel(r'Neuron')
	ax.set_xlabel(r'Time $[\mathrm{ms}]$')
	ax1a.plot(spike_list.time_axis(dt)[:-1], spike_list.firing_rate(dt, average=True), 'k', lw=1., alpha=0.7)
	ax1a.plot(spike_list.time_axis(10.)[:-1], spike_list.firing_rate(10., average=True), 'r', lw=2.)
	ax1a.grid(False)
	ax1a.set_ylabel(r'Rate $[\mathrm{sps}/s]$')


def mark_epochs(ax, epochs, cmap='jet'):
	labels = np.unique(epochs.keys())
	cm = get_cmap(len(labels), cmap)
	for k, v in epochs.items():
		label_index = np.where(k == labels)[0][0]
		ax.fill_betweenx(np.arange(ax.get_ylim()[0], ax.get_ylim()[1], 0.1), v[0], v[1],
		                 facecolor=cm(label_index), alpha=0.2)


########################################################################################################################
class SpikePlots(object):
	"""
	Wrapper object with all the methods and functions necessary to visualize spiking
	activity from a simple dot display to more visually appealing rasters,
	as well as histograms of the most relevant statistical descriptors and so on..
	"""

	def __init__(self, spikelist, start=None, stop=None, N=None):
		"""
		Initialize SpikePlot object
		:param spikelist: SpikeList object
		:param start: [float] start time for the display (if None, range is taken from data)
		:param stop: [float] stop time (if None, range is taken from data)
		"""
		if not isinstance(spikelist, signals.SpikeList):
			raise Exception("Error, argument should be a SpikeList object")
		self.spikelist = spikelist

		if start is None:
			self.start = self.spikelist.t_start
		else:
			self.start = start
		if stop is None:
			self.stop = self.spikelist.t_stop
		else:
			self.stop = stop
		if N is None:
			self.N = len(self.spikelist.id_list)

	def dot_display(self, gids=None, colors=None, with_rate=True, dt=1.0, display=True, ax=None, fig=None, save=False,
					**kwargs):
		"""
		Simplest case, dot display
		:param gids: [list] if some ids should be highlighted in a different color, this should be specified by
		providing a list of gids and a list of corresponding colors, if None, no ids are differentiated
		:param colors: [list] - list of colors corresponding to the specified gids, if None all neurons are plotted
		in the same color (blue)
		:param with_rate: [bool] - whether to display psth or not
		:param dt: [float] - delta t for the psth
		:param display: [bool] - display the figure
		:param ax: [axes handle] - axes on which to display the figure
		:param save: [bool] - save the figure
		:param kwargs: [key=value pairs] axes properties
		"""
		if (ax is not None) and (not isinstance(ax, list)) and (not isinstance(ax, mpl.axes.Axes)):
			raise ValueError('ax must be matplotlib.axes.Axes instance.')
		elif (ax is not None) and (isinstance(ax, list)):
			for axis_ax in ax:
				if not isinstance(axis_ax, mpl.axes.Axes):
					raise ValueError('ax must be matplotlib.axes.Axes instance.')

		if ax is None:
			fig = pl.figure()
			if 'suptitle' in kwargs:
				fig.suptitle(kwargs['suptitle'])
				kwargs.pop('suptitle')
			if with_rate:
				ax1 = pl.subplot2grid((30, 1), (0, 0), rowspan=23, colspan=1)
				ax2 = pl.subplot2grid((30, 1), (24, 0), rowspan=5, colspan=1, sharex=ax1)
				ax2.set(xlabel='Time [ms]', ylabel='Rate')
				ax1.set(ylabel='Neuron')
			else:
				ax1 = fig.add_subplot(111)
		else:
			if with_rate:
				assert isinstance(ax, list), "Incompatible properties... (with_rate requires two axes provided or None)"
				ax1 = ax[0]
				ax2 = ax[1]
			else:
				ax1 = ax

		if 'suptitle' in kwargs and fig is not None:
			fig.suptitle(kwargs['suptitle'])
			kwargs.pop('suptitle')

		if colors is None:
			colors = 'b'
		# extract properties from kwargs and divide them into axes properties and others
		ax_props = {k: v for k, v in kwargs.iteritems() if k in ax1.properties()}
		pl_props = {k: v for k, v in kwargs.iteritems() if k not in ax1.properties()}  # TODO: improve

		tt = self.spikelist.time_slice(self.start, self.stop)

		if gids is None:
			times = tt.raw_data()[:, 0]
			neurons = tt.raw_data()[:, 1]
			ax1.plot(times, neurons, '.', color=colors)
		else:
			# TODO @Renato, @barni, this should be reviewed and corrected..
			assert isinstance(gids, list), "Gids should be a list"
			for n, ids in enumerate(gids):
				tt1 = self.spikelist.time_slice(self.start, self.stop).id_slice(list(ids))
				times = tt1.raw_data()[:, 0]
				neurons = tt1.raw_data()[:, 1]
				ax1.plot(times, neurons, '.', color=colors[n])

		# set axis range here, it might still be overwritten below
		ax1.set(ylim=[min(self.spikelist.id_list), max(self.spikelist.id_list)], xlim=[self.start, self.stop])

		if with_rate:
			global_rate = tt.firing_rate(dt, average=True)
			mean_rate = tt.firing_rate(10., average=True)
			if gids is None:
				time = tt.time_axis(dt)[:-1]
				ax2.plot(time, global_rate, **pl_props)
			else:
				for n, ids in enumerate(gids):
					tt1 = self.spikelist.time_slice(self.start, self.stop).id_slice(list(ids))
					time = tt1.time_axis(dt)[:-1]
					rate = tt1.firing_rate(dt, average=True)
					ax2.plot(time, rate, color=colors[n], linewidth=1.0, alpha=0.8)
			ax2.plot(tt.time_axis(10.)[:-1], mean_rate, 'k', linewidth=1.5)
			ax2.set(ylim=[min(global_rate) - 1, max(global_rate) + 1], xlim=[self.start, self.stop])
		else:
			ax1.set(**ax_props)
			ax.set(ylim=[min(self.spikelist.id_list), max(self.spikelist.id_list)], xlim=[self.start, self.stop])
		if save:
			assert isinstance(save, str), "Please provide filename"
			pl.savefig(save)

		if display:
			pl.show(False)

	@staticmethod
	def mark_events(ax, input_obj, start=None, stop=None):
		"""
		Highlight stimuli presentation times in axis
		:param ax:
		:param input_obj:
		:param start:
		:param stop:
		:return:
		"""
		if not isinstance(ax, mpl.axes.Axes):
			raise ValueError('ax must be matplotlib.axes.Axes instance.')
		if start is None:
			start = ax.get_xlim()[0]
		if stop is None:
			stop = ax.get_xlim()[1]

		color_map = get_cmap(input_obj.dimensions)
		y_range = np.arange(ax.get_ylim()[0], ax.get_ylim()[1], 1)

		for k in range(input_obj.dimensions):
			onsets = input_obj.onset_times[k]
			offsets = input_obj.offset_times[k]

			assert(len(onsets) == len(offsets)), "Incorrect input object parameters"

			for idx, on in enumerate(onsets):
				if start-500 < on < stop+500:
					ax.fill_betweenx(y_range, on, offsets[idx], facecolor=color_map(k), alpha=0.3)

	def print_activity_report(self, results=None, label='', n_pairs=500):
		"""
		Displays on screen a summary of the network settings and main statistics
		:param label: Population name
		"""
		tt = self.spikelist.time_slice(self.start, self.stop)

		if results is None:
			stats = {}
			stats.update(analysis.compute_isi_stats(tt, summary_only=True, display=True))
			stats.update(analysis.compute_spike_stats(tt, time_bin=1., summary_only=True, display=True))
			stats.update(analysis.compute_synchrony(tt, n_pairs=n_pairs, bin=1., tau=20., time_resolved=False,
	                                summary_only=True, complete=False))
		else:
			stats = results

		print '\n###################################################################'
		print ' Activity recorded in [%s - %s] ms, from population %s ' % (str(self.start), str(self.stop), str(label))
		print '###################################################################'
		print 'Spiking Neurons: {0}/{1}'.format(str(len(np.nonzero(tt.mean_rates())[0])), str(self.N))
		print 'Average Firing Rate: %.2f / %.2f Hz' % (np.mean(np.array(tt.mean_rates())[np.nonzero(tt.mean_rates())[0]]),
		                                               np.mean(tt.mean_rates()))
		# print 'Average Firing Rate (normalized by N): %.2f Hz' % (np.mean(tt.mean_rates()) * len(tt.id_list)) / self.N
		print 'Fano Factor: %.2f' % stats['ffs'][0]
		print '*********************************\n\tISI metrics:\n*********************************'
		if 'lvs' in stats.keys():
			print '\t- CV: %.2f / - LV: %.2f / - LVR: %.2f / - IR: %.2f' % (stats['cvs'][0], stats['lvs'][0],
			                                                                stats['lvRs'][0], stats['iR'][0])
			print '\t- CVlog: %.2f / - H: %.2f [bits/spike]' % (stats['cvs_log'][0], stats['ents'][0])
			print '\t- 5p: %.2f ms' % stats['isi_5p'][0]
		else:
			print '\t- CV: %.2f' % np.mean(stats['cvs'])

		print '*********************************\n\tSynchrony metrics:\n*********************************'
		if 'ccs_pearson' in stats.keys():
			print '\t- Pearson CC [{0} pairs]: {1}'.format(str(n_pairs), stats['ccs_pearson'][0])
			print '\t- CC [{0} pairs]: {1}'.format(str(n_pairs), str(stats['ccs'][0]))
			if 'd_vr' in stats.keys() and isinstance(stats['d_vr'], float):
				print '\t- van Rossum distance: {0}'.format(str(stats['d_vr']))
			elif 'd_vr' in stats.keys() and not isinstance(stats['d_vr'], float):
				print '\t- van Rossum distance: {0}'.format(str(np.mean(stats['d_vr'])))
			if 'd_vp' in stats.keys() and isinstance(stats['d_vp'], float):
				print '\t- Victor Purpura distance: {0}'.format(str(stats['d_vp']))
			elif 'd_vp' in stats.keys() and not isinstance(stats['d_vp'], float):
				print '\t- Victor Purpura distance: {0}'.format(str(np.mean(stats['d_vp'])))
			if 'SPIKE_distance' in stats.keys() and isinstance(stats['SPIKE_distance'], float):
				print '\t- SPIKE similarity: %.2f / - ISI distance: %.2f ' % (stats[
						                            'SPIKE_distance'], stats['ISI_distance'])
			elif 'SPIKE_distance' in stats.keys() and not isinstance(stats['SPIKE_distance'], float):
				print '\t- SPIKE similarity: %.2f / - ISI distance: %.2f' % (np.mean(stats['SPIKE_distance']),
				                                                            np.mean(stats['ISI_distance']))
			if 'SPIKE_sync' in stats.keys():
				print '\t- SPIKE Synchronization: %.2f' % np.mean(stats['SPIKE_sync'])
		else:
			print '\t- Pearson CC [{0} pairs]: {1}'.format(str(n_pairs), np.mean(stats['ccs']))


############################################################################################
class AnalogSignalPlots(object):
	"""
	Wrapper object for all plots pertaining to continuous signals
	"""

	def __init__(self, analog_signal_list, start=None, stop=None):
		"""
		Initialize AnalogSignalPlot object
		:param analog_signal_list: AnalogSignalList object
		:param start: [float] start time for the display (if None, range is taken from data)
		:param stop: [float] stop time (if None, range is taken from data)
		"""
		if (not isinstance(analog_signal_list, signals.AnalogSignalList)) and (not isinstance(analog_signal_list,
		                                                                              signals.AnalogSignal)):
			raise Exception("Error, argument should be an AnalogSignal or AnalogSignalList")

		self.signal_list = analog_signal_list

		if start is None:
			self.start = self.signal_list.t_start
		else:
			self.start = start
		if stop is None:
			self.stop = self.signal_list.t_stop
		else:
			self.stop = stop

	def plot(self, ax=None, display=True, save=False, **kwargs):
		"""
		Simply plot the contents of the AnalogSignal
		:param ax: axis handle
		:param display: [bool]
		:param save: [bool]
		:param kwargs: extra key-word arguments - particularly important are the axis labels
		and the plot colors
		"""
		if (ax is not None) and (not isinstance(ax, mpl.axes.Axes)):
			raise ValueError('ax must be matplotlib.axes.Axes instance.')

		if ax is None:
			fig, ax = pl.subplots()

		# extract properties from kwargs and divide them into axes properties and others
		ax_props = {k: v for k, v in kwargs.iteritems() if k in ax.properties()}
		pl_props = {k: v for k, v in kwargs.iteritems() if k not in ax.properties()}  # TODO: improve
		tt = self.signal_list.time_slice(self.start, self.stop)

		if isinstance(self.signal_list, signals.AnalogSignal):
			times = tt.time_axis()
			signal = tt.raw_data()
			ax.plot(times, signal, **pl_props)

		elif isinstance(self.signal_list, signals.AnalogSignalList):
			ids = self.signal_list.raw_data()[:, 1]
			for n in np.unique(ids):
				tmp = tt.id_slice([n])
				signal = tmp.raw_data()[:, 0]
				times = tmp.time_axis()
				ax.plot(times, signal, **pl_props)

		ax.set(**ax_props)
		ax.set(xlim=[self.start, self.stop])

		if display:
			pl.show(False)
		if save:
			assert isinstance(save, str), "Please provide filename"
			pl.savefig(save)

	def plot_Vm(self, ax=None, with_spikes=True, v_reset=None, v_th=None, display=True, save=False, **kwargs):
		"""
		Special function to plot the time course of the membrane potential with or without highlighting the spike times
		:param with_spikes: [bool]
		"""

		if (ax is not None) and (not isinstance(ax, mpl.axes.Axes)):
			raise ValueError('ax must be matplotlib.axes.Axes instance.')

		if ax is None:
			fig, ax = pl.subplots()
		ax.set_xlabel('Time [ms]')
		ax.set_ylabel(r'V_{m} [mV]')
		ax.set_xlim(self.start, self.stop)

		tt = self.signal_list.time_slice(self.start, self.stop)

		if isinstance(self.signal_list, signals.AnalogSignalList):
			ids = self.signal_list.raw_data()[:, 1]
			for n in np.unique(ids):
				tmp = tt.id_slice([n])
				vm = tmp.raw_data()[:, 0]
				times = tmp.time_axis()
		elif isinstance(self.signal_list, signals.AnalogSignal):
			times = tt.time_axis()
			vm = tt.raw_data()
		else:
			raise ValueError("times and vm not specified")

		ax_props = {k: v for k, v in kwargs.iteritems() if k in ax.properties()}
		pl_props = {k: v for k, v in kwargs.iteritems() if k not in ax.properties()}  # TODO: improve

		if len(vm) != len(times):
			times = times[:-1]

		ax.plot(times, vm, 'k', **pl_props)

		if with_spikes:
			assert (v_reset is not None) and (v_th is not None), "To mark the spike times, please provide the " \
			                                                  "v_reset and v_th values"
			idxs = vm.argsort()
			possible_spike_times = [t for t in idxs if (t < len(vm) - 1) and (vm[t + 1] == v_reset) and (vm[t] !=
			                                                                    v_reset)]
			ax.vlines(times[possible_spike_times], v_th, 50., color='k', **pl_props)
			ax.set_ylim(min(vm) - 5., 10.)
		else:
			ax.set_ylim(min(vm) - 5., max(vm) + 5.)

		ax.set(**ax_props)

		if display:
			pl.show(False)
		if save:
			assert isinstance(save, str), "Please provide filename"
			pl.savefig(save)
		return ax


############################################################################################
class TopologyPlots(object):
	"""
	Class of plotting routines for network topologies

	Input:
	------
	connection_pars - ParameterSet containing all the connectivity parameters
	populations - list of Population objects in the network to plot
	colors      - color associated with each population
	"""

	def __init__(self, connection_pars, network, colormap='hsv'):
		"""
		"""
		self.network = network
		self.populations = list(signals.iterate_obj_list(network.populations))
		self.colormap = get_cmap(len(self.populations), colormap)
		self.colors = [self.colormap(n) for n in range(len(self.populations))]
		self.connection_parameters = connection_pars.as_dict()
		self.positions = None
		self.full_weights = None

	@staticmethod
	def print_network(depth=2):
		print "\nNetwork Structure: "
		nest.PrintNetwork(depth)

	def to_graph_object(self):
		"""
		Convert network to a weighted graph (using networkx)
		:return: networkx Graph object
		"""
		# global graph
		assert(has_networkx), "networkx package not installed"
		Net = nx.Graph()
		pop_names = [n.name for n in self.populations]
		pop_sizes = [n.size for n in self.populations]
		global_ids = list(itertools.chain(*[n.gids for n in self.populations]))

		# set global positions
		if not np.mean([n.topology for n in self.populations]):
			self.positions = self.set_positions()

		for n_pop in self.populations:
			# local (population) graph
			locals()['{0}'.format(str(n_pop.name))] = nx.Graph()
			# positions
			if n_pop.topology:
				node_dict = {k: tp.GetPosition([k])[0] for k in n_pop.gids}
				# create population graph
				for n_node in node_dict.keys():
					locals()['{0}'.format(str(n_pop.name))].add_node(n_node, pos=node_dict[n_node])
			elif self.positions is not None:
				pop_idx = pop_names.index(n_pop.name)
				if pop_idx == 0:
					pos = self.positions[:n_pop.size]
				else:
					pos = self.positions[np.sum(pop_sizes[:pop_idx]):np.sum(pop_sizes[:pop_idx+1])]
				node_dict = {k: pos[idx] for idx, k in enumerate(n_pop.gids)}
				# create population graph
				for n_node in node_dict.keys():
					locals()['{0}'.format(str(n_pop.name))].add_node(n_node, pos=node_dict[n_node])

			# add Population as sub-graph..
			Net.add_node(locals()['{0}'.format(str(n_pop.name))])

		# connectivity (edges)
		global_weights = self.compile_weights()
		weighted_edges = []
		for index, x in np.ndenumerate(global_weights):
			if x:
				weighted_edges.append((global_ids[index[0]], global_ids[index[1]], x))
		Net.add_weighted_edges_from(weighted_edges)

		return Net

	def set_positions(self, type='random'):
		"""
		Assign positions to the nodes, if no topology has been specified, positions will
		be assigned randomly (type='random') or on a grid (type='grid')
		"""
		network_size = np.sum(list(signals.iterate_obj_list(self.network.n_neurons)))
		if type == 'random':
			pos = (np.sqrt(network_size) * np.random.random_sample((int(network_size), 2))).tolist()
		elif type == 'grid':
			assert (np.sqrt(network_size) % 1) == 0., 'Please choose a value of N with an integer sqrt..'
			xs = np.linspace(0., np.sqrt(network_size) - 1, np.sqrt(network_size))
			pos = [[x, y] for y in xs for x in xs]
			np.random.shuffle(pos)
		else:
			pos = None
		return pos

	def compile_weights(self):
		"""
		Join all the individual weight matrices into one large global weights matrix
		:return:
		"""
		if signals.empty(self.network.synaptic_weights):
			self.network.extract_synaptic_weights()
		N = np.sum(list(signals.iterate_obj_list(self.network.n_neurons)))
		self.full_weights = np.zeros((N, N))
		pop_names = [n.name for n in self.populations]
		pop_sizes = [n.size for n in self.populations]

		for connection, weights in self.network.synaptic_weights.items():
			#print connection
			src_idx = pop_names.index(connection[1])
			tget_idx = pop_names.index(connection[0])
			#
			if src_idx == 0:
				srcs_index = [int(0), int(pop_sizes[src_idx])]
			else:
				srcs_index = [int(np.sum(pop_sizes[:src_idx])), int(np.sum(pop_sizes[:src_idx+1]))]
			if tget_idx == 0:
				tgets_index = [int(0), int(pop_sizes[tget_idx])]
			else:
				tgets_index = [int(np.sum(pop_sizes[:tget_idx])), int(np.sum(pop_sizes[:tget_idx+1]))]
			self.full_weights[srcs_index[0]:srcs_index[1], tgets_index[0]:tgets_index[1]] = np.array(weights.todense())
		return self.full_weights

	def plot_spectral_radius(self):
		"""

		:return:
		"""
		eigs = linalg.eigvals(self.full_weights)
		pl.scatter(np.real(eigs), np.imag(eigs))

	def plot_topology(self, ax=None, dim=2, display=True, save=False, **kwargs):
		"""
		Plot the network topology
		"""
		if (ax is not None) and (not isinstance(ax, mpl.axes.Axes)):
			raise ValueError('ax must be matplotlib.axes.Axes instance.')

		if ax is None and dim < 3:
			fig, ax = pl.subplots()
		elif ax is None and dim == 3:
			fig = pl.figure()
			ax = fig.add_subplot(111, projection='3d')

		ax_props = {k: v for k, v in kwargs.iteritems() if k in ax.properties()}
		plot_props = {k: v for k, v in kwargs.iteritems() if k not in ax.properties()}
		if 'suptitle' in kwargs:
			tt = kwargs['suptitle']
			kwargs.pop('suptitle')
		else:
			tt = ''
		# fig.suptitle(tt)
		ax.set(**ax_props)

		for c, p in zip(self.colors, self.populations):
			assert p.topology, "Population %s has no topology" % str(p.name)
			positions = zip(*[tp.GetPosition([n])[0] for n in nest.GetLeaves(p.layer_gid)[0]])

			if len(positions) < 3:
				ax.plot(positions[0], positions[1], 'o', color=c, label=p.name, **plot_props)
			else:
				Axes3D.scatter(positions[0], positions[1], positions[2], depthshade=True, c=c, label=p.name,
				               **plot_props)
		pl.legend(loc=1)
		if display:
			pl.show(False)
		if save:
			assert isinstance(save, str), "Please provide filename"
			pl.savefig(save)

	def plot_connections(self, synapse_types, src_ids=None, ax=None, with_mask=False, with_kernel=False, display=True,
	                     save=False):
		"""
		plot connections from sources with src_ids

		note: synapse_types needs to be a list of tuples (tget_name, source_name)
		"""
		# TODO: Account for 3D case
		if (ax is not None) and (not isinstance(ax, mpl.axes.Axes)):
			raise ValueError('ax must be matplotlib.axes.Axes instance.')
		if not synapse_types:
			raise ValueError('synapse_types must be provided')

		if ax is None:
			fig, ax = pl.subplots()

		pop_names = [m.name for m in self.populations]
		layer_ids = [m.layer_gid for m in self.populations]

		if src_ids is not None:
			src = src_ids
		else:
			src = None
		for n in synapse_types:
			src_idx = pop_names.index(n[1])
			tgt_idx = pop_names.index(n[0])
			src_layer = layer_ids[src_idx]
			tget_layer = layer_ids[tgt_idx]
			tget_gids = self.populations[tgt_idx].gids

			src_ids = src
			assert self.populations[src_idx].topology, "Population %s has no topology" % str(self.populations[src_idx].name)
			if src_ids is not None:
				assert bool(np.mean([(nn in self.populations[src_idx].gids) for nn in [src_ids]])), "Source id " \
				                                                "must be in source population of synapse_types"
			else:
				# pick center element
				src_ids = tp.FindCenterElement(src_layer)[0]

			if isinstance(src_ids, list):
				for src_gid in src_ids:
					connections = nest.GetConnections([src_gid], tget_gids)
					# exclude connections to devices
					devices = [self.populations[src_idx].attached_devices[kk][0] for kk in range(len(
						self.populations[src_idx].attached_devices))]
					# get target positions
					target_pos = zip(*[tp.GetPosition([k])[0] for k in nest.GetStatus(connections, keys='target') if
					                   k not in devices])
					# plot targets
					ax.scatter(target_pos[0], target_pos[1], 30 * np.array(nest.GetStatus(connections, keys='weight')),
			                   zorder=1, c=self.colors[tgt_idx], label='Targets in %s, from %s [%s]' % (str(n[0]),
			                                                                    str(n[1]), str(src_gid)))
					# mark sender position
					src_pos = tp.GetPosition(src_ids)[0]
					ax.plot(src_pos[0], src_pos[1], 'D', ms=30, c=self.colors[src_idx], label='Source [%s/%s]' % (
					str(src_ids), str(n[1])))
					# ax.add_patch(pl.Circle(src_pos, radius=0.1, zorder=1, fc=self.colors[src_idx], alpha=0.4, ec=''))
					if with_mask:
						mask = self.connection_parameters['conn_specs'][src_idx]['mask']
						plot_mask(src_gid, mask=mask, ax=ax, color='r')
					if with_kernel:
						kernel = self.connection_parameters['conn_specs'][src_idx]['kernel']
						plot_kernel(src_gid, kernel=kernel, ax=ax, color='r')
			else:
				connections = nest.GetConnections([src_ids], tget_gids)
				# exclude connections to devices
				devices = [self.populations[src_idx].attached_devices[kk][0] for kk in range(len(
					self.populations[src_idx].attached_devices))]
				# get target positions
				if nest.GetStatus(connections, keys='target'):
					target_pos = zip(*[tp.GetPosition([k])[0] for k in nest.GetStatus(connections, keys='target') if k not
					                   in devices])
					# plot targets
					ax.scatter(target_pos[0], target_pos[1], 40 * np.array(nest.GetStatus(connections, keys='weight')),
					           zorder=1, c=self.colors[tgt_idx], label='Targets in %s, from %s' % (str(n[0]),
								                                                                               str(src_ids)))
				else:
					print "Sources [%s] in population %s have no target in population %s" % (str(src_ids), str(n[1]),
					                                                                         str(n[0]))

				# TODO: make source marker dependent on the other values
				#  mark sender position
				src_pos = tp.GetPosition([src_ids])[0]
				ax.plot(src_pos[0], src_pos[1], 'D', ms=30, c=self.colors[src_idx], label='Source [%s/%s]' % (
				str(src_ids), str(n[1])))
				if with_mask:
					mask = self.connection_parameters['conn_specs'][src_idx]['mask']
					plot_mask(src_ids, mask=mask, ax=ax, color='r')
				if with_kernel:
					kernel = self.connection_parameters['conn_specs'][src_idx]['kernel']
					plot_kernel(src_ids, kernel=kernel, ax=ax, color='r')

				if display:
					pl.draw()
		pl.legend()
		if display:
			pl.show(False)
		if save:
			assert isinstance(save, str), "Please provide filename"
			pl.savefig(save)

	def plot_connectivity(self, synapse_types=None, ax=None, display=True, save=False):
		"""
		Display the connectivity matrix
		:param synapse_types: [list] of tuples, containing target, source names
		"""
		if (ax is not None) and (not isinstance(ax, mpl.axes.Axes)) and (not isinstance(ax, list)):
			raise ValueError('ax must be matplotlib.axes.Axes instance.')

		if synapse_types is not None:
			pop_names = [m.name for m in self.populations]
			neuron_ids = [m.gids for m in self.populations]

			for ctr, n in enumerate(synapse_types):
				src_idx = pop_names.index(n[1])
				tgt_idx = pop_names.index(n[0])

				syn_name = pop_names[tgt_idx]+pop_names[src_idx]

				if signals.empty(self.network.synaptic_weights):
					w = net_architect.extract_weights_matrix(neuron_ids[src_idx], neuron_ids[tgt_idx], True)
				else:
					w = self.network.synaptic_weights[n]

				if ax is None:
					fig, ax_pl = pl.subplots()
				elif isinstance(ax, list):
					ax_pl = ax[ctr]
				plt1 = ax_pl.imshow(w.todense(), interpolation='nearest', aspect='auto', extent=None, cmap='jet')
				divider = make_axes_locatable(ax_pl)
				cax = divider.append_axes("right", "5%", pad="3%")
				pl.colorbar(plt1, cax=cax)
				ax_pl.set_title(r'${0} \rightarrow p={1}$'.format(str(n), str(float(len(w.nonzero()[0])) / float(
						w.shape[0]*w.shape[1]))))

				if not isinstance(ax, list):
					if display:
						pl.show(False)
					if save:
						assert isinstance(save, str), "Please provide filename"
						pl.savefig(save + '{0}_connectivity.pdf'.format(str(syn_name)))
			if isinstance(ax, list):
				if display:
					pl.show(False)
				if save:
					assert isinstance(save, str), "Please provide filename"
					pl.savefig(save + '_connectivity.pdf')
		else:
			if signals.empty(self.network.synaptic_weights):
				self.network.extract_synaptic_weights()

			weights = self.network.synaptic_weights
			ctr = 0
			for syn_name, w in weights.items():
				if ax is None:
					fig, ax_pl = pl.subplots()
				elif isinstance(ax, list):
					ax_pl = ax[ctr]
				plt1 = ax_pl.imshow(w.todense(), interpolation='nearest', aspect='auto', extent=None, cmap='jet')
				divider = make_axes_locatable(ax_pl)
				cax = divider.append_axes("right", "5%", pad="3%")
				pl.colorbar(plt1, cax=cax)
				ax_pl.set_title(r'${0} \rightarrow p={1}$'.format(str(syn_name), str(float(len(w.nonzero()[0])) / float(
						w.shape[0]*w.shape[1]))))

				if not isinstance(ax, list):
					if display:
						pl.show(False)
					if save:
						assert isinstance(save, str), "Please provide filename"
						pl.savefig(save+'{0}_connectivity.pdf'.format(str(syn_name)))
			if isinstance(ax, list):
				if display:
					pl.show(False)
				if save:
					assert isinstance(save, str), "Please provide filename"
					pl.savefig(save + '_connectivity.pdf')

	def plot_connectivity_delays(self, synapse_types=None, ax=None, display=True, save=False):
		"""
		Display the connectivity matrix
		:param synapse_types: [list] of tuples, containing target, source names
		"""
		if (ax is not None) and (not isinstance(ax, mpl.axes.Axes)) and (not isinstance(ax, list)):
			raise ValueError('ax must be matplotlib.axes.Axes instance.')

		if synapse_types is not None:
			pop_names = [m.name for m in self.populations]
			neuron_ids = [m.gids for m in self.populations]

			for ctr, n in enumerate(synapse_types):
				src_idx = pop_names.index(n[1])
				tgt_idx = pop_names.index(n[0])
				syn_name = pop_names[tgt_idx] + pop_names[src_idx]

				if signals.empty(self.network.synaptic_delays):
					w = net_architect.extract_delays_matrix(neuron_ids[src_idx], neuron_ids[tgt_idx], True)
				else:
					w = self.network.synaptic_delays[n]

				if ax is None:
					fig, ax_pl = pl.subplots()
				elif isinstance(ax, list):
					ax_pl = ax[ctr]
				plt1 = ax_pl.imshow(w.todense(), interpolation='nearest', aspect='auto', extent=None, cmap='jet')
				divider = make_axes_locatable(ax_pl)
				cax = divider.append_axes("right", "5%", pad="3%")
				pl.colorbar(plt1, cax=cax)
				ax_pl.set_title(r'${0} \rightarrow p={1}$'.format(str(n), str(float(len(w.nonzero()[0])) / float(
						w.shape[0]*w.shape[1]))))

				if not isinstance(ax, list):
					if display:
						pl.show(False)
					if save:
						assert isinstance(save, str), "Please provide filename"
						pl.savefig(save + '{0}_delays.pdf'.format(str(syn_name)))
			if isinstance(ax, list):
				if display:
					pl.show(False)
				if save:
					assert isinstance(save, str), "Please provide filename"
					pl.savefig(save + '_delays.pdf')
		else:
			if signals.empty(self.network.synaptic_delays):
				self.network.extract_synaptic_delays()

			weights = self.network.synaptic_delays
			ctr = 0
			for syn_name, w in weights.items():
				if ax is None:
					fig, ax_pl = pl.subplots()
				elif isinstance(ax, list):
					ax_pl = ax[ctr]
				plt1 = ax_pl.imshow(w.todense(), interpolation='nearest', aspect='auto', extent=None, cmap='jet')
				divider = make_axes_locatable(ax_pl)
				cax = divider.append_axes("right", "5%", pad="3%")
				pl.colorbar(plt1, cax=cax)
				ax_pl.set_title(r'${0} \rightarrow p={1}$'.format(str(syn_name), str(float(len(w.nonzero()[0])) / float(
						w.shape[0]*w.shape[1]))))

				if not isinstance(ax, list):
					if display:
						pl.show(False)
					if save:
						assert isinstance(save, str), "Please provide filename"
						pl.savefig(save+'{0}_delays.pdf'.format(str(syn_name)))
			if isinstance(ax, list):
				if display:
					pl.show(False)
				if save:
					assert isinstance(save, str), "Please provide filename"
					pl.savefig(save + '_delays.pdf')

	def plot_weight_histograms(self, synapse_types=None, ax=None, display=True, save=False):
		"""
		Plot histograms for the weight distributions
		:param synapse_types: [list] of tuples, containing target, source names
		"""
		if (ax is not None) and (not isinstance(ax, mpl.axes.Axes)) and (not isinstance(ax, list)):
			raise ValueError('ax must be matplotlib.axes.Axes instance.')

		if synapse_types is not None:
			pop_names = [m.name for m in self.populations]
			neuron_ids = [m.gids for m in self.populations]

			for ctr, n in enumerate(synapse_types):
				src_idx = pop_names.index(n[1])
				tgt_idx = pop_names.index(n[0])

				syn_name = pop_names[tgt_idx]+pop_names[src_idx]

				if signals.empty(self.network.synaptic_weights):
					w = net_architect.extract_weights_matrix(neuron_ids[src_idx], neuron_ids[tgt_idx], True)
				else:
					w = self.network.synaptic_weights[n]

				if ax is None:
					fig, ax_pl = pl.subplots()
				elif isinstance(ax, list):
					ax_pl = ax[ctr]

				weights = w[w.nonzero()].todense()
				weights = list(signals.iterate_obj_list(weights))

				plot_histogram(weights, 100, ax=ax_pl)
				ax_pl.set_title(r'${0}$'.format(str(n)))

				if not isinstance(ax, list):
					if display:
						pl.show(False)
					if save:
						assert isinstance(save, str), "Please provide filename"
						pl.savefig(save + '{0}_wHist.pdf'.format(str(syn_name)))
			if isinstance(ax, list):
				if display:
					pl.show(False)
				if save:
					assert isinstance(save, str), "Please provide filename"
					pl.savefig(save + '_wHist.pdf')
		else:
			if signals.empty(self.network.synaptic_weights):
				self.network.extract_synaptic_weights()

			weights = self.network.synaptic_weights
			ctr = 0
			for syn_name, w in weights.items():
				if ax is None:
					fig, ax_pl = pl.subplots()
				elif isinstance(ax, list):
					ax_pl = ax[ctr]

				weightss = w[w.nonzero()].todense()
				weightss = list(signals.iterate_obj_list(weightss))

				plot_histogram(weightss, 100, ax=ax_pl)
				ax_pl.set_title(r'${0}$'.format(str(syn_name)))

				if not isinstance(ax, list):
					if display:
						pl.show(False)
					if save:
						assert isinstance(save, str), "Please provide filename"
						pl.savefig(save+'{0}_wHist.pdf'.format(str(syn_name)))
			if isinstance(ax, list):
				if display:
					pl.show(False)
				if save:
					assert isinstance(save, str), "Please provide filename"
					pl.savefig(save + '_wHist.pdf')

	def plot_delay_histograms(self, synapse_types=None, ax=None, display=True, save=False):
		"""
		Plot histograms for the weight distributions
		:param synapse_types: [list] of tuples, containing target, source names
		"""
		if (ax is not None) and (not isinstance(ax, mpl.axes.Axes)) and (not isinstance(ax, list)):
			raise ValueError('ax must be matplotlib.axes.Axes instance.')

		if synapse_types is not None:
			pop_names = [m.name for m in self.populations]
			neuron_ids = [m.gids for m in self.populations]

			for ctr, n in enumerate(synapse_types):
				src_idx = pop_names.index(n[1])
				tgt_idx = pop_names.index(n[0])

				syn_name = pop_names[tgt_idx]+pop_names[src_idx]

				if signals.empty(self.network.synaptic_delays):
					w = net_architect.extract_delays_matrix(neuron_ids[src_idx], neuron_ids[tgt_idx], True)
				else:
					w = self.network.synaptic_delays[n]

				if ax is None:
					fig, ax_pl = pl.subplots()
				elif isinstance(ax, list):
					ax_pl = ax[ctr]

				weights = w[w.nonzero()].todense()
				weights = list(signals.iterate_obj_list(weights))

				plot_histogram(weights, 100, ax=ax_pl)
				ax_pl.set_title(r'${0}$'.format(str(n)))

				if not isinstance(ax, list):
					if display:
						pl.show(False)
					if save:
						assert isinstance(save, str), "Please provide filename"
						pl.savefig(save + '{0}_dHist.pdf'.format(str(syn_name)))
			if isinstance(ax, list):
				if display:
					pl.show(False)
				if save:
					assert isinstance(save, str), "Please provide filename"
					pl.savefig(save + '_dHist.pdf')
		else:
			if signals.empty(self.network.synaptic_delays):
				self.network.extract_synaptic_delays()

			weights = self.network.synaptic_delays
			ctr = 0
			for syn_name, w in weights.items():
				if ax is None:
					fig, ax_pl = pl.subplots()
				elif isinstance(ax, list):
					ax_pl = ax[ctr]

				weightss = w[w.nonzero()].todense()
				weightss = list(signals.iterate_obj_list(weightss))

				plot_histogram(weightss, 100, ax=ax_pl)
				ax_pl.set_title(r'${0}$'.format(str(syn_name)))

				if not isinstance(ax, list):
					if display:
						pl.show(False)
					if save:
						assert isinstance(save, str), "Please provide filename"
						pl.savefig(save+'{0}_dHist.pdf'.format(str(syn_name)))
			if isinstance(ax, list):
				if display:
					pl.show(False)
				if save:
					assert isinstance(save, str), "Please provide filename"
					pl.savefig(save + '_dHist.pdf')


class InputPlots(object):
	"""
	Class of plotting routines for input structures

	Input:
	------
	connection_pars - ParameterSet containing all the connectivity parameters
	populations - list of Population objects in the network to plot
	colors      - color associated with each population
	"""

	def __init__(self, stim_obj=None, input_obj=None, noise_obj=None):
		"""
		"""
		self.stim = stim_obj
		self.input = input_obj
		self.noise = noise_obj

	def plot_stimulus_matrix(self, set='train', ax=None, save=False, display=True):
		"""

		"""
		if (ax is not None) and (not isinstance(ax, mpl.axes.Axes)):
			raise ValueError('ax must be matplotlib.axes.Axes instance.')
		if ax is None:
			fig, ax = pl.subplots()
			fig.suptitle('{0} Stimulus Set'.format(set))
			
		data = getattr(self.stim, set + '_set')
		labels = getattr(self.stim, set + '_set_labels')
		plt1 = ax.imshow(1 - data.todense(), interpolation='nearest', aspect='auto', extent=None,
		                 cmap='gray')
		ax.set_xlabel('Time Step')
		ax.set_ylabel('Stimulus')
		ax.set_xticks(np.arange(len(labels)))
		ax.set_yticks(np.arange(len(np.unique(np.array(labels)))))
		ax.set_yticklabels(np.unique(np.array(labels)))

		if display:
			pl.show(False)
		if save:
			assert isinstance(save, str), "Please provide filename"
			pl.savefig(save+'_StimulusMatrix_{0}.pdf'.format(str(set)))

	def plot_input_signal(self, ax=None, save=False, display=True, **kwargs):
		"""
		"""
		ax_props = {k: v for k, v in kwargs.iteritems() if k in ax.properties()}
		plot_props = {k: v for k, v in kwargs.iteritems() if k not in ax.properties()}

		if (ax is not None) and (not isinstance(ax, mpl.axes.Axes)):
			raise ValueError('ax must be matplotlib.axes.Axes instance.')
		if ax is None:
			fig = pl.figure()
			for ii in range(len(self.input.input_signal)):
				globals()['ax' + str(ii)] = pl.subplot2grid((len(self.input.input_signal), 1), (ii, 0), rowspan=1,
				                                           colspan=1)
				globals()['ax' + str(ii)].plot(self.input.time_data, self.input.input_signal[ii].signal, **plot_props)
				globals()['ax' + str(ii)].set_ylabel(r'\sigma_{u}')
				if ii < len(self.input.input_signal) - 1:
					globals()['ax' + str(ii)].set_xticklabels('')
					globals()['ax' + str(ii)].set_xlim(left=min(self.input.time_data), right=max(self.input.time_data))
				else:
					globals()['ax' + str(ii)].set_xlabel('Time [ms]')
					globals()['ax' + str(ii)].set_xlim(left=min(self.input.time_data), right=max(self.input.time_data))
				globals()['ax' + str(ii)].set_ylim([min(self.input.input_signal[ii].signal)-10., max(self.input.input_signal[ii].signal)+10.])
				globals()['ax' + str(ii)].set(**ax_props)
		else:
			for ii in range(len(self.input.input_signal)):
				ax.plot(self.input.time_data, self.input.input_signal[ii].signal, **plot_props)
				# if self.start and self.stop:
				# 	ax.set_xlim(left=self.start, right=self.stop)
				# else:
				ax.set_xlim(left=min(self.input.time_data), right=max(self.input.time_data))
				ax.set_ylim([min(self.input.input_signal[ii].signal)-10., max(self.input.input_signal[ii].signal)+10.])
				ax.set_xlabel('Time [ms]')
				ax.set_ylabel(r'\sigma_{u}')
				ax.set(**ax_props)
		if display:
			pl.show(False)
		if save:
			assert isinstance(save, str), "Please provide filename"
			if not os.path.exists(save+'_InputTimeSeries.pdf'):
				pl.savefig(save+'_InputTimeSeries.pdf')
			else: # cases when the function is called twice
				pl.savefig(save+'_InputTimeSeries2.pdf')

	def plot_noise_component(self, ax=None, save=False, display=True):
		"""
		"""
		if (ax is not None) and (not isinstance(ax, mpl.axes.Axes)):
			raise ValueError('ax must be matplotlib.axes.Axes instance.')
		if ax is None:
			fig = pl.figure()
			for ii in range(len(self.noise.noise_signal)):
				globals()['ax' + str(ii)] = pl.subplot2grid((len(self.noise.noise_signal), 1), (ii, 0), rowspan=1,
				                                           colspan=1)
				globals()['ax' + str(ii)].plot(self.noise.noise_signal[ii].time_axis(), self.noise.noise_signal[
					ii].signal)
				globals()['ax' + str(ii)].set_ylabel(r'\sigma_{u}')
				if ii < len(self.noise.noise_signal) - 1:
					globals()['ax' + str(ii)].set_xticklabels('')
					globals()['ax' + str(ii)].set_xlim(min(self.noise.time_data), max(self.noise.time_data))
				else:
					globals()['ax' + str(ii)].set_xlabel('Time [ms]')
					globals()['ax' + str(ii)].set_xlim(min(self.noise.time_data), max(self.noise.time_data))
		else:
			for ii in range(len(self.noise.noise_signal)):
				ax.plot(self.noise.noise_signal[ii].time_axis(), self.noise.noise_signal[ii].signal)
			# if self.start and self.stop:
			# 	ax.set_xlim(left=self.start, right=self.stop)
			# else:
			ax.set_xlim(left=min(self.input.time_data), right=max(self.input.time_data))
		if display:
			pl.show(False)
		if save:
			assert isinstance(save, str), "Please provide filename"
			pl.savefig(save+'NoiseTimeSeries.pdf')

	def plot_signal_and_noise(self, channel_index=0, save=False, display=True):
		"""
		"""
		if self.noise is not None:
			combined = self.input.input_signal[channel_index].signal + self.noise.noise_signal[channel_index].signal
		else:
			return

		fig = pl.figure()
		ax1 = pl.subplot2grid((3, 1), loc=(0, 0), rowspan=1, colspan=1)
		ax1.set_title('Signal')
		ax1.set_xticklabels('')
		ax1.set_xlim(left=min(self.input.time_data), right=max(self.input.time_data))

		ax2 = pl.subplot2grid((3, 1), loc=(1, 0), rowspan=1, colspan=1)
		ax2.set_xticklabels('')
		snr = st.signaltonoise(combined)
		ax2.set_title(r'Signal [{0}] + Noise $\Rightarrow$ SNR = {1}'.format(str(channel_index), str(snr)))
		ax2.set_xlim(left=min(self.input.time_data), right=max(self.input.time_data))
		ax3 = pl.subplot2grid((3, 1), loc=(2, 0), rowspan=1, colspan=1)
		ax3.set_title('Noise')
		ax3.set_xlabel('Time [ms]')
		ax3.set_xlim(min(self.noise.time_data), max(self.noise.time_data))

		ax1.plot(self.input.time_data, self.input.input_signal[channel_index].signal)
		ax2.plot(self.input.time_data, self.input.input_signal[channel_index].signal + self.noise.noise_signal[
			channel_index].signal)
		ax3.plot(self.noise.time_data, self.noise.noise_signal[channel_index].signal)

		if display:
			pl.show(False)
		if save:
			assert isinstance(save, str), "Please provide filename"
			pl.savefig(save+'SNR.pdf')


########################################################################################################################
class ActivityAnimator(object):
	"""
	ActivityIllustrator(spike_list, vm_list=None, populations=None, ids=None)
	Class for animating various spiking activities of SpikeList objects or populations.

	There is a single public function, animate_activity(...), that controls all the animations. For performance reasons,
	make sure to animate the activities of only a reduced number of neurons (~100 - 300~).
	Possible animations:
		raster
		mean rate
		membrane voltage
		trajectory

	Input:
		spike_list
		ids: None or a list of lists, grouped by ids (for coloring, etc.)

	Examples:
		>> ai = viz.ActivityIllustrator(spikelist)

		>> ai.animate_activity(time_interval=100, time_window=100, fps=60, frame_res=0.25, save=True,
				  			  filename="/path/to/saved/file", activities=["raster", "rate"])

	"""
	def __init__(self, spike_list, vm_list=None, populations=None, ids=None):
		self.spike_list 	= spike_list
		self.vm_list 		= vm_list
		self.populations 	= populations
		self.ids 			= ids

	def __anim_frame_raster(self, ax=None, start=None, stop=None, colors=['b'], shift_only=False, ax_props={}):
		"""
		Display the contents of the spike list for the chosen neurons as a raster plot.

		:param ax:
		:param start:
		:param stop:
		:param colors:
		:param shift_only:
		:param ax_props:
		:return:
		"""
		ax.clear()

		if not shift_only:
			self.raster_fr_data = []
			tt = self.sliced_spike_list

			if self.ids is None:
				times 	= tt.raw_data()[:, 0]
				neurons = tt.raw_data()[:, 1]
				ax.plot(times, neurons, '.', color=colors[0])
				self.raster_fr_data.append((times, neurons, '.', colors[0]))
			else:
				assert isinstance(self.ids, list), "Gids should be a list"
				for n, group_ids in enumerate(self.ids):
					group = tt.id_slice(group_ids)
					times = group.raw_data()[:, 0]
					neurons = group.raw_data()[:, 1]
					ax.plot(times, neurons, '.', color=colors[n])
					self.raster_fr_data.append((times, neurons, '.', colors[n]))
		else:
			assert self.raster_fr_data is not None and not isinstance(self.raster_fr_data, basestring), \
				"Previous raster frame data empty, something's gone south..."
			for fd in self.raster_fr_data:
				times, neurons, format_, color = fd
				ax.plot(times, neurons, '.', color=color)

		ax.set_xlim([start, stop])
		ax.set(**ax_props)

	def __anim_frame_mean_rate(self, ax=None, start=None, stop=None, colors=['b'], shift_only=False, dt=1., ax_props={}):
		"""

		:param ax:
		:param start:
		:param stop:
		:param colors:
		:param shift_only:
		:param dt:
		:param ax_props:
		:return:
		"""
		ax.clear()
		# only precompute it max_rate once, for size of rate subplot
		if self.max_rate is None:
			firing_rates = self.spike_list.firing_rate(dt, average=True)
			self.max_rate = max(firing_rates)
			self.min_rate = min(firing_rates)

		if not shift_only:
			self.rate_fr_data = []
			tt = self.sliced_spike_list

			if self.ids is None:
				time = tt.time_axis(dt)[:-1]
				rate = tt.firing_rate(dt, average=True)
				ax.plot(time, rate, color=colors[0])
				self.rate_fr_data.append((time, rate, colors[0]))
			else:
				assert isinstance(self.ids, list), "Gids should be a list"
				for n, group_ids in enumerate(self.ids):
					group = tt.id_slice(group_ids)
					time = group.time_axis(dt)[:-1]
					rate = group.firing_rate(dt, average=True)
					ax.plot(time, rate, colors[n])
					self.rate_fr_data.append((time, rate, colors[n]))
		else:
			assert self.rate_fr_data is not None and not isinstance(self.rate_fr_data, basestring), \
				"Previous rate frame data empty, something's gone south..."
			for fd in self.rate_fr_data:
				time, rate, color = fd
				ax.plot(time, rate, color=color)

		ax.set(ylim=[self.min_rate-1, self.max_rate+1], xlim=[start, stop])
		ax.set(**ax_props)

	def __plot_vm(self, ax=None, time_interval=None, with_spikes=True):
		pass

	def __plot_trace(self, ax=None, time_interval=None, dt=1.):
		pass

	def __plot_trajectory(self, variable=None):
		pass
	# def animate_trajectory(response_matrix, pca_fit_obj, interval=100, label='', ax=None, fig=None, display=True,
	# 					   save=False):
	#
	# 	if ax is None and fig is None:
	# 		fig = pl.figure()
	# 		ax = fig.add_subplot(111, projection='3d')
	# 		ax.grid(False)
	#
	# 	def animate(i):
	# 		X = pca_fit_obj.transform(response_matrix.as_array()[:, :i + 1].transpose())
	#
	# 		ax.clear()
	# 		ax.plot(X[:, 0], X[:, 1], X[:, 2], color='r', lw=2)
	# 		ax.set_title(label + r'$ - t = {0}$ / (3PCs) $= {1}$'.format(str(i),
	# 																	 str(round(np.sum(
	# 																		 pca_fit_obj.explained_variance_ratio_[:3]),
	# 																			   1))))
	#
	# 	ani = animation.FuncAnimation(fig, animate, interval=10)

	def __has_activity(self, activities, key):
		return bool(activities is not None and key in activities)

	def __step_binned_time(self, start=0., time_step=0.1, time_window=100):
		while True:
			yield (start, start + time_window - 1)
			start += time_step

	def animate_activity(self, time_interval=None, time_window=None, frame_res=0.25, fps=60, sim_res=1.0,
						 filename="animation", activities=None, colors=['b'], save=False, display=False):
		"""
		Main function that exposes all the animations. It uses animation.FuncAnimation to plot each frame separately,
		which are then saved in .mp4 format. Every call to animate_activity creates a single figure containing all the
		activities that were passed as parameters. Use the frame_res and fps parameters to create smooth animations.


		:param time_interval: time interval in msec between each frame of the animation [USEFUL ONLY WHEN DISPLAYING]
		:param time_window: size of time window for each frame
		:param frame_res: resolution of the time axis for the animation, i.e., how many if simulation resolution=1.0 and
			dt=0.25 there will be 4 frames for each second on the time axis.
		:param save:
		:param fps: frames per second for the video encoding
		:param sim_res: resolution of the simulation (for spike timing)
		:param filename:
		:param activities: list of strings; possible values: ["raster", "rate"]
		:param colors: list of matplotlib colors; numbers of colors must match the number of populations (id groups)
		:return:
		"""
		print("Started animating spiking activities...")

		def animate(i):
			# TODO check if performance can be increased by using blit=True.. redesign needed in that case
			start_t, stop_t = mwt.next() # start and stop time of spike window
			self.sliced_spike_list = self.spike_list.time_slice(start_t, stop_t)

			# only draw new plot at every simulation resolution (when new spikes can appear), otherwise just shift axes
			shift_only = bool(self.cnt % int(sim_res / frame_res) != 0)

			if self.__has_activity(activities, "raster"):
				self.__anim_frame_raster(ax_raster, start_t, stop_t, colors, shift_only,
										 {"xlabel": "Time [ms]", "ylabel": "Neurons"})

			if self.__has_activity(activities, "rate"):
				self.__anim_frame_mean_rate(ax_rate, start_t, stop_t, colors, shift_only,
											ax_props={"xlabel": "Time", "ylabel": "Rate [Hz]"})

			self.cnt += 1

		# some initial checks
		if self.ids is not None and len(self.ids) > 1:
			assert len(self.ids) == len(colors), "Number of population groups and #colors must match!"

		time_axis 	= self.spike_list.time_axis(time_bin=frame_res)
		nr_frames 	= int(math.floor((len(time_axis) * frame_res) - time_window + 1) / frame_res)
		mwt 		= self.__step_binned_time(min(time_axis), frame_res, time_window)
		self.cnt 	= 0  # counter for #calls to animation function (animate)

		fig = pl.figure()
		if self.__has_activity(activities, "raster"):
			ax_raster 	= pl.subplot2grid((30, 1), (0, 0), rowspan=23, colspan=1)
		if self.__has_activity(activities, "rate"):
			ax_rate 	= pl.subplot2grid((30, 1), (24, 0), rowspan=5, colspan=1)

		self.raster_fr_data = None
		self.rate_fr_data 	= None
		self.max_rate 		= None
		self.sliced_spike_list = None

		# run animation
		ani = animation.FuncAnimation(fig, animate, interval=time_interval, frames=nr_frames - 1, repeat=False)
		if save:
			ani.save(filename + '.mp4', fps=fps, extra_args=['-vcodec', 'libx264'])
		if display:
			pl.show(False)


# def plot_vm(vm, times, ax, color, lw=1, v_reset=-70., v_th=-50.):
# 	"""
#
# 	:param vm:
# 	:param times:
# 	:param ax:
# 	:param color:
# 	:param lw:
# 	:param v_reset:
# 	:param v_th:
# 	:return:
# 	"""
# 	ax.plot(times, vm, c=color)
# 	#ax.set_xlabel('Time [ms]')
# 	ax.set_ylabel(r'$V_{m}$')
# 	idxs = vm.argsort()
# 	possible_spike_times = [t for t in idxs if (t < len(vm) - 1) and (vm[t + 1] == v_reset) and (vm[t] != v_reset)]
# 	ax.vlines(times[possible_spike_times], v_th, 50., color='k')
# 	ax.set_ylim(min(vm) - 5., 10.)

# def animate_raster(spike_list, gids, window_size, display=False, save=False):
# 	"""
#
# 	:param spike_list:
# 	:param gids:
# 	:param window_size:
# 	:return:
# 	"""
# 	time_axis = spike_list.time_axis(time_bin=1.)
# 	mw = signals.moving_window(time_axis, window_size)
# 	fig = pl.figure()
# 	ax1 = pl.subplot2grid((30, 1), (0, 0), rowspan=23, colspan=1)
# 	ax2 = pl.subplot2grid((30, 1), (24, 0), rowspan=5, colspan=1, sharex=ax1)
#
# 	def animate(i):
# 		ax1.clear()
# 		ax2.clear()
# 		time_window = mw.next()
# 		spike_plot = SpikePlots(spike_list, start=min(time_window), stop=max(time_window), N=10000)
# 		spike_plot.dot_display(gids=gids, colors=['b', 'r'], with_rate=True, dt=1.0, display=False, ax=[ax1, ax2],
# 		                       fig=fig, save=False)
#
# 	ani = animation.FuncAnimation(fig, animate, interval=10)
#
# 	if display:
# 		pl.show(False)
# 	if save:
# 		ani.save('{0}_animation.gif'.format(save), fps=1000)


def plot_trajectory(response_matrix, pca_fit_obj=None, label='', color='r', ax=None, display=True, save=False):
	"""

	:param response_matrix:
	:param pca_fit_obj:
	:param label:
	:param color:
	:param ax:
	:param display:
	:param save:
	:return:
	"""
	if ax is None:
		fig = pl.figure()
		ax = fig.add_subplot(111, projection='3d')
	if pca_fit_obj is None:
		pca_fit_obj = sk.PCA(n_components=np.shape(response_matrix)[0])
	if not hasattr(pca_fit_obj, "explained_variance_ratio_"):
		pca_fit_obj.fit(response_matrix.T)
	X = pca_fit_obj.transform(response_matrix.transpose())
	print "Explained Variance (first 3 components): %s" % str(pca_fit_obj.explained_variance_ratio_)

	ax.clear()
	ax.plot(X[:, 0], X[:, 1], X[:, 2], color=color, lw=2)
	ax.set_title(label + r'$ - (3PCs) $= {0}$'.format(str(round(np.sum(pca_fit_obj.explained_variance_ratio_[:3]),
	                                                                       1))))
	#ax.grid()
	if display:
		pl.show(False)
	if save:
		pl.savefig('{0}_trajectory.pdf'.format(save))


# def animate_trajectory(response_matrix, pca_fit_obj, interval=100, label='', ax=None, fig=None, display=True,
#                        save=False):
#
# 	if ax is None and fig is None:
# 		fig = pl.figure()
# 		ax = fig.add_subplot(111, projection='3d')
# 		ax.grid(False)
#
# 	def animate(i):
# 		X = pca_fit_obj.transform(response_matrix.as_array()[:, :i+1].transpose())
#
# 		ax.clear()
# 		ax.plot(X[:, 0], X[:, 1], X[:, 2], color='r', lw=2)
# 		ax.set_title(label + r'$ - t = {0}$ / (3PCs) $= {1}$'.format(str(i),
#                     str(round(np.sum(pca_fit_obj.explained_variance_ratio_[:3]), 1))))
#
# 	ani = animation.FuncAnimation(fig, animate, interval=10)
#
# 	if display:
# 		pl.show(False)
# 	if save:
# 		ani.save('{0}_animation.gif'.format(save), fps=1000)


def plot_response(population, ids=None, spiking_activity=None, display=True, save=False):
	"""
	Plot activity matrix and the traces for specified ids, along with the spikes recorded with the spike detector
	:param population: Population object to which the responses belong
	:param ids: ids of the neurons to plot independently
	:param spiking_activity: if None the population's spiking_activity will be used instead, otherwise, should be a
	SpikeList object
	:param display:
	:param save:
	"""
	assert (population.decoding_layer is not None), "Population must have a decoding layer"
	assert (not signals.empty(population.decoding_layer.activity)), "Extract population activity first"
	fig1 = pl.figure()
	fig1.suptitle(r"Population {0} activity".format(population.name))
	dt = population.decoding_layer.activity[0].dt

	for state_idx, n_state in enumerate(population.decoding_layer.state_variables):
		globals()['ax_{0}'.format(state_idx)] = fig1.add_subplot(len(population.decoding_layer.state_variables),
		                                                         1, state_idx + 1)
		plt = globals()['ax_{0}'.format(state_idx)].imshow(population.decoding_layer.activity[state_idx].as_array(),
		                                                   aspect='auto', interpolation='nearest')
		divider = make_axes_locatable(globals()['ax_{0}'.format(state_idx)])
		cax = divider.append_axes("right", "5%", pad="4%")
		cbar = fig1.colorbar(plt, cax=cax)
		x_ticks = globals()['ax_{0}'.format(state_idx)].get_xticks()
		globals()['ax_{0}'.format(state_idx)].set_xticklabels([str(x * dt) for x in x_ticks])
		globals()['ax_{0}'.format(state_idx)].set_xlabel(r"Time [ms]")
		globals()['ax_{0}'.format(state_idx)].set_ylabel("Neuron")
		globals()['ax_{0}'.format(state_idx)].set_title("{0}".format(n_state))

	if not signals.empty(population.spiking_activity) or spiking_activity is not None:
		if spiking_activity is not None:
			sl = spiking_activity
		else:
			sl = population.spiking_activity

		if ids is not None:
			assert (isinstance(ids, list) or isinstance(ids, np.ndarray)), "Provide list or array of neuron gids"
		else:
			ids = np.random.permutation(sl.id_list)[:5]
		list_idx = ids - min(population.gids)#np.where(np.sort(np.array(population.gids)) == neuron_idx)[
		# 0][0]

		for state_idx, n_state in enumerate(population.decoding_layer.state_variables):
			globals()['fig_{0}'.format(state_idx)] = pl.figure()
			label = "population_{0}_variable_{1}".format(str(population.name), str(n_state))
			activity = population.decoding_layer.activity[state_idx]
			globals()['fig_{0}'.format(state_idx)].suptitle(r"Population {0} [${1}$] - $dt={2}$".format(str(
				population.name), str(n_state), str(activity.dt)))
			time_axis = activity.time_axis()
			for iid, neuron_id in enumerate(ids):
				spk = sl.spiketrains[int(neuron_id)]
				globals()['ax_1{0}'.format(str(iid))] = globals()['fig_{0}'.format(state_idx)].add_subplot(len(ids),
				                                                                        1, iid+1)
				globals()['ax_1{0}'.format(str(iid))].plot(time_axis, activity.as_array()[list_idx[iid], :])
				if not signals.empty(population.decoding_layer.state_matrix[state_idx]) and not signals.empty(
						population.decoding_layer.sampled_times):
					state_mat = population.decoding_layer.state_matrix[state_idx]
					if isinstance(state_mat, np.ndarray):
						globals()['ax_1{0}'.format(str(iid))].plot(population.decoding_layer.sampled_times,
						                                           state_mat[list_idx[iid], :], 'or')
				ylims = globals()['ax_1{0}'.format(str(iid))].get_ylim()
				for idxx, n_spk in enumerate(spk.spike_times):
					globals()['ax_1{0}'.format(str(iid))].vlines(n_spk, ylims[0], ylims[1], lw=2)
				# globals()['ax_1{0}'.format(str(iid))].plot(spk.spike_times, np.ones_like(spk.spike_times), 'o')
				globals()['ax_1{0}'.format(str(iid))].set_title(r"gid {0} [{1}]".format(str(neuron_id), population.name))
				globals()['ax_1{0}'.format(str(iid))].set_xlim([activity.t_start, activity.t_stop])
				globals()['ax_1{0}'.format(str(iid))].set_ylim(ylims)
			if save:
				globals()['fig_{0}'.format(state_idx)].savefig(save + label + '.pdf')

	if display:
		pl.show(block=False)
	if save:
		fig1.savefig(save+population.name+'_ResponseMatrix.pdf')


def plot_fmf(t_axis, fmf, ax, label='', display=True, save=False):
	"""
	"""
	if (ax is not None) and (not isinstance(ax, mpl.axes.Axes)):
		raise ValueError('ax must be matplotlib.axes.Axes instance.')
	if ax is None:
		fig_fmf, ax_fmf = pl.subplots()
		fig_fmf.suptitle(r'${0}$ - Fading Memory Function'.format(str(label)))
	else:
		ax_fmf = ax
		ax_fmf.set_title(r'${0}$ - Fading Memory Function'.format(str(label)))

	dx = np.min(np.diff(t_axis))
	x_axis = np.arange(len(fmf)).astype(float) * dx

	ax_fmf.plot(x_axis, fmf)

	ix = x_axis
	iy = fmf
	a = min(x_axis)
	b = max(x_axis)
	dx = np.min(np.diff(t_axis))
	verts = [(a, 0)] + list(zip(ix, iy)) + [(b, 0)]
	poly = Polygon(verts, facecolor='0.9', edgecolor='0.5')
	ax_fmf.add_patch(poly)

	y_points = [0.15, 0.30, 0.45]
	ax_fmf.text(0.25 * (a + b), y_points[0], r"$\int m(\tau) d\tau = {0} (trpz)$".format(str(np.trapz(fmf, dx=dx))),
	            horizontalalignment='center', fontsize=14)
	ax_fmf.text(0.25 * (a + b), y_points[1], r"$\int m(\tau) d\tau = {0} (simps)$".format(str(integ.simps(fmf,
	                                                                                                    dx=dx))),
	            horizontalalignment='center', fontsize=14)
	# txt =
	ax_fmf.text(0.25 * (a + b), y_points[2], r"$\sum_{%s}^{%s} m(\tau) = %s$" % (r'\tau={0}'.format(str(a)), str(b),
	                                                                             str(np.sum(fmf[1:])*dx)),
	            horizontalalignment='center', fontsize=14)

	ax_fmf.set_xlabel(r'$\tau$')
	ax_fmf.set_ylabel(r'$m(\tau)$')
	ax_fmf.set_ylim([0., 1.])
	if display:
		pl.show(False)
	if save:
		pl.savefig(save+label+'_FMF.pdf')


def plot_state_matrix(state_mat, stim_labels, ax=None, label='', display=True, save=False):
	if (ax is not None) and (not isinstance(ax, mpl.axes.Axes)):
		raise ValueError('ax must be matplotlib.axes.Axes instance.')
	if ax is None:
		fig, ax = pl.subplots()
		fig.suptitle(r'${0}$ - State Matrix'.format(str(label)))
	else:
		ax.set_title(r'${0}$ - State Matrix'.format(str(label)))

	xtick_labels = list(signals.iterate_obj_list(stim_labels))
	#step_size = len(xtick_labels)
	plt = ax.imshow(state_mat, aspect='auto', interpolation='nearest')
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", "5%", pad="4%")
	pl.colorbar(plt, cax=cax)
	ax.set_xticks(range(len(xtick_labels)))
	ax.set_xticklabels(list(signals.iterate_obj_list(stim_labels)))
	ax.set_xlabel(r"Time [sts]")
	ax.set_ylabel("Neuron")

	if display:
		pl.show(False)
	if save:
		pl.savefig(save+'_state_matrix_{0}.pdf'.format(str(label)))


def get_cmap(N, cmap='hsv'):
	"""
	Returns a function that maps each index in 0, 1, ... N-1 to a distinct
	RGB color.
	"""
	color_norm = colors.Normalize(vmin=0, vmax = N-1)
	scalar_map = cmx.ScalarMappable(norm=color_norm, cmap=cmap)

	def map_index_to_rgb_color(index):
		return scalar_map.to_rgba(index)

	return map_index_to_rgb_color


def plot_readout_performance(results_dict, display=True, save=False):
	"""
	"""
	for pop_name, pop_readouts in results_dict.items():

		for rset_name, rset in pop_readouts.items():

			if rset.has_key("sample_times"):
				fig = pl.figure()
				fig.suptitle(pop_name + rset_name)
				ax1 = fig.add_subplot(211)
				ax2 = fig.add_subplot(212, sharex=ax1)

				xlabels = rset['sample_times']
				xvalues = np.arange(len(xlabels))
				max_items = len(rset['sample_0']['pb_cc'][0])
				baseline = np.ones_like(xvalues) * 1./max_items
				readout_labels = rset['sample_0']['labels']

				for n, lab in enumerate(readout_labels):
					performances = [rset[k]['performance'][n] for k in rset.keys() if k[-1].isdigit()]
					stability = [rset[k]['norm_wout'][n] for k in rset.keys() if k[-1].isdigit()]
					ax1.plot(xvalues, performances, '-', label=lab)
					ax1.plot(xvalues, performances, 'o')
					ax1.plot(xvalues, baseline, '--r')
					ax2.plot(xvalues, stability, '-')
					ax2.plot(xvalues, stability, 'o')
				ax1.set_xlabel('')
				ax1.set_ylabel(r'$\mathrm{Perf}$')
				ax1.set_xticks(xvalues)
				ax1.set_xticklabels(xlabels)
				ax2.set_xlabel(r'Time [ms]')
				ax2.set_xticks(xvalues)
				ax2.set_xticklabels(xlabels)
				ax2.set_ylabel(r'$|W^{\mathrm{out}}|$')
				ax1.legend()
				if save:
					save_path = save + pop_name + rset_name
					fig.savefig(save_path + 'ReadoutPerformance1.pdf')
			else:
				fig = pl.figure()
				fig.suptitle(pop_name+rset_name)
				ax1 = pl.subplot2grid((4, 7), (0, 0), rowspan=4, colspan=4)
				ax2 = pl.subplot2grid((4, 7), (0, 5), rowspan=2, colspan=2)
				ax3 = pl.subplot2grid((4, 7), (2, 5), rowspan=2, colspan=2)

				xlabels = rset['labels']
				xvalues = np.arange(len(xlabels))
				max_items = len(rset['pb_cc'][0])
				baseline = np.ones_like(xvalues) * 1./max_items

				ax1.plot(xvalues, rset['performance'], '-', c='g')
				ax1.plot(xvalues, rset['performance'], 'o', c='b')
				ax1.plot(xvalues, baseline, '--r')
				ax1.set_xlabel(r'Readout')
				ax1.set_ylabel(r'$\mathrm{Perf}$')
				ax1.set_xticks(xvalues)
				ax1.set_xticklabels(xlabels)

				ax2.plot(xvalues, rset['hamming_loss'], '-', c='g')
				ax2.plot(xvalues, rset['hamming_loss'], 'o', c='b')
				ax2.set_xlabel(r'')
				ax2.set_ylabel(r'$\mathrm{L}_{\mathrm{Hamming}}$')
				ax2.set_xticks(xvalues)
				ax2.set_xticklabels([])

				ax3.plot(xvalues, rset['MSE'], '-', c='g')
				ax3.plot(xvalues, rset['MSE'], 'o', c='b')
				ax3.set_xlabel(r'')
				ax3.set_ylabel(r'$\mathrm{MSE}$')
				ax3.set_xticks(xvalues)
				ax3.set_xticklabels([])

				fig2 = pl.figure()
				fig2.suptitle(pop_name + rset_name)
				ax21 = pl.subplot2grid((3, 8), (1, 0), rowspan=1, colspan=2)
				ax22 = pl.subplot2grid((3, 8), (1, 3), rowspan=1, colspan=2)
				ax23 = pl.subplot2grid((3, 8), (1, 6), rowspan=1, colspan=2)
				# ax21 = fig2.add_subplot(131)
				# ax22 = fig2.add_subplot(132)
				# ax23 = fig2.add_subplot(133)

				lefts = np.arange(len(rset['pb_cc'][0]))
				ax21.bar(lefts, rset['pb_cc'][0])
				ax21.set_ylabel(r'$0$-lag pbcc')
				ax21.set_xlabel(r'Item')
				ax21.set_xticks([])

				ax22.plot(xvalues, rset['raw_MAE'], '--', c='g')
				ax22.plot(xvalues, rset['raw_MAE'], 'o', c='b')
				ax22.set_xlabel(r'Readout')
				ax22.set_ylabel(r'$\mathrm{MAE}$')
				ax22.set_xticks(xvalues)
				ax22.set_xticklabels(xlabels)

				ax23.plot(xvalues, rset['norm_wout'], '--', c='g')
				ax23.plot(xvalues, rset['norm_wout'], 'o', c='b')
				ax23.set_xlabel(r'Readout')
				ax23.set_ylabel(r'$|W^{\mathrm{out}}|$')
				ax23.set_xticks(xvalues)
				ax23.set_xticklabels(xlabels)
				if save:
					save_path = save + pop_name + rset_name
					fig.savefig(save_path + 'ReadoutPerformance1.pdf')
					fig2.savefig(save_path + 'ReadoutPerformance2.pdf')

	if display:
		pl.show(block=False)


def extract_encoder_connectivity(enc_layer, net, display=True, save=False):
	"""
	Extract and plot encoding layer connections
	"""
	if len(np.unique(enc_layer.connection_types)) == 1:
		tgets = list(signals.iterate_obj_list([list(signals.iterate_obj_list(n.gids)) for n in net.populations]))
		srces = list(itertools.chain(*[n.gids for n in enc_layer.generators][0]))
		enc_layer.extract_synaptic_weights(srces, tgets, syn_name='Gen_Net')
		enc_layer.extract_synaptic_delays(srces, tgets, syn_name='Gen_Net')
	elif len(enc_layer.connections) != len(np.unique(enc_layer.connection_types)):
		for con_idx, n_con in enumerate(enc_layer.connections):
			src_name = n_con[1]
			tget_name = n_con[0]
			syn_name = enc_layer.connection_types[con_idx] + str(con_idx)
			if src_name in net.population_names:
				src_gids = net.populations[net.population_names.index(src_name)].gids
			elif src_name in list([n.name for n in net.merged_populations]):
				merged_populations = list([n.name for n in net.merged_populations])
				src_gids = net.merged_populations[merged_populations.index(src_name)].gids
			elif src_name in enc_layer.encoder_names:
				src_gids = enc_layer.encoders[enc_layer.encoder_names.index(src_name)].gids
			else:
				gen_names = [x.name[0].split('0')[0] for x in enc_layer.generators]
				src_gids = list(itertools.chain(*enc_layer.generators[gen_names.index(src_name)].gids))
			if tget_name in net.population_names:
				tget_gids = net.populations[net.population_names.index(tget_name)].gids
			elif tget_name in list([n.name for n in net.merged_populations]):
				merged_populations = list([n.name for n in net.merged_populations])
				tget_gids = net.merged_populations[merged_populations.index(tget_name)].gids
			elif tget_name in enc_layer.encoder_names:
				tget_gids = enc_layer.encoders[enc_layer.encoder_names.index(tget_name)].gids
			else:
				gen_names = [x.name[0].split('0')[0] for x in enc_layer.generators]
				tget_gids = list(itertools.chain(*enc_layer.generators[gen_names.index(tget_name)].gids))
			enc_layer.extract_synaptic_weights(src_gids, tget_gids, syn_name=syn_name)
			enc_layer.extract_synaptic_delays(src_gids, tget_gids, syn_name=syn_name)
	else:
		enc_layer.extract_synaptic_weights()
		enc_layer.extract_synaptic_delays()

	for connection, matrix in enc_layer.connection_weights.items():
		globals()['fig_encW_{0}'.format(connection)], globals()['ax_encW_{0}'.format(connection)] = pl.subplots()
		source_name, target_name = connection.split('_')
		plot_connectivity_matrix(matrix.todense(), source_name, target_name, label=connection + 'W',
		                         ax=globals()['ax_encW_{0}'.format(connection)],
		                         display=display, save=save)
	for connection, matrix in enc_layer.connection_delays.items():
		globals()['fig_encd_{0}'.format(connection)], globals()['ax_encd_{0}'.format(connection)] = pl.subplots()
		source_name, target_name = connection.split('_')
		plot_connectivity_matrix(matrix.todense(), source_name, target_name, label=connection + 'd',
		                         ax=globals()['ax_encd_{0}'.format(connection)],
		                         display=display, save=save)
#
# 	def rate_map(self):


def progress_bar(progress):
	"""
	Prints a progress bar to stdout.

	Inputs:
		progress - a float between 0. and 1.

	Example:
		>> progress_bar(0.7)
			|===================================               |
	"""
	progressConditionStr = "ERROR: The argument of function visualization.progress_bar(...) must be a float between " \
	                       "0. and 1.!"
	assert (type(progress) == float) and (progress >= 0.) and (progress <= 1.), progressConditionStr
	length = 50
	filled = int(round(length * progress))
	print "|" + "=" * filled + " " * (length - filled) + "|\r",
	sys.stdout.flush()


def plot_target_out(target, output, label='', display=False, save=False):
	"""

	:param target:
	:param output:
	:param label:
	:param display:
	:param save:
	:return:
	"""
	fig2, ax2 = pl.subplots()
	fig2.suptitle(label)
	if output.shape == target.shape:
		tg = target[0]
		oo = output[0]
	else:
		tg = target[0]
		oo = output[:, 0]
	ax2ins = zoomed_inset_axes(ax2, 0.5, loc=1)
	ax2ins.plot(tg, c='r')
	ax2ins.plot(oo, c='b')
	ax2ins.set_xlim([100, 200])
	ax2ins.set_ylim([np.min(tg), np.max(tg)])

	mark_inset(ax2, ax2ins, loc1=2, loc2=4, fc="none", ec="0.5")

	pl1 = ax2.plot(tg, c='r', label='target')
	pl2 = ax2.plot(oo, c='b', label='output')
	ax2.set_xlabel('Time [ms]')
	ax2.set_ylabel('u(t)')
	ax2.legend(loc=3)
	if display:
		pl.show(False)
	if save:
		pl.savefig(save + label + '_TargetOut.pdf')


def plot_connectivity_matrix(matrix, source_name, target_name, label='', ax=None,
                             display=True, save=False):
	"""

	:param matrix:
	:param source_name:
	:param source_gids:
	:param target_name:
	:param target_gids:
	:return:
	"""
	if (ax is not None) and (not isinstance(ax, mpl.axes.Axes)):
		raise ValueError('ax must be matplotlib.axes.Axes instance.')

	if len(label.split('_')) == 2:
		title = label.split('_')[0] + '-' + label.split('_')[1]
		label = title
	else:
		title = label
	if ax is None:
		fig, ax = pl.subplots()
		fig.suptitle(r'${0}$'.format(str(title)))
	else:
		ax.set_title(r'${0}$'.format(str(title)))

	plt1 = ax.imshow(matrix, interpolation='nearest', aspect='auto', extent=None, cmap='jet')
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", "5%", pad="3%")
	pl.colorbar(plt1, cax=cax)
	ax.set_title(label)
	ax.set_xlabel('Source=' + str(source_name))
	ax.set_ylabel('Target=' + str(target_name))
	if display:
		pl.show(False)
	if save:
		pl.savefig(save + '{0}connectivityMatrix.pdf'.format(label))


def plot_response_activity(spike_list, input_stimulus, start=None, stop=None):
	"""
	Plot population responses to stimuli (spiking activity)
	:param spike_list:
	:param input_stimulus:
	:return:
	"""
	fig = pl.figure()
	ax1 = pl.subplot2grid((12, 1), (0, 0), rowspan=6, colspan=1)
	ax2 = pl.subplot2grid((12, 1), (7, 0), rowspan=2, colspan=1, sharex=ax1)
	ax3 = pl.subplot2grid((12, 1), (10, 0), rowspan=2, colspan=1, sharex=ax1)

	rp = SpikePlots(spike_list, start, stop)
	plot_props = {'xlabel': 'Time [ms]', 'ylabel': 'Neuron', 'linewidth': 1.0, 'linestyle': '-'}
	rp.dot_display(ax=[ax1, ax2], with_rate=True, display=False, save=False, **plot_props)

	rp.mark_events(ax1, input_stimulus, start, stop)
	rp.mark_events(ax2, input_stimulus, start, stop)


def plot_isi_data(results, data_label, color_map='jet', location=0, fig_handles=None, axes=None, display=True, \
                                                                                                       save=False):
	"""

	:param results:
	:param label:
	:return:
	"""
	keys = ['isi', 'cvs', 'lvs', 'lvRs', 'ents', 'iR', 'cvs_log', 'isi_5p', 'ai']
	# ISI histograms
	if fig_handles is None and axes is None:
		fig1, axes_histograms = isi_analysis_histogram_axes(label=data_label)
		fig2 = pl.figure()
	else:
		fig1 = fig_handles[0]
		fig2 = fig_handles[1]
		axes_histograms = axes
	data_list = [results[k] for k in keys]
	args = [{'xlabel': r'{0}'.format(k), 'ylabel': r'Frequency'} for k in keys]
	bins = [100 for k in keys]
	plot_histograms(axes_histograms, data_list, bins, args, cmap=color_map)

	# ISI Summary statistics
	fig2.suptitle(data_label + 'ISI statistics')
	summary_statistics(data_list, labels=keys, loc=location, fig=fig2, cmap=color_map)

	if display:
		pl.show(False)
	if save:
		fig1.savefig(save + '{0}_isi_histograms.pdf'.format(str(data_label)))
		fig2.savefig(save + '{0}_isi_summary.pdf'.format(str(data_label)))


def plot_synchrony_measures(results, label='', time_resolved=False, epochs=None, color_map='jet', display=True,
                            save=False):
	"""

	:param results:
	:param time_resolved:
	:return:
	"""
	# Synchrony distance matrices
	fig3 = pl.figure()
	fig3.suptitle('{0} - Pairwise Distances'.format(str(label)))
	ax31 = fig3.add_subplot(221)
	ax32 = fig3.add_subplot(222)
	ax33 = fig3.add_subplot(223)
	ax34 = fig3.add_subplot(224)
	image_arrays = [results['d_vr'], results['ISI_distance_matrix'], results['SPIKE_distance_matrix'],
	                results['SPIKE_sync_matrix']]
	plot_2d_parscans(image_arrays=image_arrays, axis=[ax31, ax32, ax33, ax34],
	                     fig_handle=fig3, labels=[r'$D_{vR}$', r'$D_{ISI}$', r'$D_{SPIKE}$', r'$D_{SPIKE_{S}}$'])
	if time_resolved:
		# Time resolved synchrony
		fig4 = pl.figure()
		fig4.suptitle('{0} - Time-resolved synchrony'.format(str(label)))
		ax1 = fig4.add_subplot(311)
		ax2 = fig4.add_subplot(312, sharex=ax1)
		ax3 = fig4.add_subplot(313, sharex=ax1)
		if epochs is not None:
			mark_epochs(ax1, epochs, color_map)

		x, y = results['SPIKE_sync_profile'].get_plottable_data()
		ax1.plot(x, y, '-g', alpha=0.4)
		ax1.set_ylabel(r'$S_{\mathrm{SPIKE_{s}}}(t)$')
		ax1.plot(x, signals.smooth(y, window_len=100, window='hamming'), '-g', lw=2.5)

		x3, y3 = results['ISI_profile'].get_plottable_data()
		ax2.plot(x3, y3, '-b', alpha=0.4)
		ax2.plot(x3, signals.smooth(y3, window_len=100, window='hamming'), '-b', lw=2.5)
		ax2.set_ylabel(r'$d_{\mathrm{ISI}}(t)$')

		x5, y5 = results['SPIKE_profile'].get_plottable_data()
		ax3.plot(x5, y5, '-k', alpha=0.4)
		ax3.plot(x5, signals.smooth(y5, window_len=100, window='hamming'), '-k', lw=2.5)
		ax3.set_ylabel(r'$d_{\mathrm{SPIKE}}(t)$')

	if display:
		pl.show(False)
	if save:
		fig3.savefig(save + '{0}_distance_matrices.pdf'.format(str(label)))
		if time_resolved:
			fig4.savefig(save + '{0}_time_resolved_sync.pdf'.format(str(label)))


def plot_averaged_time_resolved(results, spike_list, label='', epochs=None, color_map='jet', display=True,
                                  save=False):
	"""

	:param results:
	:return:
	"""
	# time resolved regularity
	fig5 = pl.figure()
	fig5.suptitle('{0} - Time-resolved regularity'.format(str(label)))
	stats = ['isi_5p_profile', 'cvs_profile', 'cvs_log_profile', 'lvs_profile', 'iR_profile', 'ents_profile']
	cm = get_cmap(len(stats), color_map)
	for idx, n in enumerate(stats):
		globals()['ax5{0}'.format(str(idx))] = fig5.add_subplot(len(stats), 1, idx + 1)
		data_mean = np.array([results[n][i][0] for i in range(len(results[n]))])
		data_std = np.array([results[n][i][1] for i in range(len(results[n]))])
		t_axis = np.linspace(spike_list.t_start, spike_list.t_stop, len(data_mean))
		globals()['ax5{0}'.format(str(idx))].plot(t_axis, data_mean, c=cm(idx), lw=2.5)
		globals()['ax5{0}'.format(str(idx))].fill_between(t_axis, data_mean - data_std, data_mean +
		                                                  data_std, facecolor=cm(idx), alpha=0.2)
		globals()['ax5{0}'.format(str(idx))].set_ylabel(n)
		globals()['ax5{0}'.format(str(idx))].set_xlabel('Time [ms]')
		globals()['ax5{0}'.format(str(idx))].set_xlim(spike_list.time_parameters())
		if epochs is not None:
			mark_epochs(globals()['ax5{0}'.format(str(idx))], epochs, color_map)

	# activity plots
	fig6 = pl.figure()
	fig6.suptitle('{0} - Activity Analysis'.format(str(label)))
	if not "dimensionality_profile" in results.keys():
		ax61 = pl.subplot2grid((25, 1), (0, 0), rowspan=20, colspan=1)
		ax62 = pl.subplot2grid((25, 1), (20, 0), rowspan=5, colspan=1, sharex=ax61)
	else:
		ax61 = pl.subplot2grid((24, 1), (0, 0), rowspan=20, colspan=1)
		ax62 = pl.subplot2grid((24, 1), (20, 0), rowspan=2, colspan=1, sharex=ax61)
		ax63 = pl.subplot2grid((24, 1), (22, 0), rowspan=2, colspan=1, sharex=ax61)
	plot_raster(spike_list, 1., ax61, **{'color': 'k', 'alpha': 0.8, 'marker': '|', 'markersize': 2})
	stats = ['ffs_profile']
	if "dimensionality_profile" in results.keys():
		stats.append("dimensionality_profile")

	cm = get_cmap(len(stats), color_map)
	for idx, n in enumerate(stats):
		if n != "dimensionality_profile":
			data_mean = np.array([results[n][i][0] for i in range(len(results[n]))])
			data_std = np.array([results[n][i][1] for i in range(len(results[n]))])
			t_axis = np.linspace(spike_list.t_start, spike_list.t_stop, len(data_mean))
			ax62.plot(t_axis, data_mean, c=cm(idx), lw=2.5)
			ax62.fill_between(t_axis, data_mean - data_std, data_mean +
			                  data_std, facecolor=cm(idx), alpha=0.2)
			ax62.set_ylabel(r'$\mathrm{FF}$')
			ax62.set_xlabel('Time [ms]')
			ax62.set_xlim(spike_list.time_parameters())
		else:
			data_mean = np.array(results[n])
			t_axis = np.linspace(spike_list.t_start, spike_list.t_stop, len(data_mean))
			ax63.plot(t_axis, data_mean, c=cm(idx), lw=2.5)
			ax63.set_ylabel(r'$\lambda_{\mathrm{Eff}}$')
			ax63.set_xlabel('Time [ms]')
			ax63.set_xlim(spike_list.time_parameters())
	if epochs is not None:
		mark_epochs(ax61, epochs, color_map)
		mark_epochs(ax62, epochs, color_map)
		if "dimensionality_profile" in results.keys():
			mark_epochs(ax63, epochs, color_map)

	if display:
		pl.show(False)
	if save:
		fig5.savefig(save + '{0}_time_resolved_reg.pdf'.format(str(label)))
		fig6.savefig(save + '{0}_activity_analysis.pdf'.format(str(label)))


def plot_dimensionality(results, pca_obj, rotated_data, data_label='', display=True, save=False):
	fig7 = pl.figure()
	ax71 = fig7.add_subplot(121, projection='3d')
	ax71.grid(False)
	ax72 = fig7.add_subplot(122)

	ax71.plot(rotated_data[:, 0], rotated_data[:, 1], rotated_data[:, 2], color='r', lw=2, alpha=0.8)
	ax71.set_title(r'${0} - (3 PCs) = {1}$'.format(data_label, str(round(np.sum(
		pca_obj.explained_variance_ratio_[:3]), 1))))
	ax72.plot(pca_obj.explained_variance_ratio_, 'ob')
	ax72.plot(pca_obj.explained_variance_ratio_, '-b')
	ax72.plot(np.ones_like(pca_obj.explained_variance_ratio_) * results['dimensionality'], np.linspace(0.,
	                        np.max(pca_obj.explained_variance_ratio_), len(pca_obj.explained_variance_ratio_)),
	          '--r', lw=2.5)
	ax72.set_xlabel(r'PC')
	ax72.set_ylabel(r'Variance Explained (%)')
	ax72.set_xlim([0, round(results['dimensionality']) * 2])
	ax72.set_ylim([0, np.max(pca_obj.explained_variance_ratio_)])


def plot_synaptic_currents(I_ex, I_in, time_axis):
	fig, ax = pl.subplots()
	ax.plot(time_axis, I_ex, 'b')
	ax.plot(time_axis, I_in, 'r')
	ax.plot(time_axis, np.mean(I_ex) * np.ones_like(I_ex), 'b--')
	ax.plot(time_axis, np.mean(I_in) * np.ones_like(I_in), 'r--')
	ax.plot(time_axis, np.abs(I_ex) - np.abs(I_in), c='gray')
	ax.plot(time_axis, np.mean(np.abs(I_ex) - np.abs(I_in))*np.ones_like(I_ex), '--', c='gray')


def pretty_raster(global_spike_list, analysis_interval, sub_pop_gids=None, n_total_neurons=10):
	"""
	Simple line raster to plot a subset of the populations (for publication)
	:return:
	"""
	plot_list = global_spike_list.time_slice(t_start=analysis_interval[0], t_stop=analysis_interval[1])
	new_ids = np.intersect1d(plot_list.select_ids("cell.mean_rate() > 0"),
	                         plot_list.select_ids("cell.mean_rate() < 100"))

	fig = pl.figure()
	# pl.axis('off')
	ax = fig.add_subplot(111)#, frameon=False)
	if sub_pop_gids is not None:
		assert (isinstance(sub_pop_gids, list)), "Provide a list of lists of gids"
		assert (len(sub_pop_gids) == 2), "Only 2 populations are currently covered"
		lenghts = map(len, sub_pop_gids)
		sample_neurons = [(n_total_neurons * n) / np.sum(lenghts) for n in lenghts]
		# id_ratios = float(min(lenghts)) / float(max(lenghts))
		# sample_neurons = [n_total_neurons * id_ratios, n_total_neurons * (1 - id_ratios)]

		neurons_1 = []
		neurons_2 = []
		while len(neurons_1) != sample_neurons[0] or len(neurons_2) != sample_neurons[1]:
			chosen = np.random.choice(new_ids, size=n_total_neurons, replace=False)
			neurons_1 = [x for x in chosen if x in sub_pop_gids[0]]
			neurons_2 = [x for x in chosen if x in sub_pop_gids[1]]
			if len(neurons_1) > sample_neurons[0]:
				neurons_1 = neurons_1[:sample_neurons[0]]
			if len(neurons_2) > sample_neurons[1]:
				neurons_2 = neurons_2[:sample_neurons[1]]

	else:
		chosen_ids = np.random.permutation(new_ids)[:n_total_neurons]
		new_list = plot_list.id_slice(chosen_ids)

		neuron_ids = [np.where(x == np.unique(new_list.raw_data()[:, 1]))[0][0] for x in new_list.raw_data()[:, 1]]
		tmp = [(neuron_ids[idx], time) for idx, time in enumerate(new_list.raw_data()[:, 0])]

	for idx, n in enumerate(tmp):
		ax.vlines(n[1] - analysis_interval[0], n[0] - 0.5, n[0] + 0.5, **{'color': 'k', 'lw': 1.})
	ax.set_ylim(-0.5, n_total_neurons - 0.5)
	ax.set_xlim(0., analysis_interval[1] - analysis_interval[0])
	ax.grid(False)
	# ax.set_xticks([])
	# ax.set_yticks([])
	pl.show()