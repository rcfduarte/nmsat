__author__ = 'duarte'
import matplotlib.pyplot as pl
from modules.visualization import get_cmap
import numpy as np
from scipy.stats import sem


def harvest_results(pars, analysis_dict, results_path, plot=True, display=True, save=False):
	processed = dict()
	lab = dict() # to check

	fig1 = pl.figure()
	fig1.suptitle(analysis_dict['fig_title'])
	axes = []
	for ax_n, ax_title in enumerate(analysis_dict['ax_titles']):
		ax = fig1.add_subplot(len(analysis_dict['ax_titles']), 1, ax_n + 1)

		colors = get_cmap(len(analysis_dict['key_sets'][ax_n]), 'Accent')
		for idx_k, keys in enumerate(analysis_dict['key_sets'][ax_n]):
			print "\nHarvesting {0}".format(keys)
			labels, result = pars.harvest(results_path, key_set=keys)
			if plot:
				ax.plot(pars.parameter_axes['xticks'], np.mean(result.astype(float), 1), '-', c=colors(idx_k),
				        label=analysis_dict['labels'][ax_n][idx_k])
				ax.errorbar(pars.parameter_axes['xticks'], np.mean(result.astype(float), 1), marker='o', mfc=colors(idx_k),
				            mec=colors(idx_k), ms=2, linestyle='none', ecolor=colors(idx_k), yerr=sem(result.astype(
					float), 1))
			processed.update({keys: result})
			lab.update({keys: labels})
		if plot:
			ax.set_xlabel(r'$' + pars.parameter_axes['xlabel'] + '$')
			ax.set_xlim([min(pars.parameter_axes['xticks']), max(pars.parameter_axes['xticks'])])
			ax.set_title(ax_title)
			ax.legend()
		axes.append(ax)
	if save and plot:
		fig1.savefig(save + '_Results_{0}'.format(analysis_dict['fig_title']))
	if display and plot:
		pl.show(block=False)

	return processed, lab, axes, fig1