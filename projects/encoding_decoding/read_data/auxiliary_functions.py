__author__ = 'duarte'
import matplotlib.pyplot as pl
from modules.visualization import get_cmap, violin_plot, box_plot, InputPlots, pretty_raster
from modules.signals import SpikeList
from modules.analysis import Readout
from modules.input_architect import InputSignal
import modules.parameters as prs
import numpy as np
from scipy.stats import sem, truncnorm
import itertools


def harvest_results(pars, analysis_dict, results_path, display=True, save=False):
	processed = dict()

	fig1 = pl.figure()
	fig1.suptitle(analysis_dict['fig_title'])
	axes = []
	for ax_n, ax_title in enumerate(analysis_dict['ax_titles']):
		ax = fig1.add_subplot(len(analysis_dict['ax_titles']), 1, ax_n + 1)

		colors = get_cmap(len(analysis_dict['key_sets'][ax_n]), 'Accent')
		for idx_k, keys in enumerate(analysis_dict['key_sets'][ax_n]):
			print "\nHarvesting {0}".format(keys)
			labels, result = pars.harvest(results_path, key_set=keys)
			result = prs.clean_array(result)
			if len(result.shape) > 1:
				ax.plot(pars.parameter_axes['xticks'], np.nanmean(result.astype(float), 1), 'o-', c=colors(idx_k),
				        label=analysis_dict['labels'][ax_n][idx_k])
				ax.errorbar(pars.parameter_axes['xticks'], np.nanmean(result.astype(float), 1), marker='o', mfc=colors(
					idx_k),
				            mec=colors(idx_k), ms=2, linestyle='none', ecolor=colors(idx_k), yerr=sem(result.astype(
					float), 1))
			else:
				ax.plot(pars.parameter_axes['xticks'], result.astype(float), 'o-', c=colors(idx_k),
				        label=analysis_dict['labels'][ax_n][idx_k])
			processed.update({keys: result})
		ax.set_xlabel(r'$' + pars.parameter_axes['xlabel'] + '$')
		ax.set_xlim([min(pars.parameter_axes['xticks']), max(pars.parameter_axes['xticks'])])
		ax.set_title(ax_title)
		ax.legend()
		axes.append(ax)
	if save:
		fig1.savefig(save + '_Results_{0}'.format(analysis_dict['fig_title']))
	if display:
		pl.show(block=False)

	return processed, fig1, axes


def plot_distribution(data, pos, cmap, positions, idx, ax):
	if len(data.shape) > 1:
		# extract off-diagonal elements only
		di = np.diag_indices_from(data)
		mask = np.ones_like(data).astype(bool)
		mask[di] = False
		data = data[mask].flatten()

	violin_plot(ax, [data], pos=[pos], location=0, color=[cmap(positions[idx])])
	box_plot(ax, [data], pos=[pos])
	return ax


def plot_multitrial_averages(pars, result, ax, color, label):
	ax.plot(pars.parameter_axes['xticks'], np.mean(result.astype(float), 1), '-', c=color,
	        label=label)
	ax.errorbar(pars.parameter_axes['xticks'], np.mean(result.astype(float), 1), marker='o', mfc=color,
	            mec=color, ms=2, linestyle='none', ecolor=color, yerr=sem(result.astype(
			float), 1))
	ax.set_xlabel(r'$' + pars.parameter_axes['xlabel'] + '$')
	ax.set_xlim([min(pars.parameter_axes['xticks']), max(pars.parameter_axes['xticks'])])
	# ax.set_title(ax_title)
	ax.legend()


def reconstruct_spike_pattern(inputs, stim_set, set_name='full', n_steps=None):
	"""
	Assemble a sequence of spike templates (mostly for debugging)
	:return:
	"""
	spk_patterns = inputs.spike_patterns
	set_sequence = getattr(stim_set, "{0}_set".format(set_name))
	set_seq = np.array(np.argmax(set_sequence.todense(), 0))

	if n_steps is not None:
		set_seq = set_seq[0][-n_steps:]
	else:
		set_seq = set_seq[0]

	onsets = np.arange(0., len(set_seq)*200., 200.)
	all_times = []
	all_ids = []

	for idx, t in zip(set_seq, onsets):
		all_times.append(list(spk_patterns[idx].raw_data()[:, 0] + t))
		all_ids.append(list(spk_patterns[idx].raw_data()[:, 1]))
	all_times = list(itertools.chain(*all_times))
	all_ids = list(itertools.chain(*all_ids))

	tmp = [(all_ids[n], all_times[n]) for n in range(len(all_times))]
	spiking_activity = SpikeList(tmp, np.unique(all_ids).tolist())

	return spiking_activity


def evaluate_encoding(spike_list, input_sequence, input_signal, parameter_set):
	"""

	:param spike_list:
	:param input_sequence:
	:param input_stimulus:
	:param parameter_set:
	:return:
	"""
	n_input_neurons = 8000
	analysis_interval = np.round([spike_list.t_start, spike_list.t_stop])
	inp_responses = spike_list.filter_spiketrains(dt=0.1, tau=20., start=analysis_interval[0],
	                                              stop=analysis_interval[1], N=n_input_neurons)
	inp_readout_pars = prs.copy_dict(parameter_set.decoding_pars.readout[0],
	                                 {'label': 'InputNeurons',
	                                  'algorithm': 'pinv'})

	inp_readout = Readout(prs.ParameterSet(inp_readout_pars))
	analysis_signal = input_signal.time_slice(3000., 4000.)#analysis_interval[0], analysis_interval[1])
	inp_readout.train(inp_responses, analysis_signal.as_array())
	inp_readout.test(inp_responses)
	perf = inp_readout.measure_performance(analysis_signal.as_array())

	input_out = InputSignal()
	input_out.load_signal(inp_readout.output.T, dt=input_signal.dt)
	figure2 = pl.figure()
	figure2.suptitle(r'MAE = {0}'.format(str(perf['raw']['MAE'])))
	ax21 = figure2.add_subplot(211)
	ax22 = figure2.add_subplot(212, sharex=ax21)
	InputPlots(input_obj=analysis_signal).plot_input_signal(ax22, save=False, display=False)
	ax22.set_color_cycle(None)
	InputPlots(input_obj=input_out).plot_input_signal(ax22, save=False, display=False)
	ax22.set_ylim([analysis_signal.base - 10., analysis_signal.peak + 10.])
	# inp_spikes.raster_plot(with_rate=False, ax=ax21, save=False, display=False)

	pretty_raster(spike_list, analysis_interval, n_total_neurons=100)


def rebuild_stimulus_sequence(epochs, onset_times=None):
	"""
	Rebuild the original stimulus sequence from the marked epochs
	:param epochs: 
	:return: 
	"""
	stim = epochs.keys()

	all_onsets = {}
	for k, v in epochs.items():
		onsets = []
		for interval in v:
			onsets.append(interval[0])
		all_onsets.update({k: onsets})

	onset_times = np.arange(0., 10010 * 200., 200.)

	stim_seq = []
	for t in onset_times:
		for k, v in all_onsets.items():
			if t in v:
				stim_seq.append(k)

	return stim_seq


def generate_input_connections(n_stim, gamma_in, n_targets, r, mu_w, sig_w):
	"""
	Generate structured connection matrices (direct encoding)
	:param n_stim: Number of input stimuli
	:param gamma_in: connection density
	:param n_targets: number of target units
	:param r: clustering parameter
	:param mu_w: mean weight
	:param sig_w: weight std
	:return: numpy array of weights [n_stim x n_targets]
	"""
	N_aff = int(gamma_in * n_targets)
	p_m = r
	p0 = 1 - r
	A = np.zeros((n_stim, n_targets))

	probs = p0 * np.ones((n_stim, n_targets))
	for i in range(n_stim):
		if gamma_in <= 1./n_stim:
			probs[i, int(i * N_aff):int((i+1)*N_aff)] = p_m
		else:
			overlap = n_targets - (N_aff * n_stim)
			print overlap
			probs[i, int(i * N_aff):int((i + 1) * N_aff)] = p_m

		probs[i, :] /= sum(probs[i, :])
		#print np.sum(probs)
		post_list = np.arange(n_targets)
		targets = np.random.choice(post_list, N_aff, replace=False, p=probs[i, :]).astype(int)
		A[i, targets] = 1.
	if sig_w:
		a = (0.0001 - mu_w)/sig_w
		b = ((10. * mu_w) - mu_w) / sig_w
		W = truncnorm.rvs(a, b, loc=mu_w, scale=sig_w, size=(n_stim, n_targets))
	else:
		W = np.ones((n_stim, n_targets)) * mu_w

	return A * W