__author__ = 'duarte'
from modules.visualization import plot_single_raster
from modules.visualization import plot_histogram, progress_bar
import matplotlib.pyplot as pl
import numpy as np
from scipy.optimize import curve_fit
import itertools
import nest
"""
Extra functions, specific for this project
"""


# def plot_vm(vm, times, ax, color, lw=1, v_reset=-70., v_th=-50.):
# 	ax.plot(times, vm, c=color)
# 	#ax.set_xlabel('Time [ms]')
# 	ax.set_ylabel(r'$V_{m}$')
# 	idxs = vm.argsort()
# 	possible_spike_times = [t for t in idxs if (t < len(vm) - 1) and (vm[t + 1] == v_reset) and (vm[t] != v_reset)]
# 	ax.vlines(times[possible_spike_times], v_th, 50., color='k')
# 	ax.set_ylim(min(vm) - 5., 10.)


def plot_single_neuron_response(parameter_set, activity_dict, input_times, t_axis, analysis_interval,
                                response_type='E'):
	figures1 = [pl.figure() for _ in range(parameter_set.net_pars.n_populations)]
	neuron_pars = [parameter_set.net_pars.neuron_pars[idx] for idx in range(parameter_set.net_pars.n_populations)]
	axes1 = []
	for n_fig, fig in enumerate(figures1):
		# axes1.append([fig.add_subplot(4, 1, 1), fig.add_subplot(4, 1, 2), fig.add_subplot(4, 1, 3),
		#               fig.add_subplot(4, 1, 4)])
		fig.suptitle(
			'{0} Responses at rest [in {1} neurons]'.format(str(response_type), str(parameter_set.net_pars.pop_names[
				                                                                   n_fig])))

	ctr = 0
	for pop, r in activity_dict.items():
		ax1 = figures1[ctr].add_subplot(411)
		ax1.set_xlim(analysis_interval)
		ax2 = figures1[ctr].add_subplot(412, sharex=ax1)
		ax2.set_xlim(analysis_interval)
		ax3 = figures1[ctr].add_subplot(413, sharex=ax1)
		ax3.set_xlim(analysis_interval)
		ax4 = figures1[ctr].add_subplot(414, sharex=ax1)
		ax4.set_xlim(analysis_interval)

		neuron_index = np.random.permutation(len(r['recorded_neurons']))[0]

		PSConds = r['G_syn_tot'][neuron_index]
		colors = []
		if response_type == 'E':
			receptor_indices = [1, 3]
			PSCurrs = -r['I_ex'][neuron_index] / 1000.
			colors = ['blue', 'SteelBlue', 'Navy']
			labels = [r'$\mathrm{Total}$', r'$\mathrm{AMPA}$', r'$\mathrm{NMDA}$']
		elif response_type == 'I':
			receptor_indices = [2, 4]
			PSCurrs = -r['I_in'][neuron_index] / 1000.
			colors = ['red', 'Crimson', 'OrangeRed']
			labels = [r'$\mathrm{Total}$', r'$\mathrm{GABA_{A}}$', r'$\mathrm{GABA_{B}}$']
		else:
			raise TypeError("Incorrect response_type")

		rec_cond = {}
		for r1 in receptor_indices:
			k = 'C{0}'.format(str(r1))
			rec_cond.update({k: r[k][neuron_index] * parameter_set.net_pars.neuron_pars[ctr]['rec_cond'][r1-1]})
		PSPs = r['V_m'][neuron_index]

		# plot presynaptic spike train
		plot_single_raster(input_times, ax1, t_start=analysis_interval[0], t_stop=analysis_interval[1])
		# plot post-synaptic potential
		#plot_vm(PSPs, t_axis, ax2, 'k', 2, v_reset=nest.GetStatus([neuron_index])[0]['V_reset'], v_th=nest.GetStatus(
		#		[neuron_index])[0]['V_th'])
		ax2.plot(t_axis, PSPs, 'k', lw=2)
		ax2.set_xlabel('')
		# ax2.set_ylim([nest.GetStatus([neuron_index])[0]['E_L'] - 5, nest.GetStatus([neuron_index])[0]['V_th'] -
		#               20.])
		ax2.set_ylabel(r'$\mathrm{V_{m}} [\mathrm{mV}]$')
		ax2.set_xticks([])
		ax2.set_xticklabels([])
		# plot total conductance
		ax3.plot(t_axis, PSConds, '-.', c=colors[0], lw=2., alpha=0.8, label=labels[0])

		# individual rec conductances/currents
		ctr2 = 1
		for k, v in rec_cond.items():
			ax3.plot(t_axis, v, '-', c=colors[ctr2], lw=2., alpha=1., label=labels[ctr2])
			if response_type == 'E':
				ax4.plot(t_axis, (v * PSPs) / 1000., '-', c=colors[ctr2], lw=2., alpha=1.)
			else:
				ax4.plot(t_axis, (-v * PSPs) / 1000., '-', c=colors[ctr2], lw=2., alpha=1.)
			ctr2 += 1
		ax3.set_ylabel(r'$\mathrm{G}^{\mathrm{rec}} [\mathrm{nS}]$')
		ax3.set_xticks([])
		ax3.legend()

		ax4.plot(t_axis, PSCurrs, '-.', c=colors[0], lw=2., alpha=0.8)
		ax4.set_ylabel(r'$\mathrm{I}^{\mathrm{rec}} [\mathrm{nA}]$')
		ax4.set_xlabel(r'Time [ms]')

		ctr += 1
	# pl.legend()
	pl.show()


def spike_triggered_synaptic_responses(parameter_set, activity_dict, time_window, input_times, t_axis,
                                      response_type='E', plot=True, display=True, save=False):
	"""

	:return:
	"""
	if plot:
		figures = [pl.figure() for _ in range(parameter_set.net_pars.n_populations)]
		fig3 = pl.figure()
		ax31 = fig3.add_subplot(131)
		ax31.set_title('PSC latency [ms]')
		ax32 = fig3.add_subplot(132)
		ax32.set_title('PSC amplitude [nA]')
		ax33 = fig3.add_subplot(133)
		ax33.set_title('Charge [nC]')

		for n_fig, fig in enumerate(figures):
			fig.suptitle(
				'{0} Responses at rest [in {1} neurons]'.format(str(response_type), str(parameter_set.net_pars.pop_names[
					                                                                        n_fig])))
	if response_type == 'E':
		receptor_indices = [1, 3]
		current_key = 'I_ex'
		colors = ['blue', 'SteelBlue', 'Navy']
		labels = [r'$\mathrm{Total}$', 'AMPA', 'NMDA']
	elif response_type == 'I':
		receptor_indices = [2, 4]
		current_key = 'I_in'
		colors = ['red', 'Crimson', 'OrangeRed']
		labels = [r'$\mathrm{Total}$', 'GABAA', 'GABAB']
	else:
		raise TypeError("Incorrect response_type")

	ctr = 0
	results = {}
	t = np.arange(time_window[0], time_window[1], 0.1)
	keys = ['C{0}'.format(str(r1)) for r1 in receptor_indices]

	for pop, r in activity_dict.items():
		results.update({pop: {}})
		rec_cond = {}
		rec_curr = {}
		rec_charge = {}
		rec_psc_amp = {}
		rec_psc_lat = {}
		syn_psc_amp = []
		syn_psc_lat = []
		neurons = list(r['recorded_neurons'])
		st = nest.GetStatus(neurons)
		for r1 in receptor_indices:
			k = 'C{0}'.format(str(r1))
			cond_array = np.zeros_like(r[k])
			psc_array = np.zeros_like(r[k])
			for n in range(len(neurons)):
				cond_array[n, :] = r[k][n] * st[n]['rec_cond'][r1-1]
				if response_type == 'E':
					psc_array[n, :] = (cond_array[n, :] * r['V_m'][n]) / 1000.
				else:
					psc_array[n, :] = (-cond_array[n, :] * r['V_m'][n]) / 1000.
			rec_cond.update({k: cond_array})
			rec_curr.update({k: psc_array})
			rec_psc_amp.update({k: []})
			rec_psc_lat.update({k: []})
			rec_charge.update({k: []})

		for spk_t in input_times:
			window = [spk_t + time_window[0], spk_t + time_window[1]]
			indices = [np.where(np.round(t_axis, 1) == round(window[0], 1))[0][0], np.where(np.round(t_axis,
			    1) == round(window[1], 1))[0][0]]
			print(indices)

			ctr1 = 1
			if plot:
				ax1 = figures[ctr].add_subplot(111)
				ax1.set_ylabel(r'$I^{\mathrm{rec}} [\mathrm{nA}]$')

			for r1 in receptor_indices:
				k = 'C{0}'.format(str(r1))
				single_pscs = rec_curr[k][:, indices[0]:indices[1]]
				for psc in range(single_pscs.shape[0]):
					if plot:
						ax1.plot(t, single_pscs[psc, :], c=colors[ctr1], lw=0.5, alpha=0.5)
						psc_latencies, psc_amplitudes = get_extrema(single_pscs[psc, :], t, ax1)
					else:
						psc_latencies, psc_amplitudes = get_extrema(single_pscs[psc, :], t, ax=None)
					rec_psc_amp[k].append(psc_amplitudes)
					rec_psc_lat[k].append(psc_latencies)
					rec_charge[k].append(np.abs(np.trapz(single_pscs[psc, :], dx=0.1)))

				ctr1 += 1
			total_pscs = (-r[current_key][:, indices[0]:indices[1]]) / 1000.
			for psc in range(total_pscs.shape[0]):
				if plot:
					ax1.plot(t, total_pscs[psc, :], c=colors[0], lw=1., alpha=1.)
					psc_latencies, psc_amplitudes = get_extrema(total_pscs[psc, :], t, ax1)
				else:
					psc_latencies, psc_amplitudes = get_extrema(total_pscs[psc, :], t, ax=None)
				syn_psc_amp.append(psc_amplitudes)
				syn_psc_lat.append(psc_latencies)

		ctr2 = 1
		for k in keys:
			if plot:
				plot_histogram(rec_psc_lat[k], nbins=10, norm=True, mark_mean=True, ax=ax31, color=colors[ctr2],
				               display=False, save=False)
				plot_histogram(rec_psc_amp[k], nbins=10, norm=True, mark_mean=True, ax=ax32, color=colors[ctr2],
				               display=False, save=False)
				plot_histogram(rec_charge[k], nbins=10, norm=True, mark_mean=True, ax=ax33, color=colors[ctr2],
				               display=False, save=False)

			results[pop].update({'{0}_latencies'.format(labels[ctr2]): rec_psc_lat[k],
			                '{0}_amplitudes'.format(labels[ctr2]): rec_psc_amp[k],
			                '{0}_charges'.format(labels[ctr2]): rec_charge[k]})
			ctr2 += 1
		if plot:
			plot_histogram(syn_psc_lat, nbins=10, norm=True, mark_mean=True, ax=ax31, color=colors[0],
				               display=False, save=False)
			plot_histogram(syn_psc_amp, nbins=10, norm=True, mark_mean=True, ax=ax32, color=colors[0],
			               display=False, save=False)

		results[pop].update({'psc_ratio': np.mean(rec_psc_amp[keys[1]]) / np.mean(rec_psc_amp[keys[0]])})
		results[pop].update({'q_ratio': np.mean(rec_charge[keys[1]]) / np.mean(rec_charge[keys[0]])})

		print("{0} / {1} PSC ratio = {2}".format(labels[2], labels[1], str(results[pop]['psc_ratio'])))
		print("{0} / {1} Charge ratio = {2}".format(labels[2], labels[1], str(results[pop]['q_ratio'])))

		if display:
			pl.show(block=False)
		if plot and save:
			for fig_id, fig in enumerate(figures):
				fig.savefig(save + '{0}.pdf'.format(fig_id))
			fig3.savefig(save + '_distributions.pdf')
	return results


def PSC_kinetics(activity_dict, time_window, input_times, t_axis, response_type='E', plot=True, display=True,
                 save=False):
	"""

	:param parameter_set:
	:param activity_dict:
	:param time_window:
	:param input_times:
	:param t_axis:
	:param response_type:
	:return:
	"""
	if response_type == 'E':
		current_key = 'I_ex'
		colors = ['blue', 'SteelBlue', 'Navy']
	elif response_type == 'I':
		current_key = 'I_in'
		colors = ['red', 'Crimson', 'OrangeRed']
	else:
		raise TypeError("Incorrect response_type")

	results = {}
	t = np.arange(time_window[0], time_window[1], 0.1)

	for pop, r in activity_dict.items():
		if plot:
			fig1 = pl.figure()
			fig1.suptitle(r'{0}PSCs in {1} neurons'.format(str(response_type), str(pop)))
			ax11 = fig1.add_subplot(111)
			fig2 = pl.figure()
			fig2.suptitle(r'{0}PSCs in {1} neurons'.format(str(response_type), str(pop)))
			ax21 = fig2.add_subplot(221)
			ax22 = fig2.add_subplot(222)
			ax23 = fig2.add_subplot(223)
			ax24 = fig2.add_subplot(224)

		results.update({pop: {}})
		syn_psc_amp = []
		syn_psc_lat = []
		syn_psc_rise = []
		syn_psc_decay = []
		PSCs = []
		for spk_t in input_times:
			window = [spk_t + time_window[0], spk_t + time_window[1]]
			indices = [np.where(np.round(t_axis, 1) == round(window[0], 1))[0][0], np.where(np.round(t_axis,
			    1) == round(window[1], 1))[0][0]]
			print indices

			total_pscs = (r[current_key][:, indices[0]:indices[1]]) / 1000.

			for psc in range(total_pscs.shape[0]):
				psc_latencies, psc_amplitudes = get_extrema(total_pscs[psc, :], t, ax=None)
				idx_0 = np.where(np.round(t, 1) == round(psc_latencies, 1))[0][0]
				total_pscs[psc, :] -= total_pscs[psc, idx_0]
				psc_latencies, psc_amplitudes = get_extrema(total_pscs[psc, :], t, ax=None)
				if psc_amplitudes < 0.:
					initial_guess = (-0.01, 0.5, 20.)
				else:
					initial_guess = (0.01, 0.5, 20.)
				if plot:
					ax11.plot(t, total_pscs[psc, :], c=colors[0], lw=0.5, alpha=0.3)

				syn_psc_amp.append(psc_amplitudes)
				syn_psc_lat.append(psc_latencies)

				try:
					time_axis_fit = t[idx_0:] - t[idx_0]
					fit_params, fit_cov = curve_fit(synaptic_kinetics_function, time_axis_fit, total_pscs[psc,
					                                                            idx_0:], p0=initial_guess)
					error = np.sum((total_pscs[psc, idx_0:] - synaptic_kinetics_function(t[idx_0:], *fit_params)) ** 2)
					#print fit_params, error
					if error < 0.05:
						syn_psc_rise.append(fit_params[1])
						syn_psc_decay.append(fit_params[2])
				except:
					continue

			PSCs.append(total_pscs)

		results[pop].update({'PSC_amplitudes': syn_psc_amp, 'PSC_latencies': syn_psc_lat, 'PSC_rise_times':
			syn_psc_rise, 'PSC_decay_times': syn_psc_decay})

		total_psc = np.mean(np.array(PSCs), 0)[0]

		# normalize by the value at lag 0
		psc_lat, psc_amp = get_extrema(total_psc, t)
		idx_0 = np.where(np.round(t, 1) == round(psc_lat, 1))[0][0]
		total_psc -= total_psc[idx_0]
		psc_lat, psc_amp = get_extrema(total_psc, t)
		if psc_amp < 0.:
			initial_guess = (-0.01, 0.5, 50.)
		else:
			initial_guess = (0.01, 0.5, 50.)

		time_axis_fit = t[idx_0:] - t[idx_0]
		fit_params, fit_cov = curve_fit(synaptic_kinetics_function, time_axis_fit, total_psc[idx_0:],
		                                               p0=initial_guess)
		print('{0}PSCs in {1} neurons'.format(str(response_type), str(pop)))
		print("\t- Amplitude = {0} [nA]".format(str(psc_amp)))
		print("\t- Latency = {0} [ms]".format(str(psc_lat)))
		print("\t- Rise = {0} [ms]".format(str(fit_params[1])))
		print("\t- Decay = {0} [ms]".format(str(fit_params[2])))


		results[pop].update({'mean_PSC': total_psc, 'mean_fit_a': fit_params[0], 'mean_fit_rise': fit_params[1],
		                     'mean_fit_decay': fit_params[2], 'mean_amplitude': psc_amp})

		if plot:
			label = r'$a=%10.2f, \tau_{\mathrm{r}}=%10.2f, \tau_{\mathrm{d}}=%10.2f$' % (fit_params[0], fit_params[1],
			                                                                             fit_params[2])

			ax11.plot(t[idx_0:], total_psc[idx_0:], c=colors[0], lw=3., alpha=1.)
			ax11.plot(t[idx_0:], synaptic_kinetics_function(t[idx_0:] - t[idx_0], *fit_params), 'r--', label=label)
			ax11.legend()
			ax11.set_xlim(time_window)
			ax11.set_xlabel(r'$\mathrm{Time [ms]}$')
			ax11.set_ylabel(r'$\mathrm{I_{syn} [nA]}$')

			plot_histogram(syn_psc_amp, nbins=10, norm=True, mark_mean=True, ax=ax21, color=colors[0],
		                   display=False, save=False)
			ax21.set_xlabel(r'Amplitude')
			plot_histogram(syn_psc_lat, nbins=10, norm=True, mark_mean=True, ax=ax22, color=colors[0],
			               display=False, save=False)
			ax22.set_xlabel(r'Latency')
			plot_histogram(syn_psc_rise, nbins=10, norm=True, mark_mean=True, ax=ax23, color=colors[0],
			               display=False, save=False)
			ax23.set_xlabel(r'$\tau_{\mathrm{rise}}$')
			plot_histogram(syn_psc_decay, nbins=10, norm=True, mark_mean=True, ax=ax24, color=colors[0],
			               display=False, save=False)
			ax24.set_xlabel(r'$\tau_{\mathrm{decay}}$')

			if display:
				pl.show(block=False)
			if save:
				assert isinstance(save, str), "Please provide filename"
				fig1.savefig(save + '_PSCs1.pdf')
				fig2.savefig(save + '_PSCs2.pdf')
	return results


def PSP_kinetics(activity_dict, time_window, input_times, t_axis, response_type='E', plot=True, display=True,
                 save=False):
	"""

	:param parameter_set:
	:param activity_dict:
	:param time_window:
	:param input_times:
	:param t_axis:
	:param response_type:
	:return:
	"""
	if response_type == 'E':
		colors = ['blue', 'SteelBlue', 'Navy']
	elif response_type == 'I':
		colors = ['red', 'Crimson', 'OrangeRed']
	else:
		raise TypeError("Incorrect response_type")

	results = {}
	t = np.arange(time_window[0], time_window[1], 0.1)

	for pop, r in activity_dict.items():
		if plot:
			fig1 = pl.figure()
			fig1.suptitle(r'{0}PSPs in {1} neurons'.format(str(response_type), str(pop)))
			ax11 = fig1.add_subplot(111)
			fig2 = pl.figure()
			fig2.suptitle(r'{0}PSPs in {1} neurons'.format(str(response_type), str(pop)))
			ax21 = fig2.add_subplot(221)
			ax22 = fig2.add_subplot(222)
			ax23 = fig2.add_subplot(223)
			ax24 = fig2.add_subplot(224)

		results.update({pop: {}})
		syn_psp_amp = []
		syn_psp_lat = []
		syn_psp_rise = []
		syn_psp_decay = []
		PSPs = []
		for spk_t in input_times:
			window = [spk_t + time_window[0], spk_t + time_window[1]]
			indices = [np.where(np.round(t_axis, 1) == round(window[0], 1))[0][0], np.where(np.round(t_axis,
			                                                                                         1) == round(
				window[1], 1))[0][0]]
			print indices
			total_psps = np.copy(r['V_m'][:, indices[0]:indices[1]])
			for psp in range(total_psps.shape[0]):
				var0 = total_psps[psp, :] - total_psps[psp, 0]
				psp_latencies, psp_amplitudes = get_extrema(var0, t, ax=None)
				#ax.plot(t, var0, c=colors[0], lw=0.5, alpha=0.3)
				#pl.show()
				idx_0 = np.where(np.round(t, 1) == round(psp_latencies, 1))[0][0]
				#print total_psps[psp, idx_0]
				total_psps[psp, :] -= total_psps[psp, 0]#idx_0]
				psp_latencies, psp_amplitudes = get_extrema(total_psps[psp, :], t, ax=None)
				if psp_amplitudes < 0.:
					initial_guess = (-0.01, 0.5, 50.)
				else:
					initial_guess = (0.01, 0.5, 50.)
				if plot:
					ax11.plot(t, total_psps[psp, :], c=colors[0], lw=0.5, alpha=0.3)
					#ax.plot(t, total_psps[psp, :], c=colors[0], lw=0.5, alpha=0.3)
					#pl.show()
				syn_psp_amp.append(psp_amplitudes)
				syn_psp_lat.append(psp_latencies)
				try:
					time_axis_fit = t[idx_0:] - t[idx_0]
					fit_params, fit_cov = curve_fit(synaptic_kinetics_function, time_axis_fit, total_psps[
					                                    psp, idx_0:], p0=initial_guess)
					error = np.sum((total_psps[psp, idx_0:] - synaptic_kinetics_function(t[idx_0:], *fit_params)) ** 2)
					#print fit_params, error
					#if error < 0.05:
					syn_psp_rise.append(fit_params[1])
					syn_psp_decay.append(fit_params[2])
					#ax11.plot(t, synaptic_kinetics_function(t, **fit_params), '--')
				except:
					continue
			PSPs.append(total_psps)

		results[pop].update({'PSP_amplitudes': syn_psp_amp, 'PSP_latencies': syn_psp_lat, 'PSP_rise_times':
			syn_psp_rise, 'PSP_decay_times': syn_psp_decay})

		total_psp = np.mean(np.array(PSPs), 0)[0]
		# normalize by the value at lag 0
		psp_lat, psp_amp = get_extrema(total_psp, t)
		idx_0 = np.where(np.round(t, 1) == round(psp_lat, 1))[0][0]
		total_psp -= total_psp[idx_0]
		psp_lat, psp_amp = get_extrema(total_psp, t)
		if psp_amp < 0.:
			initial_guess = (-0.01, 0.5, 50.)
		else:
			initial_guess = (0.01, 0.5, 50.)
		time_axis_fit = t[idx_0:] - t[idx_0]
		fit_params, fit_cov = curve_fit(synaptic_kinetics_function, time_axis_fit, total_psp[idx_0:],
		                                               p0=initial_guess)
		print('{0}PSPs in {1} neurons'.format(str(response_type), str(pop)))
		print("\t- Amplitude = {0} [mV]".format(str(psp_amp)))
		print("\t- Latency = {0} [ms]".format(str(psp_lat)))
		print("\t- Rise = {0} [ms]".format(str(fit_params[1])))
		print("\t- Decay = {0} [ms]".format(str(fit_params[2])))

		results[pop].update({'mean_PSP': total_psp, 'mean_fit_a': fit_params[0], 'mean_fit_rise': fit_params[1],
		                     'mean_fit_decay': fit_params[2], 'mean_amplitude': psp_amp})

		if plot:
			label = r'$a=%10.2f, \tau_{\mathrm{r}}=%10.2f, \tau_{\mathrm{d}}=%10.2f$' % (fit_params[0], fit_params[1],
			                                                                             fit_params[2])

			ax11.plot(t[idx_0:], total_psp[idx_0:], c=colors[0], lw=3., alpha=1.)
			ax11.plot(t[idx_0:], synaptic_kinetics_function(t[idx_0:] - t[idx_0], *fit_params), 'r--', label=label)
			ax11.legend()
			ax11.set_xlim(time_window)
			ax11.set_xlabel(r'$\mathrm{Time [ms]}$')
			ax11.set_ylabel(r'$\mathrm{V_{m} [mV]}$')

			plot_histogram(syn_psp_amp, nbins=10, norm=True, mark_mean=True, ax=ax21, color=colors[0],
		                   display=False, save=False)
			ax21.set_xlabel(r'Amplitude')
			plot_histogram(syn_psp_lat, nbins=10, norm=True, mark_mean=True, ax=ax22, color=colors[0],
			               display=False, save=False)
			ax22.set_xlabel(r'Latency')
			plot_histogram(syn_psp_rise, nbins=10, norm=True, mark_mean=True, ax=ax23, color=colors[0],
			               display=False, save=False)
			ax23.set_xlabel(r'$\tau_{\mathrm{rise}}$')
			plot_histogram(syn_psp_decay, nbins=10, norm=True, mark_mean=True, ax=ax24, color=colors[0],
			               display=False, save=False)
			ax24.set_xlabel(r'$\tau_{\mathrm{decay}}$')

			if display:
				pl.show(block=False)
			if save:
				assert isinstance(save, str), "Please provide filename"
				fig1.savefig(save + '_PSPs1.pdf')
				fig2.savefig(save + '_PSPs2.pdf')
	return results


def evaluate_capacity(state, input_signal, max_degree=5, max_variables=5, max_delay=100, signal_t0=1000, readout=None):
	"""

	:param state:
	:param input_signal:
	:param degree:
	:param delay:
	:param window:
	:return:

	"""
	print "\n*******************************\nCapacity Evaluation\n*******************************"
	results = {}
	for d in range(1, max_degree+1, 1):
		print "- Degree {0}".format(str(d))
		results.update({'d{0}'.format(str(d)): {'evaluated_sets': [], 'capacities': [], 'total_capacity': 0.}})

		variables = [N for N in range(1, max_variables+1, 1) if N <= d]
		for ii, N in enumerate(variables):
			print "\t- Variables {0}".format(str(N))
			power_list = generate_subset(d, N)
			print power_list
			for set in power_list:
				window_sizes = np.arange(N, max_delay + 1, 1)
				for window_len in window_sizes:
					positions = list(itertools.permutations(range(window_len), len(set)))
					for pos in positions:
						window = np.zeros(window_len)
						for i, x in enumerate(pos):
							window[x] = set[i]

						results['d{0}'.format(str(d))]['evaluated_sets'].append(window)
						capacitites = []
						polynomials = []
						for idx, n in enumerate(window):
							signal = shift_signal(input_signal, signal_t0, delay=idx)
							if n:
								P = compute_polynomial(signal, n, normalize=True)
							else:
								P = compute_polynomial(signal, n, normalize=False)
							polynomials.append(P)

						target = np.product(polynomials, 0)
						# pl.plot(target)
						# pl.show()

						output = explicit_readout(readout, state, target)
						# error0 = error(output, target)
						capacitites.append(capacity(state, output, target, method=0))
						# c_m2 = capacity(state, output, target, method=2)
						readout.reset()
						results['d{0}'.format(str(d))]['capacities'].append(capacitites)

			progress_bar(float(ii) / float(max_variables))
		results['d{0}'.format(str(d))]['total_capacity'] = np.sum(list(itertools.chain(*results['d{0}'.format(str(d))]['capacities'])))
		print "\t\t- Capacity = {0}".format(str(results['d{0}'.format(str(d))]['total_capacity']))
	return results


########################################################################################################################
# def synaptic_kinetics_function(x, a, tau_rise, tau_decay):
# 	t1 = np.exp(-x / tau_rise)
# 	t2 = np.exp(-x / tau_decay)
#
# 	return a + (t2 - t1) #((1. - t1) * ((r * t2) + ((1 - r) * t3)))

def synaptic_kinetics_function(x, a, tau_rise, tau_decay):
	return a * (np.exp(-x / tau_decay) - np.exp(-x / tau_rise))


def get_extrema(var, t_axis, ax=None):
	# if var[0] < 0.:
	# 	var2 = np.copy(var)
	# 	var2 -= var[0]
	# 	tmp = np.gradient(np.gradient(var2))
	# 	local_maxima = np.where(var2 == np.max(var2))[0]
	# else:
	tmp = np.gradient(np.gradient(np.abs(var)))
	local_maxima = np.where(np.abs(var) == np.max(np.abs(var)))[0]
	local_minima = np.argmax(tmp)
	#local_maxima = np.where(np.abs(var) == np.max(np.abs(var)))[0]
	if ax is not None:
		ax.plot(t_axis[local_minima], var[local_minima], 'or')
		ax.plot(t_axis[local_maxima], var[local_maxima], 'og')
	return t_axis[local_minima], var[local_maxima][0]


def time_shift_arrays(state, target, delay):
	"""

	:param state:
	:param target:
	:param delay:
	:return:
	"""
	return state[:, delay:], target[:, :-delay]


def shift_signal(signal, start=0, delay=0):
	"""

	:param input_signal:
	:param start_t:
	:param delay:
	:return:
	"""
	if delay:
		return signal[start-delay:-delay]
	else:
		return signal[start:]


def explicit_readout(readout, state, target):
	"""

	:param state:
	:param target:
	:return:
	"""
	readout.train(state, target, False)
	norm_wout = readout.measure_stability()
	# print "|W_out| [{0}] = {1}".format(readout.name, str(norm_wout))
	return readout.test(state, display=False)


def error(output, target):
	"""

	:param output:
	:param target:
	:return:
	"""
	if output.shape == target.shape:
		out = output
	else:
		out = output.T

	MAE = np.mean(out - target)
	MSE = mse(out, target)
	RMSE = rmse(out, target)
	NMSE = nmse(out, target)
	NRMSE = nrmse(out[0], target[0])

	# print("\t- MAE = {0}\n\t- MSE = {1}\n\t- NMSE = {2}\n\t- RMSE = ".format(str(MAE)))
	# print "\t- MSE = {0}".format(str(MSE))
	# print "\t- NMSE = {0}".format(str(NMSE))
	# print "\t- RMSE = {0}".format(str(RMSE))
	# print "\t- NRMSE = {0}".format(str(NRMSE))

	return output, {'MAE': MAE, 'MSE': MSE, 'NMSE': NMSE, 'RMSE': RMSE, 'NRMSE': NRMSE}


def capacity(state, output, target, method=0):
	"""

	:param state:
	:param output:
	:param target:
	:return:
	"""
	if output.shape == target.shape:
		est = output
		tar = target
	else:
		est = output[:, 0]
		tar = target[0, :]
	if method == 0:
		COV = (np.cov(tar, est) ** 2.)
		VARS = np.var(est) * np.var(tar)
		FMF = COV / VARS
		fmf = FMF[0, 1]
		print "C = {0}".format(str(fmf))
	elif method == 1:
		a = np.zeros((state.shape[0]))
		b = np.zeros((state.shape[0], state.shape[0]))
		c = np.zeros((state.shape[0]))
		for i in range(state.shape[0]):
			for j in range(state.shape[0]):
				a[i] = np.correlate(tar, state[i, :])
				b[i, j] = np.correlate(state[i, :], state[j, :])
				c[j] = np.correlate(state[j, :], tar)

	elif method == 2:
		a = []
		b = []

		for t in range(state.shape[1]):
			a.append(est[t] ** 2.)
			b.append(tar[t] ** 2.)
		fmf = np.mean(a) / np.mean(b)
		print "C = {0}".format(str(fmf))

	return fmf


def generate_subset(degree, variables):
	target = degree
	vals = range(degree + 1)
	leng = variables
	tmp = [x for x in itertools.chain(*[itertools.permutations(vals, leng), itertools.combinations_with_replacement(vals,
	                                    leng)])]
	return [x for x in set(tmp) if sum(x) == target]


def compute_polynomial(input, degree, normalize=True):
	from scipy.special import legendre
	from sklearn.preprocessing import MinMaxScaler
	if not normalize:
		return legendre(degree)(input)
	else:
		a = legendre(degree)(input)
		# TODO - check normalize - [0, 1] ? or [-1, 1]?
		min_max_scaler = MinMaxScaler()
		return min_max_scaler.fit_transform(a)


def determine_lognormal_parameters(mean, var, median=None):
	"""
	Returns the mu and signal parameters for a lognormal distribution, to achieve a desired mean and variance
	:param mean:
	:param var:
	:return:
	"""
	if median is None:
		mu = np.log( (mean**2.) / (np.sqrt(var + mean**2.)) )
	else:
		mu = np.log(median)
	sigma = np.sqrt( np.log( (var / (mean ** 2.)) + 1) )

	return mu, sigma