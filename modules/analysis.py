"""
=====================================================================================
Analysis Module
=====================================================================================
(partially modified from NeuroTools.analysis)

Collection of analysis and utility functions that are used by other modules

Functions:
------------
ccf 						- fast cross-correlation function, using fft
_dict_max 					- for a dict containing numerical values, return the key for the highest
crosscorrelate 				-
makekernel 					- creates kernel functions for convolution
simple_frequency_spectrum 	- simple calculation of frequency spectrum

Classes:
------------
DecodingLayer 	- reads population activity in response to patterned inputs, extracts the network state
				  (according to specifications) and trains readout weights
Readout 		- Readout object, trained to produce an estimation y(t) of output by reading out
				  population state variables.

Full Analysis interfaces:
-------------------------
noise_driven_dynamics
...

"""
#TODO remove and replace all locals()...
# other imports
import sys
import numpy as np
import itertools
import time
import copy
import scipy.optimize as opt
import matplotlib.pyplot as pl
import matplotlib as mpl
from scipy.spatial import distance
import scipy.integrate as integ
import sklearn.decomposition as sk
import sklearn.linear_model as lm
import sklearn.svm as svm
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as met
import sklearn.manifold as man
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# NMT imports
import parameters as pa
import input_architect as ia
import signals as sg
import visualization as vz
import net_architect as na
import io
from modules import check_dependency

# nest
import nest


np.seterr(all='ignore')

has_pyspike = check_dependency('pyspike')
if has_pyspike:
	import pyspike as spk


############################################################################################
def ccf(x, y, axis=None):
	"""
	Fast cross correlation function based on fft.

	Computes the cross-correlation function of two series.
	Note that the computations are performed on anomalies (deviations from
	average).
	Returns the values of the cross-correlation at different lags.

	Parameters
	----------
	x, y : 1D MaskedArrays
		The two input arrays.
	axis : integer, optional
		Axis along which to compute (0 for rows, 1 for cols).
		If `None`, the array is flattened first.

	Examples
	--------
	>> z = np.arange(5)
	>> ccf(z,z)
	array([  3.90798505e-16,  -4.00000000e-01,  -4.00000000e-01,
			-1.00000000e-01,   4.00000000e-01,   1.00000000e+00,
			 4.00000000e-01,  -1.00000000e-01,  -4.00000000e-01,
			-4.00000000e-01])
	"""
	assert x.ndim == y.ndim, "Inconsistent shape !"
	if axis is None:
		if x.ndim > 1:
			x = x.ravel()
			y = y.ravel()
		npad = x.size + y.size
		xanom = (x - x.mean(axis=None))
		yanom = (y - y.mean(axis=None))
		Fx = np.fft.fft(xanom, npad, )
		Fy = np.fft.fft(yanom, npad, )
		iFxy = np.fft.ifft(Fx.conj() * Fy).real
		varxy = np.sqrt(np.inner(xanom, xanom) * np.inner(yanom, yanom))
	else:
		npad = x.shape[axis] + y.shape[axis]
		if axis == 1:
			if x.shape[0] != y.shape[0]:
				raise ValueError("Arrays should have the same length!")
			xanom = (x - x.mean(axis=1)[:, None])
			yanom = (y - y.mean(axis=1)[:, None])
			varxy = np.sqrt((xanom * xanom).sum(1) *
							(yanom * yanom).sum(1))[:, None]
		else:
			if x.shape[1] != y.shape[1]:
				raise ValueError("Arrays should have the same width!")
			xanom = (x - x.mean(axis=0))
			yanom = (y - y.mean(axis=0))
			varxy = np.sqrt((xanom * xanom).sum(0) * (yanom * yanom).sum(0))
		Fx = np.fft.fft(xanom, npad, axis=axis)
		Fy = np.fft.fft(yanom, npad, axis=axis)
		iFxy = np.fft.ifft(Fx.conj() * Fy, n=npad, axis=axis).real
	# We just turn the lags into correct positions:
	iFxy = np.concatenate((iFxy[len(iFxy) / 2:len(iFxy)],
						   iFxy[0:len(iFxy) / 2]))
	return iFxy / varxy


def lag_ix(x,y):
	"""
	Calculate lag position at maximal correlation
	:param x:
	:param y:
	:return:
	"""
	corr = np.correlate(x,y,mode='full')
	pos_ix = np.argmax( np.abs(corr) )
	lag_ix = pos_ix - (corr.size-1)/2
	return lag_ix


def cross_correlogram(x, y, max_lag=100., dt=0.1, plot=True):
	"""
	Returns the cross-correlogram of x and y
	:param x:
	:param y:
	:param max_lag:
	:return:
	"""
	corr = np.correlate(x, y, 'full')
	pos_ix = np.argmax(np.abs(corr))
	maxlag = (corr.size - 1) / 2
	lag = np.arange(-maxlag, maxlag + 1) * dt
	cutoff = [np.where(lag == -max_lag), np.where(lag == max_lag)]

	if plot:
		fig, ax = plt.subplots()
		ax.plot(lag, corr, lw=1)
		ax.set_xlim(lag[cutoff[0]], lag[cutoff[1]])
		ax.axvline(x=lag[pos_ix], ymin=np.min(corr), ymax=np.max(corr), linewidth=1.5, color='c')
		plt.show()


def simple_frequency_spectrum(x):
	"""
	Simple frequency spectrum.

	Very simple calculation of frequency spectrum with no detrending,
	windowing, etc, just the first half (positive frequency components) of
	abs(fft(x))

	Parameters
	----------
	x : array_like
		The input array, in the time-domain.

	Returns
	-------
	spec : array_like
		The frequency spectrum of `x`.

	"""
	spec = np.absolute(np.fft.fft(x))
	spec = spec[:len(x) / 2]  # take positive frequency components
	spec /= len(x)  # normalize
	spec *= 2.0  # to get amplitudes of sine components, need to multiply by 2
	spec[0] /= 2.0  # except for the dc component
	return spec


def euclidean_distance(pos_1, pos_2, N=None):
	"""
	Function to calculate the euclidian distance between two positions

	:param pos_1:
	:param pos_2:
	:param N:
	:return:
	"""
	# If N is not None, it means that we are dealing with a toroidal space,
	# and we have to take the min distance on the torus.
	if N is None:
		dx = pos_1[0] - pos_2[0]
		dy = pos_1[1] - pos_2[1]
	else:
		dx = np.minimum(abs(pos_1[0] - pos_2[0]), N - (abs(pos_1[0] - pos_2[0])))
		dy = np.minimum(abs(pos_1[1] - pos_2[1]), N - (abs(pos_1[1] - pos_2[1])))
	return np.sqrt(dx * dx + dy * dy)


def rescale_signal(val, out_min, out_max):
	in_min = np.min(val)
	in_max = np.max(val)
	return out_min + (val - in_min) * ((out_max - out_min) / (in_max - in_min))


def autocorrelation_function(x):
	"""
	Determine the autocorrelation of signal x

	:param x:
	:return:
	"""

	n = len(x)
	data = np.asarray(x)
	mean = np.mean(data)
	c0 = np.sum((data - mean) ** 2) / float(n)

	def r(h):
		acf_lag = ((data[:n - h] - mean) * (data[h:] - mean)).sum() / float(n) / c0
		return acf_lag #round(acf_lag, 3)

	x = np.arange(n)  # Avoiding lag 0 calculation
	acf_coeffs = map(r, x)
	return acf_coeffs


def get_total_counts(spike_list, time_bin=50.):
	"""
	Determines the total spike counts for neurons with consecutive nonzero counts in bins of the specified size

	:param spike_list: SpikeList object
	:param time_bin: bin width
	:return ctr: number of neurons complying
	:return total_counts: spike count array
	"""
	assert isinstance(spike_list, sg.SpikeList), "Input must be SpikeList object"

	total_counts = []
	ctr = 0
	neuron_ids = []
	for n_train in spike_list.spiketrains:
		tmp = spike_list.spiketrains[n_train].time_histogram(time_bin=time_bin, normalized=False, binary=True)
		if np.mean(tmp) == 1:
			neuron_ids.append(n_train)
			ctr += 1
	print("{0} neurons have nonzero spike counts in bins of size {1}".format(str(ctr), str(time_bin)))
	total_counts1 = []
	for n_id in neuron_ids:
		counts = spike_list.spiketrains[n_id].time_histogram(time_bin=time_bin, normalized=False, binary=False)
		total_counts1.append(counts)
	total_counts.append(total_counts1)
	total_counts = np.array(list(itertools.chain(*total_counts)))

	return neuron_ids, total_counts


def cross_trial_cc(total_counts, display=True):
	"""

	:param total_counts:
	:return:
	"""
	if display:
		print("Computing autocorrelations..")
	units = total_counts.shape[0]

	r = []
	for nn in range(units):
		if display:
			vz.progress_bar(float(nn) / float(units))
		rr = autocorrelation_function(total_counts[nn, :])
		if not np.isnan(np.mean(rr)):
			r.append(rr) #[1:])

	return np.array(r)


def acc_function(x, a, b, tau):
	"""
	Generic exponential function (to use whenever we want to fit an exponential function to data)
	:param x:
	:param a:
	:param b:
	:param tau: decay time constant
	:return:
	"""
	return a * (np.exp(-x / tau) + b)


def err_func(params, x, y, func):
	"""
	Error function for fitting a function

	Parameters
	----------
	params : tuple
		A tuple with the parameters of `func` according to their order of input

	x : float array
		An independent variable.

	y : float array
		The dependent variable.

	func : function
		A function with inputs: `(x, *params)`

	Returns
	-------
	The marginals of the fit to x/y given the params
	"""
	return y - func(x, *params)


def check_signal_dimensions(input_signal, target_signal):
	"""
	Raise error if dimensionalities of signals don't match

	:param input_signal: array
	:param target_signal: array
	:return:
	"""
	if input_signal.shape != target_signal.shape:
		raise RuntimeError("Input shape (%s) and target_signal shape (%s) should be the same." % (input_signal.shape,
		                                                                                          target_signal.shape))


def nrmse(input_signal, target_signal):
	"""
	(from Oger)
	Calculates the normalized root mean square error (NRMSE) of the input signal compared to the target signal.

	:param input_signal: array
	:param target_signal: array
	:return: NRMSE
	"""
	check_signal_dimensions(input_signal, target_signal)
	if len(target_signal) == 1:
		raise NotImplementedError('The NRMSE is not defined for signals of length 1 since they have no variance.')
	input_signal = input_signal.flatten()
	target_signal = target_signal.flatten()

	# Use normalization with N-1, as in matlab
	var = target_signal.std(ddof=1) ** 2

	error = (target_signal - input_signal) ** 2

	return np.sqrt(error.mean() / var)


def nmse(input_signal, target_signal):
	"""
	(from Oger)
	Calculates the normalized mean square error (NMSE) of the input signal compared to the target signal.
	:param input_signal: array
	:param target_signal: array
	:return: RMSE
	"""
	check_signal_dimensions(input_signal, target_signal)
	input_signal = input_signal.flatten()
	targetsignal = target_signal.flatten()

	var = targetsignal.std()**2

	error = (targetsignal - input_signal) ** 2
	return error.mean() / var


def rmse(input_signal, target_signal):
	"""
	(from Oger)
	Calculates the root mean square error (RMSE) of the input signal compared target target signal.
	:param input_signal: array
	:param target_signal: array
	:return: RMSE
	"""
	check_signal_dimensions(input_signal, target_signal)

	error = (target_signal.flatten() - input_signal.flatten()) ** 2
	return np.sqrt(error.mean())


def mse(input_signal, target_signal):
	"""
	(from Oger)
	Calculates the mean square error (MSE) of the input signal compared target signal.
	:param input_signal: array
	:param target_signal: array
	:return: MSE
	"""
	check_signal_dimensions(input_signal, target_signal)
	error = (target_signal.flatten() - input_signal.flatten()) ** 2
	return error.mean()


# def loss_01(input_signal, target_signal):
# 	"""
# 	(from Oger)
# 	Returns the fraction of timesteps where input_signal is unequal to target_signal
# 	:param input_signal: array
# 	:param target_signal: array
# 	:return: loss
# 	"""
# 	check_signal_dimensions(input_signal, target_signal)
# 	return np.mean(np.any(input_signal != target_signal, 1))


# def cosine(input_signal, target_signal):
# 	"""
# 	(from Oger)
# 	Compute cosine of the angle between two vectors. This error measure measures the extent to which two vectors
# 	point in the same direction. A value of 1 means complete alignment, a value of 0 means the vectors are orthogonal.
# 	:param input_signal: array
# 	:param target_signal: array
# 	:return: cos
# 	"""
# 	check_signal_dimensions(input_signal, target_signal)
# 	return float(np.dot(input_signal, target_signal)) / (np.linalg.norm(input_signal) * np.linalg.norm(
# 			target_signal))


# def ce(input_signal, target_signal):
# 	"""
# 	(from Oger)
# 	Compute cross-entropy loss function. Returns the negative log-likelyhood of the target_signal labels as predicted by
# 	the input_signal values.
# 	:param input_signal: array
# 	:param target_signal: array
# 	:return:
# 	"""
# 	check_signal_dimensions(input_signal, target_signal)
#
# 	if np.rank(target_signal)>1 and target_signal.shape[1] > 1:
# 		error = np.sum(-np.log(input_signal[target_signal == 1]))
#
# 		if np.isnan(error):
# 			inp = input_signal[target_signal == 1]
# 			inp[inp == 0] = float(np.finfo(input_signal.dtype).tiny)
# 			error = -np.sum(np.log(inp))
# 	else:
# 		error = -np.sum(np.log(input_signal[target_signal == 1]))
# 		error -= np.sum(np.log(1 - input_signal[target_signal == 0]))
#
# 		if np.isnan(error):
# 			inp = input_signal[target_signal == 1]
# 			inp[inp == 0] = float(np.finfo(input_signal.dtype).tiny)
# 			error = -np.sum(np.log(inp))
# 			inp = 1 - input_signal[target_signal == 0]
# 			inp[inp == 0] = float(np.finfo(input_signal.dtype).tiny)
# 			error -= np.sum(np.log(inp))
#
# 	return error


# TODO when classes are finished, integrate this function and update arguments
def compute_isi_stats(spike_list, summary_only=True, display=True):
	"""
	Compute all relevant isi metrics
	:param spike_list: SpikeList object
	:param summary_only: bool - store only the summary statistics or all the data (memory!)
	:param display: bool - display progress / time
	:return: dictionary with all the relevant data
	"""
	if display:
		print("\nAnalysing inter-spike intervals...")
		t_start = time.time()
	results = dict()

	results['cvs'] 		= spike_list.cv_isi(float_only=True)
	results['lvs'] 		= spike_list.local_variation()
	results['lvRs'] 	= spike_list.local_variation_revised(float_only=True)
	results['ents'] 	= spike_list.isi_entropy(float_only=True)
	results['iR'] 		= spike_list.instantaneous_regularity(float_only=True)
	results['cvs_log'] 	= spike_list.cv_log_isi(float_only=True)
	results['isi_5p'] 	= spike_list.isi_5p(float_only=True)
	results['ai'] 		= spike_list.adaptation_index(float_only=True)

	if not summary_only:
		results['isi'] 	= np.array(list(itertools.chain(*spike_list.isi())))
	else:
		results['isi'] = []
		cvs 	= results['cvs']
		lvs 	= results['lvs']
		lvRs 	= results['lvRs']
		H 		= results['ents']
		iRs 	= results['iR']
		cvs_log = results['cvs_log']
		isi_5p 	= results['isi_5p']
		ai 		= results['ai']

		results['cvs'] 		= (np.mean(cvs), np.var(cvs))
		results['lvs'] 		= (np.mean(lvs), np.var(lvs))
		results['lvRs'] 	= (np.mean(lvRs), np.var(lvRs))
		results['ents'] 	= (np.mean(H), np.var(H))
		results['iR'] 		= (np.mean(iRs), np.var(iRs))
		results['cvs_log'] 	= (np.mean(cvs_log), np.var(cvs_log))
		results['isi_5p'] 	= (np.mean(isi_5p), np.var(isi_5p))
		results['ai'] 		= (np.mean(ai), np.var(ai))

	if display:
		print("Elapsed Time: {0} s".format(str(round(time.time()-t_start, 3))))

	return results

# TODO when classes are finished, integrate this function and update arguments
def compute_spike_stats(spike_list, time_bin=1., summary_only=False, display=False):
	"""
	Compute relevant statistics on population firing activity (f. rates, spike counts)
	:param spike_list: SpikeList object
	:param time_bin: float - bin width to determine spike counts
	:param summary_only: bool - store only the summary statistics or all the data (memory!)
	:param display: bool - display progress / time
	:return: dictionary with all the relevant data
	"""
	# TODO computing FANO FACTOR with a bin width of 1. doesn't make sense, no?
	time_bin = 50.
	if display:
		print("\nAnalysing spiking activity...")
		t_start = time.time()
	results = {}
	rates = np.array(spike_list.mean_rates())
	rates = rates[~np.isnan(rates)]
	counts = spike_list.spike_counts(dt=time_bin, normalized=False, binary=False)
	ffs = np.array(spike_list.fano_factors(time_bin))
	if summary_only:
		results['counts'] 			= (np.mean(counts[~np.isnan(counts)]), np.var(counts[~np.isnan(counts)]))
		results['mean_rates'] 		= (np.mean(rates), np.var(rates))
		results['ffs'] 				= (np.mean(ffs[~np.isnan(ffs)]), np.var(ffs[~np.isnan(ffs)]))
		results['corrected_rates'] 	= (np.mean(rates[np.nonzero(rates)[0]]), np.std(rates[np.nonzero(rates)[0]]))
	else:
		results['counts'] 			= counts
		results['mean_rates'] 		= rates
		results['corrected_rates'] 	= rates[np.nonzero(rates)[0]]
		results['ffs'] 				= ffs[~np.isnan(ffs)]
		results['spiking_neurons'] 	= spike_list.id_list
	if display:
		print("Elapsed Time: {0} s".format(str(round(time.time() - t_start, 3))))
	return results


# def time_resolved_fano_factor(spike_list, time_points, **params):
# 	"""
# 	Regress the spike-count variance versus the mean and report the slope (FF).
# 	Analysis is done at multiple time points.
# 	The distribution of mean counts is matched (via downselection), so that all time points
# 	have the same distribution of mean counts
# 	:param spike_list: SpikeList object - binary spike count vectors will be extracted. The data analyzed takes
# 	the form of a binary matrix with 1 row per trial / neuron and 1 column per ms.
# 	:param time_points: time points to report
# 	:param params: [dict] - if any of the fields is not provided, it will be replaced by the default value
# 		- 'box_width' - width of sliding window
# 		- 'match_reps' - number of random choices regarding which points to throw away when matching distributions
# 		- 'bin_spacing' - bin width when computing distributions of mean counts
# 		- 'align_time' - time of event that data are aligned to (output times are expressed relative to this)
# 		- 'weighted_regression' - self-explanatory
# 	:return results: [dict] -
# 		- 'fano_factor' - FF for each time (after down-sampling to match distribution across times)
# 		- 'fano_95CI' - 95% confidence intervals on the FF
# 		- 'scatter_data' - data for variance VS mean scatter plot
# 		- 'fano_factors' - FF for all data points (no down-sampling or matching)
# 		- 'fano_all_95CI' - 95% confidence intervals for the above
# 		- ''
# 	Based on Churchland et al. (2010) Stimulus onset quenches neural variability: a widespread cortical phenomenon.
# 	"""
# 	#TODO - incomplete (attempt to adapt from Matlab code)
# 	default_params = {'box_width': 50, 'match_reps': 10, 'bin_spacing': 0.25, 'align_time': 0, 'weighted_regression':
# 		True}
# 	parameter_fields = ['box_width', 'match_reps', 'bin_spacing', 'align_time', 'weighted_regression']
# 	for k, v in default_params.items():
# 		if params.has_key(k):
# 			default_params[k] = params[k]
#
# 	# Acquire binary count data
# 	counts = spike_list.spike_counts(dt=1., normalized=False, binary=True)
# 	weighting_epsilon = 1. * default_params['box_width'] / 1000.
#
# 	# main
# 	max_rate = 0   # keep track of max rate across all times / conditions
# 	# trial_count =
# 	t_start = time_points - np.floor(default_params['box_width']/2.) + 1
# 	t_end = time_points - np.ceil(default_params['box_width']/2.) + 1

# TODO when classes are finished, integrate this function and update arguments
def compute_synchrony(spike_list, n_pairs=500, time_bin=1., tau=20., time_resolved=False, display=True, depth=4):
	"""
	Apply various metrics of spike train synchrony
		Note: Has dependency on PySpike package.
	:param spike_list: SpikeList object
	:param n_pairs: number of neuronal pairs to consider in the pairwise correlation measures
	:param time_bin: time_bin (for pairwise correlations)
	:param tau: time constant (for the van Rossum distance)
	:param time_resolved: bool - perform time-resolved synchrony analysis (PySpike)
	:param summary_only: bool - retrieve only a summary of the results
	:param complete: bool - use all metrics or only the ccs (due to computation time, memory)
	:param display: bool - display elapsed time message
	:return results: dict
	"""
	if display:
		print("\nAnalysing spike synchrony...")
		t_start = time.time()

	if has_pyspike:
		spike_trains = sg.to_pyspike(spike_list)
	results = dict()

	if time_resolved and has_pyspike:
		results['SPIKE_sync_profile'] 	= spk.spike_sync_profile(spike_trains)
		results['ISI_profile'] 			= spk.isi_profile(spike_trains)
		results['SPIKE_profile'] 		= spk.spike_profile(spike_trains)

	if depth == 1 or depth == 3:
		results['ccs_pearson'] 	= spike_list.pairwise_pearson_corrcoeff(n_pairs, time_bin=time_bin, all_coef=False)
		ccs 					= spike_list.pairwise_cc(n_pairs, time_bin=time_bin)
		results['ccs'] 			= (np.mean(ccs), np.var(ccs))

		if depth >= 3:
			results['d_vp'] = spike_list.distance_victorpurpura(n_pairs, cost=0.5)
			results['d_vr'] = np.mean(spike_list.distance_van_rossum(tau=tau))
			if has_pyspike:
				results['ISI_distance'] 		= spk.isi_distance(spike_trains)
				results['SPIKE_distance'] 		= spk.spike_distance(spike_trains)
				results['SPIKE_sync_distance'] 	= spk.spike_sync(spike_trains)
	else:
		results['ccs_pearson'] 	= spike_list.pairwise_pearson_corrcoeff(n_pairs, time_bin=time_bin, all_coef=True)
		results['ccs'] 			= spike_list.pairwise_cc(n_pairs, time_bin=time_bin)

		if depth >= 3:
			results['d_vp'] 		= spike_list.distance_victorpurpura(n_pairs, cost=0.5)
			results['d_vr'] 		= spike_list.distance_van_rossum(tau=tau)
			if has_pyspike:
				results['ISI_distance_matrix'] 		= spk.isi_distance_matrix(spike_trains)
				results['SPIKE_distance_matrix'] 	= spk.spike_distance_matrix(spike_trains)
				results['SPIKE_sync_matrix'] 		= spk.spike_sync_matrix(spike_trains)
				results['ISI_distance'] 			= spk.isi_distance(spike_trains)
				results['SPIKE_distance'] 			= spk.spike_distance(spike_trains)
				results['SPIKE_sync_distance']		= spk.spike_sync(spike_trains)

	if display:
		print("Elapsed Time: {0} s".format(str(round(time.time() - t_start, 3))))
	return results

# TODO add more comment, what stats exactly are computed?
def compute_analog_stats(population, parameter_set, variable_names, analysis_interval=None, plot=False):
	"""
	Extract, analyse and store analog data
	:param population: Population object
	:param parameter_set: full ParameterSet object
	:param variable_names: names of the variables of interest ?
	:param analysis_interval: time interval to analyse
	:param plot: bool
	:return results: dict
	"""
	results = dict()
	pop_idx = parameter_set.net_pars.pop_names.index(population.name)
	if not population.analog_activity:
		results['recorded_neurons'] = []
		print("No analog variables recorded from {0}".format(str(population.name)))
		return results
	else:
		if isinstance(population.analog_activity, list):
			for idx, nn in enumerate(variable_names):
				locals()[nn] = population.analog_activity[idx]
				assert isinstance(locals()[nn], sg.AnalogSignalList), "Analog activity should be saved as " \
				                                                    "AnalogSignalList"
		else:
			locals()[variable_names[0]] = population.analog_activity

		if plot:
			# pick one neuron to look at its signals (to plot)
			single_idx = np.random.permutation(locals()[variable_names[0]].id_list())[0]
			reversals = []

		# store the ids of the recorded neurons
		results['recorded_neurons'] = locals()[variable_names[0]].id_list()

		for idx, nn in enumerate(variable_names):
			if analysis_interval is not None:
				locals()[nn] = locals()[nn].time_slice(analysis_interval[0], analysis_interval[1])

			time_axis = locals()[nn].time_axis()

			if plot and 'E_{0}'.format(nn[-2:]) in parameter_set.net_pars.neuron_pars[pop_idx]:
				reversals.append(parameter_set.net_pars.neuron_pars[pop_idx]['E_{0}'.format(nn[-2:])])

			if len(results['recorded_neurons']) > 1:
				results['mean_{0}'.format(nn)] = locals()[nn].mean(axis=1)
				results['std_{0}'.format(nn)] = locals()[nn].std(axis=1)

		if len(results['recorded_neurons']) > 1:
			results['mean_I_ex'] = []
			results['mean_I_in'] = []
			results['EI_CC'] = []
			for idxx, nnn in enumerate(results['recorded_neurons']):
				for idx, nn in enumerate(variable_names):
					locals()['signal_' + nn] = locals()[nn].analog_signals[nnn].signal
				if ('signal_V_m' in locals()) and ('signal_g_ex' in locals()) and ('signal_g_in' in locals()):
					E_ex = parameter_set.net_pars.neuron_pars[pop_idx]['E_ex']
					E_in = parameter_set.net_pars.neuron_pars[pop_idx]['E_in']
					E_current = locals()['signal_g_ex'] * (locals()['signal_V_m'] - E_ex)
					E_current /= 1000.
					I_current = locals()['signal_g_in'] * (locals()['signal_V_m'] - E_in)
					I_current /= 1000.
					results['mean_I_ex'].append(np.mean(E_current))
					results['mean_I_in'].append(np.mean(I_current))
					cc = np.corrcoef(E_current, I_current)
					results['EI_CC'].append(np.unique(cc[cc != 1.]))
				elif ('signal_I_ex' in locals()) and ('signal_I_in' in locals()):
					results['mean_I_ex'].append(np.mean(locals()['signal_I_ex'])/1000.)  # /1000. to get results in nA
					results['mean_I_in'].append(np.mean(locals()['signal_I_in'])/1000.)
					cc = np.corrcoef(locals()['signal_I_ex'], locals()['signal_I_in'])
					results['EI_CC'].append(np.unique(cc[cc != 1.]))
			results['EI_CC'] = np.array(list(itertools.chain(*results['EI_CC'])))
			# remove nans and infs
			results['EI_CC'] = np.extract(np.logical_not(np.isnan(results['EI_CC'])), results['EI_CC'])
			results['EI_CC'] = np.extract(np.logical_not(np.isinf(results['EI_CC'])), results['EI_CC'])

		if 'V_m' in variable_names and plot:
			results['single_Vm'] = locals()['V_m'].analog_signals[single_idx].signal
			results['single_idx'] = single_idx
			results['time_axis'] = locals()['V_m'].analog_signals[single_idx].time_axis()
			variable_names.remove('V_m')

			for idxx, nnn in enumerate(variable_names):
				cond = locals()[nnn].analog_signals[single_idx].signal

				if 'I_ex' in variable_names:
					results['I_{0}'.format(nnn[-2:])] = cond
					results['I_{0}'.format(nnn[-2:])] /= 1000.
				elif 'g_ex' in variable_names:
					rev = reversals[idxx]
					results['I_{0}'.format(nnn[-2:])] = cond * (results['single_Vm'] - rev)
					results['I_{0}'.format(nnn[-2:])] /= 1000.
				else:
					results['single_{0}'.format(nnn)] = cond
		return results


def compute_dimensionality(activity_matrix, pca_obj=None, label='', plot=False, display=True, save=False):
	"""
	Measure the effective dimensionality of population responses. Based on Abbott et al. (). Interactions between
	intrinsic and stimilus-evoked activity in recurrent neural networks
	:param activity_matrix: matrix to analyze (NxT)
	:param pca_obj: if pre-computed, otherwise None
	:return:
	"""
	assert(check_dependency('sklearn')), "PCA analysis requires scikit learn"
	if display:
		print("Determining effective dimensionality..")
		t_start = time.time()
	if pca_obj is None:
		pca_obj = sk.PCA(n_components=np.shape(activity_matrix)[0])
	if not hasattr(pca_obj, "explained_variance_ratio_"):
		pca_obj.fit(activity_matrix.T)
	# Dimensionality
	dimensionality = 1. / np.sum((pca_obj.explained_variance_ratio_ ** 2))
	if display:
		print("Effective dimensionality = {0}".format(str(round(dimensionality, 2))))
		print("Elapsed Time: {0} s".format(str(round(time.time() - t_start, 3))))
	if plot:
		X = pca_obj.fit_transform(activity_matrix.T).T
		vz.plot_dimensionality(dimensionality, pca_obj, X, data_label=label, display=display, save=save)
	return dimensionality


def compute_timescale(activity_matrix, time_axis, max_lag=1000, method=0):
	"""
	Determines the time scale of fluctuations in the population activity

	:param activity_matrix: np.array with size NxT
	:param time_axis:
	:param max_lag:
	?? :param method: based on autocorrelation (0) or on power spectra (1)
	:return:
	"""
	# TODO modify / review / extend / correct / update
	time_scales = []
	final_acc 	= []
	errors 		= []
	acc 		= cross_trial_cc(activity_matrix)
	initial_guess = 1., 0., 10.
	for n_signal in range(acc.shape[0]):
		# TODO err_func not defined here..
		fit, _ = opt.leastsq(err_func, initial_guess, args=(time_axis, acc[n_signal, :max_lag], acc_function))

		if fit[2] > 0:
			error_rates = np.sum((acc[n_signal, :max_lag] - acc_function(time_axis[:max_lag], *fit)) ** 2)
			print("Timescale [ACC] = {0} ms / error = {1}".format(str(fit[2]), str(error_rates)))
			time_scales.append(fit[2])
			errors.append(error_rates)

			final_acc.append(acc[n_signal, :max_lag])
	final_acc = np.array(final_acc)

	mean_fit, _ = opt.leastsq(err_func, initial_guess, args=(time_axis, np.mean(final_acc, 0), acc_function))
	error_rates = np.sum((np.mean(final_acc, 0) - acc_function(time_axis, *mean_fit)) ** 2)
	print("Timescale = {0} ms / error = {1}".format(str(mean_fit[2]), str(error_rates)))
	print("Accepted dimensions = {0}".format(str(float(final_acc.shape[0]) / float(acc.shape[0]))))

	return final_acc, mean_fit, acc_function, time_scales


def dimensionality_reduction(state_matrix, data_label='', labels=None, metric=None, standardize=True, plot=True,
                             colormap='jet', display=True, save=False):
	"""
	Fit and test various algorithms, to extract a reasonable 3D projection of the data for visualization
	:param activity_matrix: matrix to analyze (NxT)
	:param metric: [str] metric to use (if None all will be tested)
	:param standardize:
	:param plot:
	:param display:
	:param save:
	:return:
	"""
	# TODO extend and test - and include in the analyse_activity_dynamics function
	metrics = ['PCA', 'FA', 'LLE', 'IsoMap', 'Spectral', 'MDS', 't-SNE']
	if metric is not None:
		assert(metric in metrics), "Incorrect metric"
		metrics = [metric]
	if labels is None:
		raise TypeError("Please provide stimulus labels")
	else:
		n_elements = np.unique(labels)
	colors_map = vz.get_cmap(N=len(n_elements), cmap=colormap)

	for met in metrics:
		if met == 'PCA':
			print("\nPrincipal Component Analysis")
			t_start = time.time()
			pca_obj = sk.PCA(n_components=3)
			X_r = pca_obj.fit(state_matrix.T).transform(state_matrix.T)
			if display:
				print("Elapsed time: {0} s".format(str(time.time() - t_start)))
				print("Explained Variance (first 3 components): %s" % str(pca_obj.explained_variance_ratio_))
			exp_var = [round(n, 2) for n in pca_obj.explained_variance_ratio_]

			if plot:
				fig1 = pl.figure()
				ax11 = fig1.add_subplot(111, projection='3d')
				ax11.set_xlabel(r'$PC_{1}$')
				ax11.set_ylabel(r'$PC_{2}$')
				ax11.set_zlabel(r'$PC_{3}$')
				fig1.suptitle(r'${0} - PCA (var = {1})$'.format(str(data_label), str(exp_var)), fontsize=20)
				vz.scatter_projections(X_r, labels, colors_map, ax=ax11)
				if save:
					fig1.savefig(save + data_label + '_PCA.pdf')
				if display:
					pl.show(False)
		elif met == 'FA':
			print("\nFactor Analysis")
			t_start = time.time()
			fa2 = sk.FactorAnalysis(n_components=len(n_elements))
			state_fa = fa2.fit_transform(state_matrix.T)
			score = fa2.score(state_matrix.T)
			if display:
				print("Elapsed time: {0} s / Score (NLL): {1}".format(str(time.time() - t_start), str(score)))
			if plot:
				fig2 = pl.figure()
				fig2.suptitle(r'Factor Analysis')
				ax21 = fig2.add_subplot(111, projection='3d')
				vz.scatter_projections(state_fa, labels, colors_map, ax=ax21)
				if save:
					fig2.savefig(save + data_label + '_FA.pdf')
				if display:
					pl.show(False)
		elif met == 'LLE':
			print("\nLocally Linear Embedding")
			if plot:
				fig3 = pl.figure()
				fig3.suptitle(r'Locally Linear Embedding')

			methods = ['standard']#, 'ltsa', 'hessian', 'modified']
			labels = ['LLE']#, 'LTSA', 'Hessian LLE', 'Modified LLE']

			for i, method in enumerate(methods):
				t_start = time.time()
				fit_obj = man.LocallyLinearEmbedding(n_neighbors=199, n_components=3, eigen_solver='auto',
				                                 method=method, n_jobs=-1)
				Y = fit_obj.fit_transform(state_matrix.T)
				if display:
					print("\t{0} - {1} s / Reconstruction error = {2}".format(method, str(time.time() - t_start), str(
					fit_obj.reconstruction_error_)))
				if plot:
					ax = fig3.add_subplot(2, 2, i + 1, projection='3d')
					ax.set_title(method)
					vz.scatter_projections(Y, labels, colors_map, ax=ax)
			if plot and save:
				fig3.savefig(save + data_label + '_LLE.pdf')
			if plot and display:
				pl.show(False)
		elif met == 'IsoMap':
			print("\nIsoMap Embedding")
			t_start = time.time()
			iso_fit = man.Isomap(n_neighbors=199, n_components=3, eigen_solver='auto', path_method='auto',
			                 neighbors_algorithm='auto', n_jobs=-1)
			iso = iso_fit.fit_transform(state_matrix.T)
			if display:
				# print("Elapsed time: {0} s / Reconstruction error = {1}".format(str(time.time() - t_start), str(
				# 		iso_fit.reconstruction_error_)))
				print("Elapsed time: {0} s / Reconstruction error = ".format(str(time.time() - t_start)))
			if plot:
				fig4 = pl.figure()
				fig4.suptitle(r'IsoMap Embedding')
				ax41 = fig4.add_subplot(111, projection='3d')
				vz.scatter_projections(iso, labels, colors_map, ax=ax41)
				if save:
					fig4.savefig(save + data_label + '_IsoMap.pdf')
				if display:
					pl.show(False)
		elif met == 'Spectral':
			print("\nSpectral Embedding")
			fig5 = pl.figure()
			fig5.suptitle(r'Spectral Embedding')

			affinities = ['nearest_neighbors', 'rbf']
			for i, n in enumerate(affinities):
				t_start = time.time()
				spec_fit = man.SpectralEmbedding(n_components=3, affinity=n, n_jobs=-1)
				spec = spec_fit.fit_transform(state_matrix.T)
				if display:
					# print("\t{0} - {1} s / Reconstruction error = {2}".format(n, str(time.time() - t_start), str(
					# 	spec_fit.reconstruction_error_)))
					print("Elapsed time: {0} s / Reconstruction error = ".format(str(time.time() - t_start)))
				if plot:
					ax = fig5.add_subplot(1, 2, i + 1, projection='3d')
					# ax.set_title(n)
					vz.scatter_projections(spec, labels, colors_map, ax=ax)
					# pl.imshow(spec_fit.affinity_matrix_)
				if plot and save:
					fig5.savefig(save + data_label + '_SE.pdf')
				if plot and display:
					pl.show(False)
		elif met == 'MDS':
			print("\nMultiDimensional Scaling")
			t_start = time.time()
			mds = man.MDS(n_components=3, n_jobs=-1)
			mds_fit = mds.fit_transform(state_matrix.T)
			if display:
				# print("Elapsed time: {0} s / Reconstruction error = {1}".format(str(time.time() - t_start), str(
				# 		mds.reconstruction_error_)))
				print("Elapsed time: {0} s / Reconstruction error = ".format(str(time.time() - t_start)))
			if plot:
				fig6 = pl.figure()
				fig6.suptitle(r'MultiDimensional Scaling')
				ax61 = fig6.add_subplot(111, projection='3d')
				vz.scatter_projections(mds_fit, labels, colors_map, ax=ax61)
				if save:
					fig6.savefig(save + data_label + '_MDS.pdf')
		elif met == 't-SNE':
			print("\nt-SNE")
			t_start = time.time()
			tsne = man.TSNE(n_components=3, init='pca')
			tsne_emb = tsne.fit_transform(state_matrix.T)
			if display:
				# print("Elapsed time: {0} s / Reconstruction error = {1}".format(str(time.time() - t_start), str(
				# 		tsne.reconstruction_error_)))
				print("Elapsed time: {0} s / Reconstruction error = ".format(str(time.time() - t_start)))
			if plot:
				fig7 = pl.figure()
				fig7.suptitle(r't-SNE')
				ax71 = fig7.add_subplot(111, projection='3d')
				vz.scatter_projections(tsne_emb, labels, colors_map, ax=ax71)
				if save:
					fig7.savefig(save + data_label + '_t_SNE.pdf')
				if display:
					pl.show(False)
		else:
			raise NotImplementedError("Metric {0} is not currently implemented".format(met))


# TODO when classes are finished, integrate this function and update arguments
def characterize_population_activity(population_object, parameter_set, analysis_interval, prng=None, epochs=None,
									 plot=True, display=True, save=False, color_map="coolwarm", color_subpop=False,
									 analysis_pars=None):
	"""
	Compute all the relevant metrics of recorded activity (spiking and analog signals), providing
	a thorough characterization and quantification of population dynamics

	:return results: dict
	:param population_object: Population or Network object whose activity should be analyzed
	:param parameter_set: complete ParameterSet
	:param analysis_interval: list or tuple with [start_time, stop time] specifying the time interval to analyse
	:param prng: numpy.random object for precise experiment reproduction
	:return:
	"""
	if analysis_pars is None:
		raise ValueError("Analysis parameters are required for characterizing population activity!")

	ap = analysis_pars
	pars_activity = ap.population_activity
	subpop_names = None

	if isinstance(population_object, na.Population):
		gids = None
		base_population_object = None
	elif isinstance(population_object, na.Network):
		new_population = population_object.merge_subpopulations(sub_populations=population_object.populations,
																name='Global', merge_activity=True, store=False)
		gids = [n.id_list for n in list(sg.iterate_obj_list(population_object.spiking_activity))]
		subpop_names = population_object.population_names

		if not gids:
			gids = [np.array(n.gids) for n in list(sg.iterate_obj_list(population_object.populations))]

		base_population_object 	= population_object
		population_object 		= new_population
	else:
		raise TypeError("Incorrect population object. Must be Population or Network object")

	results = {'spiking_activity': {}, 'analog_activity': {}, 'metadata': {'population_name': population_object.name}}

	########################################################################################################
	# Spiking activity analysis
	if population_object.spiking_activity:
		spike_list = population_object.spiking_activity
		assert isinstance(spike_list, sg.SpikeList), "Spiking activity should be SpikeList object"
		spike_list = spike_list.time_slice(analysis_interval[0], analysis_interval[1])

		results['spiking_activity'].update(compute_spikelist_metrics(spike_list, population_object.name, ap))

		if plot and ap.depth % 2 == 0: # save all data
			vz.plot_isi_data(results['spiking_activity'][population_object.name],
										data_label=population_object.name, color_map=color_map, location=0,
							  			display=display, save=save)
			if has_pyspike:
				vz.plot_synchrony_measures(results['spiking_activity'][population_object.name],
											label=population_object.name, time_resolved=pars_activity.time_resolved,
											epochs=epochs, display=display, save=save)

		if pars_activity.time_resolved:
			# *** Averaged time-resolved metrics
			results['spiking_activity'][population_object.name].update(compute_time_resolved_statistics(spike_list,
			                        label=population_object.name, time_bin=pars_activity.time_bin, epochs=epochs,
									window_len=pars_activity.window_len, color_map=color_map,
									display=display, plot=plot, save=save))
		if plot:
			results['metadata']['spike_list'] = spike_list

		if color_subpop and subpop_names:
			results['metadata'].update({'sub_population_names': subpop_names, 'sub_population_gids': gids,
									'spike_data_file': ''})

		if gids and ap.depth >= 3:
			if len(gids) == 2:
				locations = [-1, 1]
			else:
				locations = [0 for _ in range(len(gids))]

			for indice, name in enumerate(subpop_names):
				results['spiking_activity'].update(compute_spikelist_metrics(spike_list, name, ap))
				if plot and ap.depth % 2 == 0: # save all data
					vz.plot_isi_data(results['spiking_activity'][name], data_label=name, color_map=color_map,
												location=locations[indice], display=display, save=save)
					if has_pyspike:
						vz.plot_synchrony_measures(results['spiking_activity'][name], label=name,
						                        			  time_resolved=pars_activity.time_resolved,
															  display=display, save=save)
				if pars_activity.time_resolved:
					# *** Averaged time-resolved metrics
					results['spiking_activity'][name].update(compute_time_resolved_statistics(spike_list,
			                        label=population_object.name, time_bin=pars_activity.time_bin,
			                        epochs=epochs, color_map=color_map, display=display,
									plot=plot, save=save, window_len=pars_activity.window_len))

	else:
		print("Warning, the network is not spiking or no spike recording devices were attached.")

	# Analog activity analysis
	if population_object.analog_activity and base_population_object is not None:
		results['analog_activity'] = {}
		for pop_n, pop in enumerate(base_population_object.populations):
			if bool(pop.analog_activity):
				results['analog_activity'].update({pop.name: {}})
				pop_idx = parameter_set.net_pars.pop_names.index(pop.name)
				if parameter_set.net_pars.analog_device_pars[pop_idx] is None:
					break
				variable_names = list(np.copy(parameter_set.net_pars.analog_device_pars[pop_idx]['record_from']))

				results['analog_activity'][pop.name].update(compute_analog_stats(pop, parameter_set, variable_names,
				                                                                 analysis_interval, plot))
	if plot:
		vz.plot_state_analysis(parameter_set, results, summary_only=bool(ap.depth % 2 != 0),
										  start=analysis_interval[0], stop=analysis_interval[1],
										  display=display, save=save)
	return results


def epoch_based_analysis(population_object, epochs):
	"""
	Analyse population activity on a trial-by-trial basis
	:return:
	"""
	pass
	# unique_labels = np.unique(epochs.keys())
	# full_data_set = {lab: {} for lab in unique_labels}
	# for epoch_label, epoch_times in epochs.items():
	# 	full_data_set[lab].update({epoch_label: spike_list.time_slice(epoch_times[0], epoch_times[1])})


def analyse_activity_dynamics(activity_matrix, epochs=None, label='', plot=False, display=False, save=False):
	"""
	Perform standard analyses on population activity
	:param activity_matrix: NxT continuous activity matrix
	:return: results dictionary
	"""
	if isinstance(activity_matrix, sg.AnalogSignalList):
		activity_matrix = activity_matrix.as_array()
	assert(isinstance(activity_matrix, np.ndarray)), "Activity matrix must be numpy array or AnalogSignalList"
	assert(check_dependency('sklearn')), "Scikits-learn is necessary for this analysis"

	results = {}

	pca_obj = sk.PCA(n_components=activity_matrix.shape[0])
	X = pca_obj.fit_transform(activity_matrix.T)
	print("Explained Variance (first 3 components): %s" % str(pca_obj.explained_variance_ratio_[:3]))
	results.update({'dimensionality': compute_dimensionality(activity_matrix, pca_obj=pca_obj, display=True)})
	if plot:
		vz.plot_dimensionality(results, pca_obj, X, data_label=label, display=display, save=save)
	if epochs is not None:
		for epoch_label, epoch_time in epochs.items():
			# print epoch_label
			resp = activity_matrix[:, int(epoch_time[0]):int(epoch_time[1])]

			results.update({epoch_label: {}})
			results[epoch_label].update(analyse_activity_dynamics(resp, epochs=None, label=epoch_label, plot=False,
			                                                     display=False, save=False))
	# TODO extend to other standard measurements
	return results


def compute_time_resolved_statistics(spike_list, label='', time_bin=1., window_len=100, epochs=None,
                                     color_map='colorwarm', display=True, plot=False, save=False):
	"""

	:param spike_list:
	:param label:
	:param time_bin:
	:param window_len:
	:param epochs:
	:param color_map:
	:param display:
	:param plot:
	:param save:
	:return:
	"""
	time_axis = spike_list.time_axis(time_bin=time_bin)
	steps = len(list(sg.moving_window(time_axis, window_len)))
	mw = sg.moving_window(time_axis, window_len)
	results = dict()
	print("\nAnalysing activity in moving window..")

	for n in range(steps):
		if display:
			vz.progress_bar(float(float(n) / steps))
		time_window = mw.next()
		local_list = spike_list.time_slice(t_start=min(time_window), t_stop=max(time_window))
		local_isi = compute_isi_stats(local_list, summary_only=True, display=False)
		local_spikes = compute_spike_stats(local_list, time_bin=time_bin, summary_only=True, display=False)

		if n == 0:
			rr = {k + '_profile': [] for k in local_isi.keys()}
			rr.update({k + '_profile': [] for k in local_spikes.keys()})
		else:
			for k in local_isi.keys():
				rr[k + '_profile'].append(local_isi[k])
				if n == steps - 1:
					results.update({k + '_profile': rr[k + '_profile']})
			for k in local_spikes.keys():
				rr[k + '_profile'].append(local_spikes[k])
				if n == steps - 1:
					results.update({k + '_profile': rr[k + '_profile']})
	if plot:
		vz.plot_averaged_time_resolved(results, spike_list, label=label, epochs=epochs,
		                            color_map=color_map, display=display, save=save)

	return results


def compute_spikelist_metrics(spike_list, label, analysis_pars):
	"""

	:return:
	"""
	ap = analysis_pars
	pars_activity = ap.population_activity
	results = {label: {}}

	# TODO when isi_stats and spike_stats are finished / integrated into classes, update summary_only parameters
	# ISI statistics
	results[label].update(compute_isi_stats(spike_list, summary_only=bool(ap.depth % 2 != 0)))

	# Firing activity statistics
	results[label].update(compute_spike_stats(spike_list, time_bin=pars_activity.time_bin,
											  summary_only=bool(ap.depth % 2 != 0)))

	# Synchrony measures
	results[label].update(compute_synchrony(spike_list, n_pairs=pars_activity.n_pairs,
											time_bin=pars_activity.time_bin, tau=pars_activity.tau,
											time_resolved=pars_activity.time_resolved, depth=ap.depth))

	return results


def single_neuron_dcresponse(population_object, parameter_set, start=None, stop=None, plot=True, display=True,
							 save=False):
	"""
	extract relevant data and analyse single neuron fI curves and other measures
	:param population_object:
	:param parameter_set:
	:param start:
	:param stop:
	:param plot:
	:param display:
	:param save:
	:return:
	"""
	input_amplitudes = parameter_set.encoding_pars.generator.model_pars[0]['amplitude_values']
	input_times = parameter_set.encoding_pars.generator.model_pars[0]['amplitude_times']

	spike_list = population_object.spiking_activity
	vm_list = population_object.analog_activity[0]

	# extract single response:
	min_spk = spike_list.first_spike_time()
	idx = max(np.where(input_times < min_spk)[0])
	interval = [input_times[idx], input_times[idx+1]]
	single_spk = spike_list.time_slice(interval[0], interval[1])
	single_vm = vm_list.time_slice(interval[0], interval[1])

	if len(population_object.analog_activity) > 1:
		other_analogs = [x for x in population_object.analog_activity[1:]]
		single_analogs = [x.time_slice(interval[0], interval[1]) for x in other_analogs]
	else:
		other_analogs = None

	if start is not None and stop is not None:
		spike_list = spike_list.time_slice(start, stop)
		vm_list = vm_list.time_slice(start, stop)

	output_rate = []
	for idx, t in enumerate(input_times):
		if idx >= 1:
			output_rate.append(spike_list.mean_rate(input_times[idx-1], t))

	isiis = single_spk.isi()[0]
	k = 2 # disregard initial transient
	n = len(isiis)
	l = []
	for iddx, nn in enumerate(isiis):
		if iddx > k:
			l.append((nn - isiis[iddx-1]) / (nn + isiis[iddx-1]))
	A = np.sum(l)/(n-k-1)
	A2 = np.mean(l)

	if plot:
		fig = pl.figure()
		ax1 = pl.subplot2grid((10, 3), (0, 0), rowspan=6, colspan=1)
		ax2 = pl.subplot2grid((10, 3), (0, 1), rowspan=6, colspan=1)
		ax3 = pl.subplot2grid((10, 3), (0, 2), rowspan=6, colspan=1)
		ax4 = pl.subplot2grid((10, 3), (7, 0), rowspan=3, colspan=3)

		props = {'xlabel': r'I [pA]', 'ylabel': r'Firing Rate [spikes/s]'}
		vz.plot_fI_curve(input_amplitudes[:-1], output_rate, ax=ax1, display=False, save=False, **props)

		props.update({'xlabel': r'$\mathrm{ISI}$', 'ylabel': r'$\mathrm{ISI} [\mathrm{ms}]$',
							   'title': r'$AI = {0}$'.format(str(A2))})
		pr2 = props.copy()
		pr2.update({'inset': {'isi': isiis}})
		vz.plot_singleneuron_isis(spike_list.isi()[0], ax=ax2, display=False, save=False, **pr2)

		props.update({'xlabel': r'$\mathrm{ISI}_{n} [\mathrm{ms}]$', 'ylabel': r'$\mathrm{ISI}_{n+1} [\mathrm{ms}]$',
					  'title': r'$AI = {0}$'.format(str(A))})
		vz.recurrence_plot(isiis, ax=ax3, display=False, save=False, **props)

		vm_plot = vz.AnalogSignalPlots(single_vm, start=interval[0], stop=interval[0]+1000)
		props = {'xlabel': r'Time [ms]', 'ylabel': '$V_{m} [\mathrm{mV}]$'}
		if other_analogs is not None:
			for signal in single_analogs:
				ax4.plot(signal.time_axis(), signal.as_array()[0, :], 'g')
		if 'V_reset' in parameter_set.net_pars.neuron_pars[0].keys() and 'V_th' in parameter_set.net_pars.neuron_pars[0].keys():
			ax4 = vm_plot.plot_Vm(ax=ax4, with_spikes=True, v_reset=parameter_set.net_pars.neuron_pars[0]['V_reset'],
							 v_th=parameter_set.net_pars.neuron_pars[0]['V_th'], display=False, save=False, **props)
		else:
			if 'single_spk' in locals():
				spikes = single_spk.spiketrains[single_spk.id_list[0]].spike_times
				ax4.vlines(spikes, ymin=np.min(single_vm.raw_data()[:, 0]), ymax=np.max(single_vm.raw_data()[:, 0]))
			ax4 = vm_plot.plot_Vm(ax=ax4, with_spikes=False, v_reset=None,
							 v_th=None, display=False, save=False, **props)
		if display:
			pl.show()
		if save:
			assert isinstance(save, str), "Please provide filename"
			fig.savefig(save + population_object.name + '_SingleNeuron_DCresponse.pdf')

	return dict(input_amplitudes=input_amplitudes[:-1], input_times=input_times, output_rate=np.array(output_rate),
				   isi=spike_list.isi()[0], vm=vm_list.analog_signals[vm_list.analog_signals.keys()[0]].signal,
				time_axis=vm_list.analog_signals[vm_list.analog_signals.keys()[0]].time_axis(), AI=A)


def single_neuron_responses(population_object, parameter_set, pop_idx=0, start=None, stop=None, plot=True, display=True,
							save=False):
	"""
	Responses of a single neuron (population_object.populations[pop_idx] should be the single neuron)
	:return:
	"""
	results = dict(rate=0, isis=[0, 0], cv_isi=0, ff=None, vm=[], I_e=[], I_i=[], time_data=[])
	single_neuron_params = parameter_set.net_pars.neuron_pars[pop_idx]

	if parameter_set.net_pars.record_spikes[pop_idx]:
		spike_list = population_object.spiking_activity
		if start is not None and stop is not None:
			time_axis = np.arange(start, stop, 0.1)
			results['time_data'] = time_axis

		if list(spike_list.raw_data()):
			if start is not None and stop is not None:
				spike_list = spike_list.time_slice(start, stop)

			results['rate'] = spike_list.mean_rate()
			results['isi'] = spike_list.isi()
			results['cv_isi'] = spike_list.cv_isi(True)
			if results['rate']:
				results['ff'] = spike_list.fano_factor(1.)
		else:
			print("No spikes recorded")
	else:
		print("No spike recorder attached to {0}".format(population_object.name))

	if parameter_set.net_pars.record_analogs[pop_idx]:
		for idx, nn in enumerate(population_object.analog_activity_names): #parameter_set.net_pars.analog_device_pars[pop_idx]['record_from']):
			globals()[nn] = population_object.analog_activity[idx]

			if list(globals()[nn].raw_data()):
				if start is not None and stop is not None:
					globals()[nn] = globals()[nn].time_slice(start, stop)
				else:
					time_axis = globals()[nn].time_axis()
					results['time_data'] = time_axis[:-1]

				iddds = list(globals()[nn].analog_signals.iterkeys())
				if nn == 'V_m':
					results['vm'] = globals()[nn].analog_signals[int(min(iddds))].signal
				elif nn == 'I_ex':
					results['I_e'] = -globals()[nn].analog_signals[int(min(iddds))].signal
					# results['I_e'] /= 1000.
				elif nn == 'I_in':
					results['I_i'] = -globals()[nn].analog_signals[int(min(iddds))].signal #/ 1000.
				elif nn == 'g_in':
					E_in = parameter_set.net_pars.neuron_pars[pop_idx]['E_in']
					results['I_i'] = -globals()[nn].analog_signals[int(min(iddds))].signal * (results['vm'] - E_in)
					results['I_i'] /= 1000.
				elif nn == 'g_ex':
					E_ex = parameter_set.net_pars.neuron_pars[pop_idx]['E_ex']
					results['I_e'] = -globals()[nn].analog_signals[int(min(iddds))].signal * (results['vm'] - E_ex)
					results['I_e'] /= 1000.
				else:
					results[nn] = globals()[nn].analog_signals[int(min(iddds))].signal
				# TODO: add case when record g_ex/g_in
			else:
				print("No recorded analog {0}".format(str(nn)))
	else:
		print("No recorded analogs from {0}".format(population_object.name))
	if plot:
		fig = pl.figure()
		ax1 = pl.subplot2grid((10, 10), (0, 0), rowspan=4, colspan=4)
		ax2 = pl.subplot2grid((10, 10), (0, 5), rowspan=4, colspan=5)
		ax3 = pl.subplot2grid((10, 10), (5, 0), rowspan=2, colspan=10)
		ax4 = pl.subplot2grid((10, 10), (8, 0), rowspan=2, colspan=10, sharex=ax3)
		fig.suptitle(r'Population ${0}$ - Single Neuron Activity [${1}, {2}$]'.format(population_object.name,
																					  str(start),
																					str(stop)))
		props = {'xlabel': '', 'ylabel': '', 'xticks': [], 'yticks': [], 'yticklabels': '', 'xticklabels': ''}
		ax2.set(**props)
		if parameter_set.net_pars.record_spikes[pop_idx] and list(spike_list.raw_data()):
			ax2.text(0.5, 0.9, r'ACTIVITY REPORT', color='k', fontsize=16, va='center', ha='center')
			ax2.text(0.2, 0.6, r'- Firing Rate = ${0}$ spikes/s'.format(str(results['rate'])), color='k', fontsize=12,
					 va='center', ha='center')
			ax2.text(0.2, 0.4, r'- $CV_{0} = {1}$'.format('{ISI}', str(results['cv_isi'])), color='k', fontsize=12,
					 va='center', ha='center')
			ax2.text(0.2, 0.2, r'- Fano Factor = ${0}$'.format(str(results['ff'])), color='k', fontsize=12,
					 va='center', ha='center')

			props = {'xlabel': r'ISI', 'ylabel': r'Frequency', 'histtype': 'stepfilled', 'alpha': 1.}
			ax1.set_yscale('log')
			vz.plot_histogram(results['isi'], nbins=10, norm=True, mark_mean=True, ax=ax1, color='b',
			                       display=False,
							   save=False, **props)
			spikes = spike_list.spiketrains[spike_list.id_list[0]].spike_times

		if parameter_set.net_pars.record_analogs[pop_idx]:
			props2 = {'xlabel': r'Time [ms]', 'ylabel': r'$V_{m} [mV]$'}
			ap = vz.AnalogSignalPlots(globals()['V_m'], start, stop)
			if 'V_reset' in single_neuron_params.keys() and 'V_th' in single_neuron_params.keys():
				ax4 = ap.plot_Vm(ax=ax4, with_spikes=True, v_reset=single_neuron_params['V_reset'],
								v_th=single_neuron_params['V_th'], display=False, save=False, **props2)
			else:
				if 'spikes' in locals():
					ax4.vlines(spikes, ymin=np.min(globals()['V_m'].raw_data()[:, 0]), ymax=np.max(globals()[
																									   'V_m'].raw_data()[:, 0]))
				ax4 = ap.plot_Vm(ax=ax4, with_spikes=False, v_reset=None,
								 v_th=None, display=False, save=False, **props2)

			ax4.set_xticks(np.linspace(start, stop, 5))
			ax4.set_xticklabels([str(x) for x in np.linspace(start, stop, 5)])
			if results.has_key('I_e') and not sg.empty(results['I_e']):
				props = {'xlabel': '', 'xticklabels': '', 'ylabel': r'$I_{\mathrm{syn}} [nA]$'}
				ax3.set(**props)
				ax3.plot(results['time_data'], -results['I_e']/1000, 'b', lw=1)
				ax3.plot(results['time_data'], -results['I_i']/1000, 'r', lw=1)
				ax3.plot(results['time_data'], (-results['I_e']-results['I_i'])/1000, 'gray', lw=1)
			else:
				keys = [n for n in parameter_set.net_pars.analog_device_pars[pop_idx]['record_from'] if n != 'V_m']
				for k in keys:
					ax3.plot(results['time_data'], results[k], label=r'$'+k+'$')
				ax3.legend()
		if display:
			pl.show()
		if save:
			assert isinstance(save, str), "Please provide filename"
			if isinstance(fig, mpl.figure.Figure):
				fig.savefig(save + population_object.name + '_SingleNeuron.pdf')
	return results


def ssa_lifetime(pop_obj, parameter_set, input_off=1000., display=True):
	"""

	:param spike_list:
	:return:
	"""
	results = dict(ssa={})
	if display:
		print("\nSelf-sustaining Activity Lifetime: ")
	if isinstance(pop_obj, na.Network):
		gids = []
		new_SpkList = sg.SpikeList([], [], parameter_set.kernel_pars.transient_t,
		                           parameter_set.kernel_pars.sim_time + \
		                           parameter_set.kernel_pars.transient_t,
		                           np.sum(list(sg.iterate_obj_list(
								  pop_obj.n_neurons))))
		for ii, n in enumerate(pop_obj.spiking_activity):
			gids.append(n.id_list)
			for idd in n.id_list:
				new_SpkList.append(idd, n.spiketrains[idd])
			results['ssa'].update({str(pop_obj.population_names[ii]+'_ssa'): {'last_spike': n.last_spike_time(),
																			  'tau': n.last_spike_time() -
																					 input_off}})
			if display:
				print("- {0} Survival = {1} ms".format(str(pop_obj.population_names[ii]), str(results['ssa'][str(
					pop_obj.population_names[ii]+'_ssa')]['tau'])))

		results['ssa'].update({'Global_ssa': {'last_spike': new_SpkList.last_spike_time(),
										  'tau': new_SpkList.last_spike_time() - input_off}})
		if display:
			print("- {0} Survival = {1} ms".format('Global', str(results['ssa']['Global_ssa']['tau'])))

	elif isinstance(pop_obj, na.Population):
		name = pop_obj.name
		spike_list = pop_obj.spiking_activity.spiking_activity
		results['ssa'].update({name+'_ssa': {'last_spike': spike_list.last_spike_time(),
						 'tau': spike_list.last_spike_time() - input_off}})
		if display:
			print("- {0} Survival = {1} ms".format(str(name), str(results['ssa'][name+'_ssa']['tau'])))
	else:
		raise ValueError("Input must be Network or Population object")

	return results


def fmf_readout(response, target, readout, index, label='', plot=False, display=False, save=False):
	"""
	(to be removed)
	This function is specific to the fading memory estimation, and performs all the necessary computations to compute
	the fading memory function
	:return:
	"""
	label += str(round(np.median(response), 1))
	state = response[:, index:]
	target = target[:, :-index]
	readout.train(state, target)
	norm_wout = readout.measure_stability()
	print("|W_out| [{0}] = {1}".format(readout.name, str(norm_wout)))

	output = readout.test(state)

	if output.shape == target.shape:
		MAE = np.mean(output - target)
		MSE = mse(output, target)
		RMSE = rmse(output, target)
		NMSE = nmse(output, target)
		NRMSE = nrmse(output[0], target[0])

		print("\t- MAE = {0}".format(str(MAE)))
		print("\t- MSE = {0}".format(str(MSE)))
		print("\t- NMSE = {0}".format(str(NMSE)))
		print("\t- RMSE = {0}".format(str(RMSE)))
		print("\t- NRMSE = {0}".format(str(NRMSE)))

		COV = (np.cov(target, output) ** 2.)
		VARS = np.var(output) * np.var(target)
		FMF = COV / VARS
		fmf = FMF[0, 1]
		print("M[k] = {0}".format(str(FMF[0, 1])))
	else:
		MAE = np.mean(output.T - target)
		MSE = mse(output.T, target)
		RMSE = rmse(output.T, target)
		NMSE = nmse(output.T, target)
		NRMSE = nrmse(output[:, 0], target[0])

		print("\t- MAE = {0}".format(str(MAE)))
		print("\t- MSE = {0}".format(str(MSE)))
		print("\t- NMSE = {0}".format(str(NMSE)))
		print("\t- RMSE = {0}".format(str(RMSE)))
		print("\t- NRMSE = {0}".format(str(NRMSE)))

		COV = np.cov(target[0, :], output[:, 0]) ** 2.
		VARS = np.var(target) * np.var(output)
		FMF = COV / VARS
		fmf = FMF[0, 1]
		print("\t- M[k] = {0}".format(str(FMF[0, 1])))

	if plot:
		vz.plot_target_out(target, output, label, display, save)

	return output, {'MAE': MAE, 'MSE': MSE, 'NMSE': NMSE, 'RMSE': RMSE,
	                'NRMSE': NRMSE, 'norm_wOut': norm_wout, 'fmf': fmf}


# TODO update
def evaluate_fading_memory(net, parameter_set, input, total_time, normalize=True,
						   debug=False, plot=True, display=True, save=False):
	"""

	:param net:
	:param parameter_set:
	:param input:
	:param total_time:
	:param normalize:
	:param debug:
	:param plot:
	:param display:
	:param save:
	:return:
	"""
	results = {}
	#######################################################################################
	# Train Readouts
	# =====================================================================================
	# Set targets
	cut_off_time = parameter_set.kernel_pars.transient_t
	t_axis = np.arange(cut_off_time, total_time, parameter_set.input_pars.noise.resolution)
	global_target = input.noise_signal.time_slice(t_start=cut_off_time, t_stop=total_time).as_array()

	# Set baseline random output (for comparison)
	input_noise_r2 = ia.InputNoise(parameter_set.input_pars.noise,
	                               stop_time=total_time)
	input_noise_r2.generate()
	input.re_seed(parameter_set.kernel_pars.np_seed)

	baseline_out = input_noise_r2.noise_signal.time_slice(t_start=cut_off_time,
														  t_stop=total_time).as_array()

	if normalize:
		global_target /= parameter_set.input_pars.noise.noise_pars.amplitude
		global_target -= np.mean(global_target)  # parameter_set.input_pars.noise.noise_pars.mean
		baseline_out /= parameter_set.input_pars.noise.noise_pars.amplitude
		baseline_out -= np.mean(baseline_out)  # parameter_set.input_pars.noise.noise_pars.mean

	print("\n*******************************\nFading Memory Evaluation\n*******************************\nBaseline (" \
		  "random): ")

	# Error
	MAE = np.mean(np.abs(baseline_out[0] - global_target[0]))
	MSE = mse(baseline_out, global_target)
	RMSE = rmse(baseline_out, global_target)
	NMSE = nmse(baseline_out, global_target)
	NRMSE = nrmse(baseline_out[0], global_target[0])

	print("\t- MAE = {0}".format(str(MAE)))
	print("\t- MSE = {0}".format(str(MSE)))
	print("\t- NMSE = {0}".format(str(NMSE)))
	print("\t- RMSE = {0}".format(str(RMSE)))
	print("\t- NRMSE = {0}".format(str(NRMSE)))
	# memory
	COV = (np.cov(global_target, baseline_out) ** 2.)
	VARS = np.var(baseline_out) * np.var(global_target)
	FMF = COV / VARS
	print("\t- M[0] = {0}".format(str(FMF[0, 1])))
	results['Baseline'] = {'MAE': MAE,
						   'MSE': MSE,
						   'NMSE': NMSE,
						   'RMSE': RMSE,
						   'NRMSE': NRMSE,
						   'M[0]': FMF[0, 1]}

	#################################
	# Train Readouts
	#################################
	read_pops = []

	if not sg.empty(net.merged_populations):
		for n_pop in net.merged_populations:
			if save:
				save_path = save + n_pop.name
			else:
				save_path = False

			results['{0}'.format(n_pop.name)] = {}
			if hasattr(n_pop, "decoding_pars"):
				print("\nPopulation {0}".format(n_pop.name))
				read_pops.append(n_pop)
				internal_indices = [int(readout.name[len(readout.name.rstrip('0123456789')):])+1 for readout in
									n_pop.readouts]

				for index, readout in enumerate(n_pop.readouts):
					internal_idx = internal_indices[index]
					if len(n_pop.response_matrix) == 1:
						response_matrix = n_pop.response_matrix[0].as_array()
						if internal_idx == 1:
							output, results_1 = fmf_readout(response_matrix, global_target, readout, internal_idx,
															label=n_pop.name, plot=plot, display=display,
															save=save_path)
						else:
							output, results_1 = fmf_readout(response_matrix, global_target, readout, internal_idx,
															label=n_pop.name, plot=False, display=False, save=False)

						results['{0}'.format(n_pop.name)].update(
								{'Readout_{1}'.format(n_pop.name, str(index)): results_1})

					else:
						for resp_idx, n_response in enumerate(n_pop.response_matrix):
							response_matrix = n_response.as_array()
							if internal_idx == 1:
								output, results_1 = fmf_readout(response_matrix, global_target, readout, internal_idx,
																label=n_pop.name, plot=plot, display=display,
																save=save_path)
							else:
								output, results_1 = fmf_readout(response_matrix, global_target, readout, internal_idx,
																label=n_pop.name, plot=plot, display=display,
																save=save_path)

							results['{0}'.format(n_pop.name)].update(
									{'Readout_{0}_R{1}'.format(str(resp_idx), str(index)): results_1})
	if not sg.empty(net.state_extractors):
		for n_pop in net.populations:
			if save:
				save_path = save + n_pop.name
			else:
				save_path = False

			results['{0}'.format(n_pop.name)] = {}
			if hasattr(n_pop, "decoding_pars"):
				print("\nPopulation {0}".format(n_pop.name))
				read_pops.append(n_pop)
				internal_indices = [int(readout.name[len(readout.name.rstrip('0123456789')):])+1 for readout in
									n_pop.readouts]

				if len(n_pop.response_matrix) == 1:
					for index, readout in enumerate(n_pop.readouts):
						internal_idx = internal_indices[index]
						response_matrix = n_pop.response_matrix[0].as_array()

						if internal_idx == 1:
							output, results_1 = fmf_readout(response_matrix, global_target, readout, internal_idx,
															label=n_pop.name, plot=True,
															display=display, save=save_path)
						else:
							output, results_1 = fmf_readout(response_matrix, global_target, readout, internal_idx,
															label=n_pop.name, plot=False, display=False, save=False)

						results['{0}'.format(n_pop.name)].update(
								{'Readout_{1}'.format(n_pop.name, str(index)): results_1})

				else:
					for resp_idx, n_response in enumerate(n_pop.response_matrix):
						partition_idx = len(n_pop.readouts) / len(n_pop.response_matrix)
						readout_set = n_pop.readouts[resp_idx * partition_idx:(resp_idx+1)*partition_idx]
						internal_idxx = [int(n.name.strip('mem'))+1 for n in readout_set]

						for index, readout in enumerate(readout_set):
							internal_idx = internal_idxx[index]
							response_matrix = n_response.as_array()

							if internal_idx == 1:
								output, results_1 = fmf_readout(response_matrix, global_target, readout, internal_idx,
																label=n_pop.name, plot=plot, display=display,
																save=save_path)
							else:
								output, results_1 = fmf_readout(response_matrix, global_target, readout, internal_idx,
																label=n_pop.name, plot=False, display=False, save=False)

							results['{0}'.format(n_pop.name)].update(
									{'Readout_{0}_R{1}'.format(str(resp_idx), str(index)): results_1})

	for pop in read_pops:
		dx = np.min(np.diff(t_axis))
		if plot:
			globals()['fig_{0}'.format(pop.name)] = plt.figure()
			globals()['fig_{0}1'.format(pop.name)] = plt.figure()

		if len(pop.response_matrix) == 1:
			fmf = [results[pop.name][x]['fmf'] for idx, x in enumerate(np.sort(results[pop.name].keys()))]
			initial_guess = 1., 1., 10.
			steps = np.arange(0., len(fmf)*dx, dx)
			fit_params, _ = opt.leastsq(err_func, initial_guess, args=(steps, fmf, acc_function))
			error = np.sum((fmf - acc_function(steps, *fit_params)) ** 2)
			MC_trap = np.trapz(fmf, dx=dx)
			MC_simp = integ.simps(fmf, dx=dx)
			MC_trad = np.sum(fmf[1:])
			results[pop.name]['MC'] = {'MC_trap': MC_trap, 'MC_simp': MC_simp, 'MC_trad': MC_trad}

			if plot:
				ax_1 = globals()['fig_{0}'.format(pop.name)].add_subplot(111)
				ax_2 = globals()['fig_{0}1'.format(pop.name)].add_subplot(111)
				vz.plot_fmf(t_axis, fmf, ax_1, label=pop.name, display=display, save=save_path)
				vz.plot_acc(steps, np.array(fmf), fit_params, acc_function, title=r'Fading Memory Fit',
						 ax=ax_2, display=display, save=str(save_path) + 'fmf')
		else:
			ax_ctr = 0
			remove_keys = []
			for resp_idx, n_response in enumerate(pop.response_matrix):
				ax_ctr += 1
				remove_keys.append('MC'+str(resp_idx))
				# split readout results:
				readout_labels = [x for x in results[pop.name].keys() if x not in remove_keys and int(x[8]) == resp_idx]
				sorted_labels = [int(n[n.index('_R')+2:]) for n in readout_labels]
				sorted_indices = np.argsort(sorted_labels)
				sorted_labels = [readout_labels[n] for n in sorted_indices]
				readout_set_results = [results[pop.name][x] for x in sorted_labels]
				fmf = [x['fmf'] for x in readout_set_results]
				initial_guess = 1., 1., 10.
				steps = np.arange(0., len(fmf) * dx, dx)
				fit_params, _ = opt.leastsq(err_func, initial_guess, args=(steps, fmf, acc_function))
				error = np.sum((fmf - acc_function(steps, *fit_params)) ** 2)

				MC_trap = np.trapz(fmf, dx=dx)
				MC_simp = integ.simps(fmf, dx=dx)
				MC_trad = np.sum(fmf[1:])
				results[pop.name]['MC'+str(resp_idx)] = {'MC_trap': MC_trap, 'MC_simp': MC_simp, 'MC_trad': MC_trad*dx}

				if plot:
					globals()['ax1_{0}'.format(resp_idx)] = globals()['fig_{0}'.format(pop.name)].add_subplot(1,
									len(pop.response_matrix), ax_ctr)
					globals()['ax1_{0}1'.format(resp_idx)] = globals()['fig_{0}1'.format(pop.name)].add_subplot(1,
									len(pop.response_matrix), ax_ctr)

					if save:
						save_pth = save_path + str(resp_idx)
					else:
						save_pth = False

					plot_fmf(t_axis, fmf, globals()['ax1_{0}'.format(resp_idx)],
							 label=pop.name + 'State_{0}'.format(str(resp_idx)), display=display, save=save_pth)
					plot_acc(steps, np.array([fmf]), fit_params, acc_function, title=r'Fading Memory Fit',
							 ax=globals()['ax1_{0}1'.format(resp_idx)], display=display, save=save_path)

		return results


# TODO is this not redundant with analyse_activity_dynamics? - should be combined, but now they are doing different
# things..
def analyse_state_matrix(state, stim_labels, label='', plot=True, display=True, save=False):
	"""

	:param state:
	:param stim:
	:return:
	"""
	pca_obj = sk.PCA(n_components=3)
	X_r = pca_obj.fit(state.T).transform(state.T)
	print("Explained Variance (first 3 components): %s" % str(pca_obj.explained_variance_ratio_))

	if not isinstance(stim_labels, dict):
		label_seq = np.array(list(sg.iterate_obj_list(stim_labels)))
		n_elements = np.unique(label_seq)
		if plot:
			fig1 = pl.figure()
			ax1 = fig1.add_subplot(111)
			vz.plot_state_matrix(state, stim_labels, ax=ax1, label=label, display=False, save=False)

			fig2 = pl.figure()
			fig2.clf()
			exp_var = [round(n, 2) for n in pca_obj.explained_variance_ratio_]
			fig2.suptitle(r'${0} - PCA (var = {1})$'.format(str(label), str(exp_var)),
						  fontsize=20)

			ax2 = fig2.add_subplot(111, projection='3d')
			colors_map = vz.get_cmap(N=len(n_elements), cmap='Paired')
			ax2.set_xlabel(r'$PC_{1}$')
			ax2.set_ylabel(r'$PC_{2}$')
			ax2.set_zlabel(r'$PC_{3}$')

			ccs = [colors_map(ii) for ii in range(len(n_elements))]
			for color, index, lab in zip(ccs, n_elements, n_elements):
				locals()['sc_{0}'.format(str(index))] = ax2.scatter(X_r[np.where(np.array(list(itertools.chain(
					label_seq))) == index)[0], 0], X_r[np.where(np.array(list(itertools.chain(label_seq))) == index)[
								0],  1], X_r[np.where(np.array(list(itertools.chain(label_seq))) == index)[0], 2],
																	s=50, c=color, label=lab)
			scatters = [locals()['sc_{0}'.format(str(ind))] for ind in n_elements]
			#pl.legend(tuple(scatters), tuple(n_elements))
			pl.legend(loc=0, handles=scatters)

			if display:
				pl.show(block=False)
			if save:
				fig1.savefig(save + 'state_matrix_{0}.pdf'.format(label))
				fig2.savefig(save + 'pca_representation_{0}.pdf'.format(label))
	else:
		if plot:
			fig1 = pl.figure()
			ax = fig1.add_subplot(111, projection='3d')
			ax.plot(X_r[:, 0], X_r[:, 1], X_r[:, 2], color='r', lw=2)
			ax.set_title(label + r'$ - (3PCs) $= {0}$'.format(str(round(np.sum(pca_obj.explained_variance_ratio_[:3]),
																	1))))
			ax.grid()
			if display:
				pl.show(False)
			if save:
				fig1.savefig(save + 'pca_representation_{0}.pdf'.format(label))


def advanced_state_analysis(state, stim_labels=None, label='', plot=True, display=True, save=False):
	"""
	"""
	pass


# TODO is this still needed? there's the same function in DecodingLayer - that is evaluate_decoding - yes,
# this is evaluate_encoding ;)
def evaluate_encoding(enc_layer, parameter_set, analysis_interval, input_signal, plot=True, display=True, save=False):
	"""
	Determine the quality of the encoding method (if there are encoders), by reading out the state of the encoders
	and training it to reconstruct the input signal
	:param enc_layer:
	:return:
	"""
	assert(isinstance(analysis_interval, list)), "Incorrect analysis_interval"
	results = dict()
	for idx, n_enc in enumerate(enc_layer.encoders):
		new_pars = pa.ParameterSet(pa.copy_dict(parameter_set.as_dict()))
		new_pars.kernel_pars.data_prefix = 'Input Encoder {0}'.format(n_enc.name)
		# results['input_activity_{0}'.format(str(idx))] = characterize_population_activity(n_enc,
		#                                                                   parameter_set=new_pars,
		#                                                                   analysis_interval=analysis_interval,
		#                                                                   epochs=None, time_bin=1., complete=False,
		#                                                                   time_resolved=False, color_map='colorwarm',
		#                                                                   plot=plot, display=display, save=save)

		if isinstance(n_enc.spiking_activity, sg.SpikeList) and not n_enc.spiking_activity.empty():
			inp_spikes = n_enc.spiking_activity.time_slice(analysis_interval[0], analysis_interval[1])
			tau = parameter_set.decoding_pars.state_extractor.filter_tau
			n_input_neurons = np.sum(parameter_set.encoding_pars.encoder.n_neurons)
			if n_enc.decoding_layer is not None:
				inp_responses = n_enc.decoding_layer.extract_activity(start=analysis_interval[0],
				                                                      stop=analysis_interval[1], save=False,
				                                                      reset=False)[0]
				inp_readout_pars = pa.copy_dict(n_enc.decoding_layer.decoding_pars.readout[0])
			else:
				# TEST!! TODO
				inp_responses = inp_spikes.filter_spiketrains(dt=input_signal.dt,
				                                              tau=tau, start=analysis_interval[0],
				                                              stop=analysis_interval[1], N=n_input_neurons)
				inp_readout_pars = pa.copy_dict(parameter_set.decoding_pars.readout[0],
				                                {'label': 'InputNeurons',
			                                  'algorithm': parameter_set.decoding_pars.readout[0]['algorithm'][0]})


			inp_readout = Readout(pa.ParameterSet(inp_readout_pars))
			analysis_signal = input_signal.time_slice(analysis_interval[0], analysis_interval[1])
			inp_readout.train(inp_responses, analysis_signal.as_array())
			inp_readout.test(inp_responses)
			perf = inp_readout.measure_performance(analysis_signal.as_array())

			input_out = ia.InputSignal()
			input_out.load_signal(inp_readout.output.T, dt=input_signal.dt, onset=analysis_interval[0],
				inherit_from=analysis_signal)

			if plot:
				figure2 = pl.figure()
				figure2.suptitle(r'MAE = {0}'.format(str(perf['raw']['MAE'])))
				ax21 = figure2.add_subplot(211)
				ax22 = figure2.add_subplot(212, sharex=ax21)
				vz.InputPlots(input_obj=analysis_signal).plot_input_signal(ax22, save=False, display=False)
				ax22.set_color_cycle(None)
				vz.InputPlots(input_obj=input_out).plot_input_signal(ax22, save=False, display=False)
				ax22.set_ylim([analysis_signal.base-10., analysis_signal.peak+10.])
				inp_spikes.raster_plot(with_rate=False, ax=ax21, save=False, display=False)
				if display:
					pl.show(block=False)
				if save:
					figure2.savefig(save+'_EncodingQuality.pdf')
	return results


# def analyse_performance_results(net, enc_layer=None, plot=True, display=True, save=False):
# 	"""
# 	Re-organizes performance results
# 	(may be too case-sensitive!!)
# 	"""
# 	# TODO - deprecated...
# 	from modules.signals import empty
# 	results = {}
#
# 	if enc_layer is not None:
# 		all_populations = list(itertools.chain(*[net.merged_populations, net.populations, enc_layer.encoders]))
# 	else:
# 		all_populations = list(itertools.chain(*[net.merged_populations, net.populations]))
#
# 	for n_pop in all_populations:
# 		if hasattr(n_pop, "decoding_pars"):
# 			results[n_pop.name] = {}
# 			readout_labels = list(np.sort(n_pop.decoding_pars['readout']['labels']))
# 			readout_type = [n[:3] for n in readout_labels]
# 			if 'mem' in readout_type and 'cla' in readout_type: # special case
# 				last_mem = readout_type[::-1].index('mem')
# 				readout_labels.insert(last_mem + 1, readout_labels[0])
# 				readout_labels.pop(0)
# 				readout_type = [n[:3] for n in readout_labels]
# 				last_mem = readout_type[::-1].index('mem')
# 				first_mem = readout_type.index('mem')
# 				readout_labels[first_mem:last_mem - 1] = readout_labels[first_mem:last_mem - 1][::-1]
#
# 			pop_readouts = n_pop.readouts
# 			pop_state_variables = n_pop.state_variables
# 			print(pop_state_variables)
# 			if empty(n_pop.state_sample_times):
# 				for idx_state, n_state in enumerate(n_pop.state_extractors):
# 					pop_readout_labels = [n.name for n in pop_readouts[idx_state]]
# 					readout_idx = [np.where(n == np.array(readout_labels))[0][0] for n in pop_readout_labels]
# 					readout_set = list(np.array(pop_readouts[idx_state])[readout_idx])
# 					results[n_pop.name].update({'ReadoutSet{0}'.format(str(idx_state)): {}})
# 					indices = [n.index for n in readout_set]
# 					results[n_pop.name]['ReadoutSet{0}'.format(str(idx_state))] = compile_performance_results(
# 						readout_set, state_variable=pop_state_variables[idx_state])
# 			else:
# 				assert (len(pop_readouts) == len(n_pop.state_sample_times)), "Inconsistent readout set"
# 				n_states = len(pop_readouts[0])
# 				for n_state in range(n_states):
# 					results[n_pop.name].update({'ReadoutSet{0}'.format(str(n_state)): {}})
# 					results[n_pop.name]['ReadoutSet{0}'.format(str(n_state))].update(
# 						{'sample_times': n_pop.state_sample_times})
#
# 					for n_sample_time in range(len(n_pop.state_sample_times)):
# 						readout_set = pop_readouts[n_sample_time][n_state]
# 						results[n_pop.name]['ReadoutSet{0}'.format(str(n_state))].update({'sample_{0}'.format(
# 							n_sample_time): {}})
# 						results[n_pop.name]['ReadoutSet{0}'.format(str(n_state))]['sample_{0}'.format(
# 							n_sample_time)] = compile_performance_results(readout_set,
# 																		  state_variable=pop_state_variables[n_state])
#
# 	if plot:
# 		vz.plot_readout_performance(results, display=display, save=save)
# 	return results


def compile_performance_results(readout_set, state_variable=''):
	"""
	"""
	# TODO - deprecated
	results = {
		'accuracy': np.array([n.performance['label']['performance'] for n in readout_set]),
		'hamming_loss': np.array([n.performance['label']['hamm_loss'] for n in readout_set]),
		'MSE': np.array([n.performance['max']['MSE'] for n in readout_set if not sg.empty(n.performance['max'])]),
		'raw_MAE': np.array([n.performance['raw']['MAE'] for n in readout_set if not sg.empty(n.performance['raw'])]),
		'precision': np.array([n.performance['label']['precision'] for n in readout_set]),
		'f1_score': np.array([n.performance['label']['f1_score'] for n in readout_set]),
		'recall': np.array([n.performance['label']['recall'] for n in readout_set]),
		'confusion_matrices': [n.performance['label']['confusion'] for n in readout_set],
		'jaccard': np.array([n.performance['label']['jaccard'] for n in readout_set]),
		'class_support': [n.performance['label']['class_support'] for n in readout_set],
		'norm_wout': np.array([n.measure_stability() for n in readout_set]),
		'labels': [n.name for n in readout_set],
		'indices': [n.index for n in readout_set],
		'state_variable': state_variable}
	return results


def analyse_state_divergence(parameter_set, net, clone, plot=True, display=True, save=False):
	"""
	Analyse how responses from net and clone diverge (primarily for perturbation analysis)
	:param parameter_set:
	:param net: Network object
	:param clone: Network object
	:return:
	"""
	results = dict()
	pop_idx = net.population_names.index(parameter_set.kernel_pars.perturb_population)
	start = parameter_set.kernel_pars.transient_t
	stop = parameter_set.kernel_pars.sim_time + parameter_set.kernel_pars.transient_t
	activity_time_vector = np.arange(parameter_set.kernel_pars.transient_t,
									 parameter_set.kernel_pars.sim_time + parameter_set.kernel_pars.transient_t,
									 parameter_set.kernel_pars.resolution)
	perturbation_time = parameter_set.kernel_pars.perturbation_time + parameter_set.kernel_pars.transient_t
	observation_time = max(activity_time_vector) - perturbation_time

	if not sg.empty(net.populations[pop_idx].spiking_activity.spiketrains):
		time_vec = net.populations[pop_idx].spiking_activity.time_axis(1)[:-1]
		perturbation_idx = np.where(time_vec == perturbation_time)
		rate_native = net.populations[pop_idx].spiking_activity.firing_rate(1, average=True)
		rate_clone = clone.populations[pop_idx].spiking_activity.firing_rate(1, average=True)

		binary_native = net.populations[pop_idx].spiking_activity.compile_binary_response_matrix(
				parameter_set.kernel_pars.resolution, start=parameter_set.kernel_pars.transient_t,
				stop=parameter_set.kernel_pars.sim_time+parameter_set.kernel_pars.transient_t,
				N=net.populations[pop_idx].size, display=True)
		binary_clone = clone.populations[pop_idx].spiking_activity.compile_binary_response_matrix(
				parameter_set.kernel_pars.resolution, start=parameter_set.kernel_pars.transient_t,
				stop=parameter_set.kernel_pars.sim_time+parameter_set.kernel_pars.transient_t,
				N=clone.populations[pop_idx].size, display=True)

		r_cor = []
		hamming = []
		for idx, t in enumerate(time_vec):
			if not sg.empty(np.corrcoef(rate_native[:idx], rate_clone[:idx])) and np.corrcoef(rate_native[:idx],
								rate_clone[:idx])[0, 1] != np.nan:
				r_cor.append(np.corrcoef(rate_native[:idx], rate_clone[:idx])[0, 1])
			else:
				r_cor.append(0.)
			binary_state_diff = binary_native[:, idx] - binary_clone[:, idx]
			if not sg.empty(np.nonzero(binary_state_diff)[0]):
				hamming.append(float(len(np.nonzero(binary_state_diff)[0]))/float(net.populations[pop_idx].size))
			else:
				hamming.append(0.)

		results['rate_native'] = rate_native
		results['rate_clone'] = rate_clone
		results['rate_correlation'] = np.array(r_cor)
		results['hamming_distance'] = np.array(hamming)
		results['lyapunov_exponent'] = {}
	if not sg.empty(net.populations[pop_idx].decoding_layer.activity):
		responses_native = net.populations[pop_idx].decoding_layer.activity
		responses_clone = clone.populations[pop_idx].decoding_layer.activity
		response_vars = parameter_set.decoding_pars.state_extractor.state_variable
		print("\n Computing state divergence: ")
		labels = []
		for resp_idx, n_response in enumerate(responses_native):
			print("\t- State variable {0}".format(str(response_vars[resp_idx])))
			response_length = len(n_response.time_axis())
			distan = []
			for t in range(response_length):
				distan.append(distance.euclidean(n_response.as_array()[:, t], responses_clone[resp_idx].as_array()[:, t]))
				if display:
					vz.progress_bar(float(t)/float(response_length))

			results['state_{0}'.format(str(response_vars[resp_idx]))] = np.array(distan)
			labels.append(str(response_vars[resp_idx]))

			if np.array(distan).any():
				initial_distance = distan[min(np.where(np.array(distan) > 0.0)[0])]
			else:
				initial_distance = 0.
			final_distance = distan[-1]
			lyapunov = (np.log(final_distance) / observation_time) - np.log(initial_distance) / observation_time
			print("Lyapunov Exponent = {0}".format(lyapunov))
			results['lyapunov_exponent'].update({response_vars[resp_idx]: lyapunov})

	if plot:
		if not sg.empty(net.populations[pop_idx].spiking_activity.spiketrains):
			fig = pl.figure()
			fig.suptitle(r'$LE = {0}$'.format(str(results['lyapunov_exponent'].items())))
			ax1a = pl.subplot2grid((12, 1), (0, 0), rowspan=8, colspan=1)
			ax1b = ax1a.twinx()

			ax2a = pl.subplot2grid((12, 1), (8, 0), rowspan=2, colspan=1, sharex=ax1a)
			ax2b = ax2a.twinx()

			ax3 = pl.subplot2grid((12, 1), (10, 0), rowspan=2, colspan=1, sharex=ax1a)

			#ax4 = pl.subplot2grid((12, 1), (16, 0), rowspan=4, colspan=1)

			rp1 = vz.SpikePlots(net.populations[pop_idx].spiking_activity, start, stop)
			rp2 = vz.SpikePlots(clone.populations[pop_idx].spiking_activity, start, stop)

			plot_props1 = {'xlabel': 'Time [ms]', 'ylabel': 'Neuron', 'color': 'r', 'linewidth': 1.0,
						  'linestyle': '-'}
			plot_props2 = {'xlabel': 'Time [ms]', 'ylabel': 'Neuron', 'color': 'k', 'linewidth': 1.0,
						  'linestyle': '-'}
			rp1.dot_display(ax=[ax1a, ax2a], with_rate=True, default_color='r', display=False, save=False, **plot_props1)
			rp2.dot_display(ax=[ax1b, ax2b], with_rate=True, default_color='k', display=False, save=False, **plot_props2)

			ax3.plot(time_vec, r_cor, '-b')
			ax3.set_ylabel('CC')
			# mark perturbation time
			yrange_1 = np.arange(ax1a.get_ylim()[0], ax1a.get_ylim()[1], 1)
			ax1a.plot(perturbation_time * np.ones_like(yrange_1), yrange_1, 'r--')
			yrange_2 = np.arange(ax2a.get_ylim()[0], ax2a.get_ylim()[1], 1)
			ax2a.plot(perturbation_time * np.ones_like(yrange_2), yrange_2, 'r--')
			yrange_3 = np.arange(ax3.get_ylim()[0], ax3.get_ylim()[1], 1)
			ax3.plot(perturbation_time * np.ones_like(yrange_3), yrange_3, 'r--')
			if display:
				pl.show(False)
			if save:
				assert isinstance(save, str), "Please provide filename"
				fig.savefig(save + 'LE_analysis.pdf')

		if not sg.empty(net.populations[pop_idx].decoding_layer.activity):
			fig2 = pl.figure()
			ax4 = fig2.add_subplot(211)
			ax5 = fig2.add_subplot(212, sharex=ax4)
			for lab in labels:
				ax4.plot(activity_time_vector, results['state_{0}'.format(lab)][:len(activity_time_vector)], label=lab)
			ax4.set_ylabel(r'$d_{E}$')

			if 'hamming_distance' in results.keys():
				ax5.plot(time_vec, results['hamming_distance'], c='g')
				ax5.set_ylabel(r'$d_{H}$')

			yrange_4 = np.arange(ax4.get_ylim()[0], ax4.get_ylim()[1], 1)
			ax4.plot(perturbation_time * np.ones_like(yrange_4), yrange_4, 'r--')
			yrange_5 = np.arange(ax5.get_ylim()[0], ax5.get_ylim()[1], 1)
			ax5.plot(perturbation_time * np.ones_like(yrange_5), yrange_5, 'r--')

			ax4.set_xlabel(r'')
			#ax4.set_xticklabels([])
			ax4.set_xlim(np.min(activity_time_vector), np.max(activity_time_vector))
			ax4.legend(loc=0)
			ax5.set_xlabel(r'Time [ms]')

			if display:
				pl.show(False)
			if save:
				assert isinstance(save, str), "Please provide filename"
				fig2.savefig(save + 'state_divergence.pdf')
	return results


def calculate_ranks(network):
	"""
	Calculate the rank of all state matrices stored in the population's decoding_layers
	:return dict: {population_name: {state_variable: rank}}
	"""
	results = dict()
	for ctr, n_pop in enumerate(list(itertools.chain(*[network.merged_populations, network.populations]))):
		results[n_pop.name] = []
		states = []

		if n_pop.decoding_layer is not None:
			if not sg.empty(n_pop.decoding_layer.state_matrix) and isinstance(n_pop.decoding_layer.state_matrix[0],
			                                                                  list):
				states = list(itertools.chain(*n_pop.decoding_layer.state_matrix))
			elif not sg.empty(n_pop.decoding_layer.state_matrix):
				states = n_pop.decoding_layer.state_matrix

		for idx_state, n_state in enumerate(states):
			results[n_pop.name].update({n_pop.decoding_layer.state_variables[idx_state]: get_state_rank(n_state)})

	return results


def get_state_rank(matrix):
	"""
	Calculate the rank of all state matrices
	:return:
	"""
	return np.linalg.matrix_rank(matrix)


########################################################################################################################
class Readout(object):
	"""
	Readout object, trained to produce an estimation y(t) of output by reading out population state variables.
	"""
	def __init__(self, initializer, display=True):
		"""
		Create and initialize Readout object

		:param initializer: ParameterSet object or dictionary specifying Readout parameters
		"""
		self.name = initializer.label
		self.rule = initializer.algorithm
		self.weights = None
		self.fit_obj = None
		self.output = None
		self.index = None
		self.norm_wout = None
		self.performance = {}
		if display:
			print(("\t- Readout {0} [trained with {1}]".format(self.name, self.rule)))

	def set_index(self):
		"""
		For a specific case, in which the readout name contains a time index
		"""
		index = int(''.join(c for c in self.name if c.isdigit()))
		if self.name[:3] == 'mem':
			self.index = -index
		else:
			self.index = index

	def train(self, state_train, target_train, index=None, accepted=None, display=True):
		"""
		Train readout
		:param state_train: np.ndarray state matrix
		:param target_train: np.ndarray()
		"""
		assert (isinstance(state_train, np.ndarray)), "Provide state matrix as array"
		assert (isinstance(target_train, np.ndarray)), "Provide target matrix as array"
		if accepted is not None:
			state_train = state_train[:, accepted]
			target_train = target_train[:, accepted]

		if index is not None and index < 0:
			index = -index
			state_train = state_train[:, index:]
			target_train = target_train[:, :-index]
		elif index is not None and index > 0:
			state_train = state_train[:, :-index]
			target_train = target_train[:, index:]

		if display:
			print("\nTraining Readout {0} [{1}]".format(str(self.name), str(self.rule)))
		if self.rule == 'pinv':
			reg = lm.LinearRegression(fit_intercept=False, n_jobs=-1)
			reg.fit(state_train.T, target_train.T)
			self.weights = reg.coef_#np.dot(np.linalg.pinv(np.transpose(state_train)), np.transpose(target_train))
			self.fit_obj = reg #[]

		elif self.rule == 'ridge':
			# Get w_out by ridge regression:
			# a) Obtain regression parameters by cross-validation
			alphas = 10.0 ** np.arange(-5, 4)
			reg = lm.RidgeCV(alphas, fit_intercept=False)
			# b) fit using the best alpha...
			reg.fit(state_train.T, target_train.T)
			# c) get the regression coefficients
			self.weights = reg.coef_
			self.fit_obj = reg

		elif self.rule == 'logistic':
			C = 10.0 ** np.arange(-5, 5)
			reg = lm.LogisticRegressionCV(C, cv=5, penalty='l2', dual=False,
															fit_intercept=False, n_jobs=-1)
			reg.fit(state_train.T, np.argmax(np.array(target_train), 0))
			self.weights = reg.coef_
			self.fit_obj = reg

		elif self.rule == 'perceptron':
			reg = lm.Perceptron(fit_intercept=False)
			reg.fit(state_train.T, np.argmax(np.array(target_train), 0))
			self.weights = reg.coef_
			self.fit_obj = reg

		elif self.rule == 'svm-linear':
			reg = svm.SVC(kernel='linear')
			reg.fit(state_train.T, np.argmax(np.array(target_train), 0))
			self.weights = reg.coef_
			self.fit_obj = reg

		elif self.rule == 'svm-rbf':
			reg = svm.SVC(kernel='rbf')
			print("Performing 5-fold CV for svm-rbf hyperparameters...")
			# use exponentially spaces C...
			C_range = 10.0 ** np.arange(-2, 9)
			# ... and gamma
			gamma_range = 10.0 ** np.arange(-5, 4)
			param_grid = dict(gamma=gamma_range, C=C_range)
			# pick only a subset of train dataset...
			target_test = target_train[:, :target_train.shape[1] / 2]
			state_test = state_train[:, :target_train.shape[1] / 2]
			cv = StratifiedKFold(y=np.argmax(np.array(target_test), 0), n_folds=5)
			grid = GridSearchCV(reg, param_grid=param_grid, cv=cv, n_jobs=-1)
			# use the test dataset (it's much smaller...)
			grid.fit(state_test.T, np.argmax(np.array(target_test), 0))
			print("The best classifier is: ", grid.best_estimator_)

			# use best parameters:
			reg = grid.best_estimator_
			reg.fit(state_train.T, np.argmax(np.array(target_train), 0))
			self.weights = reg.coef0
			self.fit_obj = reg

		elif self.rule == 'elastic':
			# l1_ratio_range = np.logspace(-5, 1, 60)
			print("Performing 5-fold CV for ElasticNet hyperparameters...")
			enet = lm.ElasticNetCV(n_jobs=-1)
			enet.fit(state_train.T, np.argmax(np.array(target_train), 0))

			self.fit_obj = enet
			self.weights = enet.coef_

		elif self.rule == 'bayesian_ridge':
			model = lm.BayesianRidge()
			model.fit(state_train.T, np.argmax(np.array(target_train), 0))

			self.fit_obj = model
			self.weights = model.coef_

		return self.weights, self.fit_obj

	def test(self, state_test, target_test=None, index=None, accepted=None, display=True):
		"""
		Acquire readout output in test phase
		"""
		assert (isinstance(state_test, np.ndarray)), "Provide state matrix as array"
		if target_test is not None:
			assert (isinstance(target_test, np.ndarray)), "Provide target matrix as array"

		if accepted is not None:
			state_test = state_test[:, accepted]
			if target_test is not None:
				target_test = target_test[:, accepted]

		if index is not None and index < 0:
			index = -index
			state_test = state_test[:, index:]
			if target_test is not None:
				target_test = target_test[:, :-index]
		elif index is not None and index > 0:
			state_test = state_test[:, :-index]
			if target_test is not None:
				target_test = target_test[:, index:]

		if display:
			print("\nTesting Readout {0}".format(str(self.name)))
		self.output = None
		self.output = self.fit_obj.predict(state_test.T)

		if target_test is not None:
			return self.output, target_test
		else:
			return self.output

	@staticmethod
	def parse_outputs(output, target, dimensions, set_size, method='WTA', k=1):
		"""

		:param self:
		:return:
		"""
		is_binary_target = np.all(np.unique(target) == [0., 1.])
		is_binary_output = np.all(np.unique(output) == [0., 1.])
		is_labeled_target = (len(target.shape) == 1)
		is_labeled_output = (len(output.shape) == 1)

		# select correct dimensions:
		if not is_labeled_target:
			assert (target.shape[0] == dimensions and target.shape[1] == set_size), \
				"Incorrect target dimensions ({0})".format(str(target.shape))
		else:
			assert (target.shape[0] == set_size), \
				"Incorrect target dimensions ({0})".format(str(target.shape))

		if not is_labeled_output:
			assert (output.shape[0] == dimensions and output.shape[1] == set_size), \
				"Incorrect output dimensions ({0})".format(str(output.shape))
		else:
			assert (output.shape[0] == set_size), \
				"Incorrect output dimensions ({0})".format(str(output.shape))

		# set binary_output and output_labels
		if not is_binary_output:
			binary_output = np.zeros((dimensions, set_size))
			if method == 'WTA':
				for kk in range(output.shape[1]):
					args = np.argsort(output[:, kk])[-k:]
					binary_output[args, kk] = 1.
			else:
				for kk in range(output.shape[1]):
					binary_output[np.where(output[:, kk] >= 0), kk] = 1.
		else:
			binary_output = output
		if not is_labeled_output:
			output_labels = np.argmax(output, 0)
		else:
			output_labels = output

		# set binary_target and target_labels
		if not is_binary_target:
			binary_target = np.zeros((target.shape[0], target.shape[1]))
			if method == 'WTA':
				for kk in range(target.shape[1]):
					args = np.argsort(target[:, kk])[-k:]
					binary_target[args, kk] = 1.
		else:
			binary_target = target
		if not is_labeled_target:
			target_labels = np.argmax(target, 0)
		else:
			target_labels = target

		return binary_output, output_labels, binary_target, target_labels

	def measure_performance(self, target, output=None, comparison_function=None, display=True):
		"""
		Compute readout performance according to different metrics.
		:param target: target output [numpy.array (binary or real-valued) or list (labels)]
		:param output:
		:param labeled:
		:return:
		"""
		assert (isinstance(target, np.ndarray)), "Provide target matrix as array"

		if output is None:
			output = self.output

		# check what type of data is provided
		is_binary_target = np.all(np.unique(target) == [0., 1.])
		is_binary_output = np.all(np.unique(output) == [0., 1.])
		is_labeled_target = (len(target.shape) == 1)
		is_labeled_output = (len(output.shape) == 1)

		if output.shape != target.shape and not is_labeled_output:
			output = output.T

		# set dimensions
		if not is_labeled_target:
			n_out = target.shape[0]
			test_steps = target.shape[1]
		elif not is_labeled_output:
			n_out = output.shape[0]
			test_steps = output.shape[1]
		else:
			raise TypeError("Incorrect data shapes")

		if comparison_function is None:
			binary_output, output_labels, binary_target, target_labels = self.parse_outputs(output, target,
			                                                    dimensions=n_out, set_size=test_steps, k=1)
		else:
			binary_output, output_labels, binary_target, target_labels = self.parse_outputs(output, target,
			                                                    dimensions=n_out, set_size=test_steps,
			                                                    method=comparison_function)
		# print binary_output.shape, binary_target.shape
		# print output_labels, target_labels
		# initialize results dictionary - raw takes the direct readout output, max the binarized output and label the
		#  labels of each step
		performance = {'raw': {}, 'max': {}, 'label': {}}

		if not is_labeled_output:  # some readouts just provide class labels
			# Raw performance measures
			performance['raw']['MSE'] = met.mean_squared_error(target, output)
			performance['raw']['MAE'] = met.mean_absolute_error(target, output)
			print("Readout {0} [raw ouput]: \n  - MSE = {1}".format(str(self.name), str(performance['raw']['MSE'])))
			# if is_binary_target and not is_binary_output and len(output.shape) > 1:
			# 	pb_cc = []
			# 	for n in range(target.shape[0]):
			# 		pb_cc.append(mst.pointbiserialr(np.array(target)[n, :], np.array(output)[n, :])[0])
			# 	performance['raw']['point_bisserial'] = pb_cc

			# Max performance measures
			performance['max']['MSE'] = met.mean_squared_error(binary_target, binary_output)
			performance['max']['MAE'] = met.mean_absolute_error(binary_target, binary_output)
			# performance['max']['performance'] = 1. - np.mean(np.abs(binary_target - binary_output))

			performance['label']['performance'] = met.accuracy_score(target_labels, output_labels)
			performance['label']['hamm_loss'] 	= met.hamming_loss(target_labels, output_labels)
			performance['label']['precision'] 	= met.average_precision_score(binary_output, binary_target,
																			average='weighted')
			performance['label']['f1_score'] 	= met.f1_score(binary_target, binary_output, average='weighted')
			performance['label']['recall'] 		= met.recall_score(target_labels, output_labels, average='weighted')
			performance['label']['confusion'] 	= met.confusion_matrix(target_labels, output_labels)
			performance['label']['jaccard'] 	= met.jaccard_similarity_score(target_labels, output_labels)
			performance['label']['class_support'] = met.precision_recall_fscore_support(target_labels, output_labels)

			print("Readout {0} Performance: \n  - Labels = {1}".format(str(self.name), str(performance['label'][
				'performance'])))
			if display:
				print(met.classification_report(target_labels, output_labels))

		self.performance = performance
		return performance

	def measure_stability(self, display=True):
		"""
		Determine the stability of the solution (norm of weights)
		"""
		self.norm_wout = np.linalg.norm(self.weights)
		if display:
			print("|W_out| [{0}] = {1}".format(self.name, str(self.norm_wout)))
		return self.norm_wout

	def copy(self):
		"""
		Copy the readout object
		:return: new Readout object
		"""
		return copy.deepcopy(self)

	def reset(self):
		"""
		Reset current readout
		:return:
		"""
		initializer = pa.ParameterSet({'label': self.name, 'algorithm': self.rule})
		self.__init__(initializer, False)

	def plot_weights(self, display=True, save=False):
		"""
		Plots a histogram with the current weights
		"""
		vz.plot_w_out(self.weights, label=self.name+'-'+self.rule, display=display, save=save)

	def plot_confusion(self, display=True, save=False):
		"""
		"""
		vz.plot_confusion_matrix(self.performance['label']['confusion'], label=self.name, display=display,
		                                    save=save)


########################################################################################################################
class DecodingLayer(object):
	"""
	The Decoder reads population activity in response to patterned inputs,
	extracts the network state (according to specifications) and trains readout weights
	"""
	def __init__(self, initializer, population):
		"""
		Create and connect decoders to population

		:param initializer: ParameterSet object or dictionary specifying decoding parameters
		:param population:
		"""
		if isinstance(initializer, dict):
			initializer = pa.ParameterSet(initializer)
		assert isinstance(initializer, pa.ParameterSet), "StateExtractor must be initialized with ParameterSet or " \
														  "dictionary"
		self.decoding_pars = initializer
		self.state_variables = initializer.state_variable
		self.reset_state_variables = initializer.reset_states
		self.average_states = initializer.average_states
		self.extractors = []
		self.readouts = [[] for _ in range(len(self.state_variables))]
		self.activity = [None for _ in range(len(self.state_variables))]
		self.state_matrix = [[] for _ in range(len(self.state_variables))]
		self.initial_states = [None for _ in range(len(self.state_variables))]
		self.total_delays = [0. for _ in range(len(self.state_variables))]
		self.source_population = population
		self.state_sample_times = None
		self.sampled_times = []
		self.extractor_resolution = [[] for _ in range(len(self.state_variables))]
		self.standardize_states = initializer.standardize

		for state_idx, state_variable in enumerate(self.state_variables):
			state_specs = initializer.state_specs[state_idx]
			if state_variable == 'V_m':
				mm_specs = pa.extract_nestvalid_dict(state_specs, param_type='device')
				mm 		 = nest.Create('multimeter', 1, mm_specs)
				self.extractors.append(mm)
				nest.Connect(mm, population.gids)
				original_neuron_status = nest.GetStatus(population.gids)
				self.initial_states[state_idx] 		 = np.array([x['V_m'] for x in original_neuron_status])
				self.extractor_resolution[state_idx] = state_specs['interval']
			elif state_variable == 'spikes':
				rec_neuron_pars = {'model': 'iaf_psc_delta', 'V_m': 0., 'E_L': 0., 'C_m': 1.,
				                   'tau_m': state_specs['tau_m'],
				                   'V_th': sys.float_info.max, 'V_reset': 0.,
				                   'V_min': 0.}
				# rec_neuron_pars.update(state_specs)
				filter_neuron_specs = pa.extract_nestvalid_dict(rec_neuron_pars, param_type='neuron')

				rec_neurons = nest.Create(rec_neuron_pars['model'], len(population.gids), filter_neuron_specs)
				if 'origin' in state_specs.keys():
					rec_mm = nest.Create('multimeter', 1, {'record_from': ['V_m'],
					                                       'record_to': ['memory'],
					                                       'interval': state_specs['interval'],
					                                       'origin': state_specs['origin']})
				else:
					rec_mm = nest.Create('multimeter', 1, {'record_from': ['V_m'],
					                                       'record_to': ['memory'],
					                                       'interval': state_specs['interval']})
				self.extractors.append(rec_mm)
				nest.Connect(rec_mm, rec_neurons)
				# connect population to recording neurons with fixed delay == 0.1 (was rec_neuron_pars['interval'])
				nest.Connect(population.gids, rec_neurons, 'one_to_one',
				             syn_spec={'weight': 1., 'delay': 0.1, 'model': 'static_synapse'})

				self.initial_states[state_idx] = np.zeros((len(rec_neurons),))
				self.extractor_resolution[state_idx] = state_specs['interval']
			else:
				if state_variable in nest.GetStatus(population.gids[0])[0]['recordables']:
					mm_specs = pa.extract_nestvalid_dict(state_specs, param_type='device')
					mm = nest.Create('multimeter', 1, mm_specs)
					self.extractors.append(mm)
					nest.Connect(mm, population.gids)
					self.initial_states[state_idx] = np.zeros((len(population.gids),))
					self.extractor_resolution[state_idx] = state_specs['interval']
				else:
					raise NotImplementedError("Acquisition from state variable {0} not implemented yet".format(
						state_variable))
			print("- State acquisition from Population {0} [{1}] - id {2}".format(population.name, state_variable,
			                                                                      self.extractors[-1]))
			if hasattr(initializer, "readout"):
				pars_readout = pa.ParameterSet(initializer.readout[state_idx])
				implemented_algorithms = ['pinv', 'ridge', 'logistic', 'svm-linear', 'svm-rbf', 'perceptron', 'elastic',
				                          'bayesian_ridge']

				for n_readout in range(pars_readout.N):
					if len(pars_readout.algorithm) == pars_readout.N:
						alg = pars_readout.algorithm[n_readout]
					elif len(pars_readout.algorithm) == 1:
						alg = pars_readout.algorithm[0]
					else:
						raise TypeError("Please provide readout algorithm for each readout or a single string, "
										"common to all readouts")

					assert (alg in implemented_algorithms), "Algorithm {0} not implemented".format(alg)

					readout_dict = {'label': pars_readout.labels[n_readout], 'algorithm': alg}
					self.readouts[state_idx].append(Readout(pa.ParameterSet(readout_dict)))

		assert (len(np.unique(np.array(self.extractor_resolution))) == 1), "Output resolution must be common to " \
		                                                                   "all state extractors"

	def flush_records(self):
		"""
		Clear data from NEST devices
		:return:
		"""
		for idx, n_device in enumerate(self.extractors):
			nest.SetStatus(n_device, {'n_events': 0})
			if nest.GetStatus(n_device)[0]['to_file']:
				io.remove_files(nest.GetStatus(n_device)[0]['filenames'])
			print((" - State extractor {0} [{1}] from Population {2}".format(str(self.state_variables[idx]),
			                                                               str(n_device[0]),
			                                                               str(self.source_population.name))))

	def flush_states(self):
		"""
		Clear all data
		:return:
		"""
		print(("\n- Deleting state and activity data from all decoders attached to {0}".format(str(
			self.source_population.name))))
		self.activity = [None for _ in range(len(self.state_variables))]
		self.state_matrix = [[] for _ in range(len(self.state_variables))]

	def extract_activity(self, start=None, stop=None, save=True):
		"""
		Read recorded activity from devices and store it
		:param start:
		:param stop:
		:return:
		"""
		all_responses = []
		print(("\nExtracting and storing recorded activity from state extractors [Population {0}]:".format(str(
			self.source_population.name))))
		for idx, n_state in enumerate(self.extractors):
			print(("  - Reading extractor {0} [{1}]".format(n_state, str(self.state_variables[idx]))))
			start_time1 = time.time()
			if nest.GetStatus(n_state)[0]['to_memory']:
				initializer = n_state
			else:
				initializer = nest.GetStatus(n_state)[0]['filenames']

			# compensate delay in 'spikes' state variable
			if self.state_variables[idx] == 'spikes':
				if not self.total_delays[idx]:
					self.determine_total_delay()
				time_shift = self.total_delays[idx] #self.extractor_resolution[idx] #

			if isinstance(initializer, basestring) or isinstance(initializer, list) or (isinstance(initializer,
			                                                        tuple) and isinstance(initializer[0], basestring)):
				# read data from file
				data = io.extract_data_fromfile(initializer)
				if data is not None:
					if len(data.shape) != 2:
						data = np.reshape(data, (int(len(data)/2), 2))
					if data.shape[1] == 2:
						raise NotImplementedError("Reading spiking activity directly not implemented")
					else:
						neuron_ids = data[:, 0]
						times = data[:, 1]
						if self.state_variables[idx] == 'spikes':
							times -= time_shift
						if start is not None and stop is not None:
							idx1 = np.where(times >= start)[0]
							idx2 = np.where(times <= stop)[0]
							idxx = np.intersect1d(idx1, idx2)
							times = times[idxx]
							neuron_ids = neuron_ids[idxx]
							data = data[idxx, :]
						for nn in range(data.shape[1]):
							if nn > 1:
								sigs = data[:, nn]
								tmp = [(neuron_ids[n], sigs[n]) for n in range(len(neuron_ids))]
								responses = sg.AnalogSignalList(tmp, np.unique(neuron_ids).tolist(), times=times,
								                                t_start=start, t_stop=stop)
			elif isinstance(initializer, tuple) or isinstance(initializer, int):
				# read data in memory
				status_dict = nest.GetStatus(initializer)[0]['events']
				times = status_dict['times']

				if self.state_variables[idx] == 'spikes':
					times -= time_shift

				if start is not None and stop is not None:
					idxx = np.where((times >= start - 0.00001) & (times <= stop + 0.000001))[0]
					times = times[idxx]
					status_dict['V_m'] = status_dict['V_m'][idxx]
					status_dict['senders'] = status_dict['senders'][idxx]
				tmp = [(status_dict['senders'][n], status_dict['V_m'][n]) for n in range(len(status_dict['senders']))]
				responses = sg.AnalogSignalList(tmp, np.unique(status_dict['senders']).tolist(), times=times,
				                                dt=self.extractor_resolution[idx], t_start=start, t_stop=stop)

				# ############# DEBUG ##################################################################################
				# dbg_status_dict = nest.GetStatus(initializer)[0]['events']
				# print "\n\n ----  DEBUGGING  ----- \n"
				# n_id, n_as = responses.analog_signals.iteritems().next()
				# assert n_id == 154 # iaf_psc_delta neuron attached to neuron 0
				#
				# print "\n@spikelist.spiketrains[0]: {0}".format(self.source_population.spiking_activity.spiketrains[1])
				#
				# dbg_vms = [t for idx, t in enumerate(dbg_status_dict['V_m']) if dbg_status_dict['senders'][idx] == 154]
				# print "\n\narray idx -- time point -- vm\n"
				# for idx, vm in enumerate(dbg_vms):
				# 	if idx > 0 and vm > dbg_vms[idx - 1]:
				# 		print "{0}, {1}, {2}".format(idx, idx - 1, vm)
				#
				# sml = self.source_population.spiking_activity.filter_spiketrains(dt=0.1, tau=20., start=0.1, stop=200.1)[0].tolist()
				# print "\n\n@filtered spikes [(arr index, aligned time, V_m)]"
				# for idx, val in enumerate(sml):
				# 	if idx > 0 and val > sml[idx - 1]:
				# 		print idx, idx + 1, val
				# exit(0)
			else:
				raise TypeError("Incorrect Decoder ID")

			all_responses.append(responses)
			print(("Elapsed time: {0} s".format(str(time.time()-start_time1))))
		if save:
			for idx, n_response in enumerate(all_responses):
				self.activity[idx] = n_response
		else:
			return all_responses

	def extract_state_vector(self, time_point=200., save=True, reset=False):
		"""
		Read population responses within a local time window and extract a single state vector at the specified time
		:param time_point: in ms
		:param lag: length of local time window = [time_point-lag, time_point]
		:param save: bool - store state vectors in the decoding layer or return them
		:param reset:
		:return:
		"""
		# set the lag to  be == 2*resolution
		lag = np.unique(self.extractor_resolution)[0] #* 2.
		self.sampled_times.append(time_point)
		if not sg.empty(self.activity):
			responses = self.activity
		else:
			responses = self.extract_activity(start=time_point-lag, stop=time_point, save=False)

		state_vectors = []
		if save and (isinstance(responses, list) and len(self.state_matrix) != len(responses)):
			self.state_matrix = [[] for _ in range(len(responses))]
		elif not save and (isinstance(responses, list)):
			state_vectors = [[] for _ in range(len(responses))]
		elif not save:
			state_vectors = []
		for idx, n in enumerate(responses):
			state_vector = n.as_array()[:, -1]
			if save:
				self.state_matrix[idx].append(state_vector)
			else:
				state_vectors[idx].append(state_vector)
		if reset:
			self.reset_states()
		if not save:
			return state_vectors

	def compile_state_matrix(self, sampling_times=None):
		"""
		After gathering all state vectors, compile a standard state matrix

		:param sampling_times:
		:return:
		"""
		assert self.state_matrix, "State matrix elements need to be stored before calling this function"
		states = []
		if len(self.state_matrix) > 1 and sampling_times is None:
			states = []
			for n_idx, n_state in enumerate(self.state_matrix):
				st = np.array(n_state).T
				if self.standardize_states[n_idx]:
					st = StandardScaler().fit_transform(st.T).T
				states.append(st)
			self.state_matrix = states
		elif len(self.state_matrix) > 1 and sampling_times is not None:
			for n_idx, n_state in enumerate(self.state_matrix):
				states = []
				for idx_state, n_state_mat in enumerate(n_state):
					st = np.array(n_state_mat).T
					if self.standardize_states[n_idx]:
						st = StandardScaler().fit_transform(st.T).T
					states.append(st)
					# states.append(np.array(n_state_mat).T)
				self.state_matrix[n_idx] = states
		elif len(self.state_matrix) == 1 and sampling_times is not None:
			states = []
			for idx_state, n_state_mat in enumerate(self.state_matrix[0]):
				st = np.array(n_state_mat).T
				if self.standardize_states[idx_state]:
					st = StandardScaler().fit_transform(st.T).T
				states.append(st)
				# states.append(np.array(n_state_mat).T)
			self.state_matrix[0] = states
		else:
			st = np.array(self.state_matrix[0]).T
			if self.standardize_states[0]:
				st = StandardScaler().fit_transform(st.T).T
			states.append(st)
			# states.append(st)
			self.state_matrix = states

	def evaluate_decoding(self, n_neurons=10, display=False, save=False):
		"""
		Make sure the state extraction process is consistent

		:param spike_list: raw spiking activity data for this population
		:param n_neurons: choose n random neurons to plot
		:return:
		"""
		spike_list = self.source_population.spiking_activity
		start = spike_list.t_start
		stop = spike_list.t_stop
		neuron_ids = np.random.permutation(spike_list.id_list)[:n_neurons]

		if sg.empty(self.activity):
			self.extract_activity(start=start, stop=stop, save=True)
		vz.plot_response(self.source_population, ids=neuron_ids, display=display, save=save)

	def reset_states(self):
		"""
		Sets all state variables to 0
		:return:
		"""
		for idx_state, n_state in enumerate(self.state_variables):
			if self.reset_state_variables[idx_state]:
				# print("\nReseting {0} state in Population {1}".format(n_state, self.source_population.name))
				if n_state == 'V_m':
					print("Resetting V_m can lead to incorrect results!")
					for idx, neuron_id in enumerate(self.source_population.gids):
						nest.SetStatus([neuron_id], {'V_m': self.initial_states[idx_state][idx]})
				elif n_state == 'spikes':
					recording_neuron_gids = nest.GetStatus(nest.GetConnections(self.extractors[idx_state]), 'target')
					for idx, neuron_id in enumerate(recording_neuron_gids):
						nest.SetStatus([neuron_id], {'V_m': self.initial_states[idx_state][idx]})
				else:
					try:
						for idx, neuron_id in enumerate(self.source_population.gids):
							nest.SetStatus([neuron_id], {n_state: self.initial_states[idx_state][idx]})
					except ValueError:
						print(("State variable {0} cannot be reset".format(n_state)))

	def determine_total_delay(self):
		"""
		Determine the connection delays involved in the decoding layer
		:return:
		"""
		for idx, extractor_id in enumerate(self.extractors):
			status_dict = nest.GetStatus(nest.GetConnections(source=extractor_id))
			tget_gids = [n['target'] for n in status_dict]
			source_neurons = [x for x in tget_gids if x in self.source_population.gids]
			if sg.empty(source_neurons):
				assert (self.state_variables[idx] == 'spikes'), "No connections to {0} extractor".format(
					self.state_variables[idx])
				assert (np.array([nest.GetStatus([x])[0]['model'] == 'iaf_psc_delta' for x in tget_gids]).all()), \
					"No connections to {0} extractor".format(
					self.state_variables[idx])

				net_to_decneurons = na.extract_delays_matrix(src_gids=self.source_population.gids[:10],
				                                                        tgets_gids=tget_gids, progress=False)
				net_to_decneurons_delay = np.unique(np.array(net_to_decneurons[net_to_decneurons.nonzero()].todense()))
				assert (len(net_to_decneurons_delay) == 1), "Heterogeneous delays in decoding layer are not supported.."

				# decneurons_to_mm = na.extract_delays_matrix(src_gids=extractor_id, tgets_gids=tget_gids, progress=False)
				# decneurons_to_mm_delay = np.unique(np.array(decneurons_to_mm[decneurons_to_mm.nonzero()].todense()))
				# assert (len(decneurons_to_mm_delay) == 1), "Heterogeneous delays in decoding layer are not " \
				#                                             "supported.."

				self.total_delays[idx] = float(net_to_decneurons_delay)# + decneurons_to_mm_delay)
			else:
				# delays = na.extract_delays_matrix(src_gids=extractor_id, tgets_gids=tget_gids, progress=False)
				# delay = np.unique(np.array(delays[delays.nonzero()].todense()))
				# assert (len(delay) == 1), "Heterogeneous delays in decoding layer are not supported.."
				self.total_delays[idx] = 0.#0.float(delay)
		print(("\n- total delays in Population {0} DecodingLayer {1}: {2} ms".format(str(self.source_population.name),
		                                                                          str(self.state_variables),
		                                                                          str(self.total_delays))))


'''

	def copy_readout_set(self, n=1):
		"""
		Returns n copies of all the readouts attached to the population
		:param n: number of copies
		:return: list of Readout objects
		"""
		assert self.readouts, "Population {0} doesn't have any readouts attached!".format(self.name)

		all_copies = []
		for n_copy in range(n):
			if isinstance(self.readouts[0], list):
				# nested readouts (multiple state variables for current population)
				copy_readouts = [[] for _ in range(len(self.readouts))]
				for set_index in range(len(self.readouts)):
					for n_readout, readout in enumerate(self.readouts[set_index]):
						copy_readouts[set_index].append(readout.copy())
			else:
				copy_readouts = []
				for n_readout, readout in enumerate(self.readouts):
					copy_readouts.append(readout.copy())
			all_copies.append(copy_readouts)
		return all_copies

'''


def reset_decoders(net, enc_layer):
	"""
	Reset all decoders
	:param net:
	:param enc_layer:
	:return:
	"""
	for ctr, n_pop in enumerate(list(itertools.chain(*[net.merged_populations,
	                                                   net.populations, enc_layer.encoders]))):
		if n_pop.decoding_layer is not None:
			n_pop.decoding_layer.reset_states()



# def train_all_readouts(parameters, net, stim, input_signal, encoding_layer, flush=False, debug=False, plot=True,
# 					   display=True, save=False):
# 	"""
# 		Train all readouts attached to network object
# 	:param parameters:
# 	:return:
# 	"""
# 	from modules.na import Network
# 	from modules.input_architect import InputSignal
# 	from modules.signals import empty
# 	assert(isinstance(net, Network)), "Please provide Network object"
# 	assert(isinstance(parameters, prs.ParameterSet)), "parameters must be a ParameterSet object"
# 	assert(isinstance(input_signal, InputSignal) or isinstance(input_signal, np.ndarray)), \
# 		"input_signal must be an InputSignal object or numpy array / matrix"
#
# 	sampling_rate = parameters.decoding_pars.global_sampling_times
# 	if isinstance(input_signal, np.ndarray):
# 		target 		= input_signal
# 		set_labels 	= stim.train_set_labels
# 	elif sampling_rate is None or isinstance(sampling_rate, list) or isinstance(sampling_rate, np.ndarray):
# 		target 		= stim.train_set.todense()
# 		set_labels 	= stim.train_set_labels
# 	else:
# 		unfold_n = int(round(sampling_rate ** (-1)))
# 		if input_signal.online:
# 			if not isinstance(input_signal.duration_parameters[0], float) or not isinstance(
# 					input_signal.interval_parameters[0], float):
# 				# TODO - implement other variants
# 				raise NotImplementedError("Input signal duration has to be constant.. Variants are not implemented yet")
# 			else:
# 				total_samples = (input_signal.duration_parameters[0] + input_signal.interval_parameters[0]) * len(
# 					stim.train_set_labels)
# 				step_size = input_signal.duration_parameters[0] + input_signal.interval_parameters[0]
# 				target = np.repeat(stim.train_set.todense(), step_size, axis=1)
# 				assert(target.shape[1] == total_samples), "Inconsistent dimensions in setting continuous targets"
# 		else:
# 			target = input_signal.generate_square_signal()[:, ::int(unfold_n)]
# 		onset_idx = [[] for _ in range(target.shape[0])]
# 		offset_idx = [[] for _ in range(target.shape[0])]
# 		labels = []
# 		set_labels = {}
# 		for k in range(target.shape[0]):
# 			stim_idx = np.where(stim.train_set.todense()[k, :])[1]
# 			if stim_idx.shape[1]:
# 				labels.append(np.unique(np.array(stim.train_set_labels)[stim_idx])[0])
# 				if input_signal.online:
# 					iddxs = np.array(np.where(target[k, :])[1])
# 				else:
# 					iddxs = np.array(np.where(target[k, :])[0])
# 				idx_diff = np.array(np.diff(iddxs))
# 				if len(idx_diff.shape) > 1:
# 					idx_diff = idx_diff[0]
# 					iddxs = iddxs[0]
# 				onset_idx[k] = [x for idd, x in enumerate(iddxs) if idx_diff[idd-1] > 1 or x == 0]
# 				offset_idx[k] = [x for idd, x in enumerate(iddxs) if idd<len(iddxs)-1 and (idx_diff[idd] > 1 or x == len(
# 					target[k, :]))]
# 				offset_idx.append(iddxs[-1])
# 		set_labels.update({'dimensions': target.shape[0], 'labels': labels, 'onset_idx': onset_idx, 'offset_idx':
# 			offset_idx})
#
# 	if isinstance(save, dict):
# 		if save['label']:
# 			paths = save
# 			save = True
# 		else:
# 			save = False
#
# 	# read from all state matrices
# 	for ctr, n_pop in enumerate(list(itertools.chain(*[net.merged_populations, net.populations,
# 													   encoding_layer.encoders]))):
# 		if not empty(n_pop.state_matrix):
# 			state_dimensions = np.array(n_pop.state_matrix).shape
# 			population_readouts = n_pop.readouts
# 			chunker = lambda lst, sz: [lst[i:i + sz] for i in range(0, len(lst), sz)]
# 			n_pop.readouts = chunker(population_readouts, len(population_readouts) / state_dimensions[0])
# 			# copy readouts for each state matrix
# 			if n_pop.state_sample_times:
# 				n_copies = len(n_pop.state_sample_times)
# 				all_readouts = n_pop.copy_readout_set(n_copies)
# 				n_pop.readouts = all_readouts
#
# 			for idx_state, n_state in enumerate(n_pop.state_matrix):
# 				if not isinstance(n_state, list):
# 					print("\nTraining {0} readouts from Population {1}".format(str(n_pop.decoding_pars['readout'][
# 																					   'N']), str(n_pop.name)))
# 					label = n_pop.name + '-Train-StateVar{0}'.format(str(idx_state))
# 					if save:
# 						np.save(paths['activity'] + label, n_state)
# 					if debug:
# 						if save:
# 							save_path = paths['figures'] + label
# 						else:
# 							save_path = False
# 						analyse_state_matrix(n_state, set_labels, label=label, plot=plot, display=display,
# 											 save=save_path)
# 					for readout in n_pop.readouts[idx_state]:
# 						readout.set_index()
# 						discrete_readout_train(n_state, target, readout, readout.index)
# 				else:
# 					for iddx_state, nn_state in enumerate(n_state):
# 						readout_set = n_pop.readouts[iddx_state]
# 						print("\nTraining {0} readouts from Population {1} [t = {2}]".format(
# 							str(n_pop.decoding_pars['readout']['N']), str(n_pop.name), str(n_pop.state_sample_times[iddx_state])))
# 						label = n_pop.name + '-Train-StateVar{0}-sample{1}'.format(str(idx_state),
# 																				   str(iddx_state))
# 						if save:
# 							np.save(paths['activity'] + label, n_state)
# 						if debug:
# 							if save:
# 								save_path = paths['figures'] + label
# 							else:
# 								save_path = False
# 							analyse_state_matrix(nn_state, stim.train_set_labels, label=label, plot=plot,
# 												 display=display,
# 												 save=save_path)
# 						for readout in readout_set[idx_state]:
# 							readout.set_index()
# 							discrete_readout_train(nn_state, target, readout, readout.index)
# 				if flush:
# 					n_pop.flush_states()
#
#
# def test_all_readouts(parameters, net, stim, input_signal, encoding_layer=None, flush=False, debug=False, plot=True,
# 					  display=True, save=False):
# 	"""
# 	Test and measure performance of all readouts attached to Network object
# 	:param net:
# 	:param stim:
# 	:param flush:
# 	:return:
# 	"""
# 	assert (isinstance(net, na.Network)), "Please provide Network object"
# 	assert (isinstance(parameters, prs.ParameterSet)), "parameters must be a ParameterSet object"
# 	assert (isinstance(input_signal, ips.InputSignal) or isinstance(input_signal, np.ndarray)), \
# 		"input_signal must be an InputSignal object or numpy array / matrix"
#
# 	sampling_rate = parameters.decoding_pars.global_sampling_times
# 	if isinstance(input_signal, np.ndarray):
# 		# if a custom training target was provided
# 		target 		= input_signal
# 		set_labels 	= stim.test_set_labels
# 	elif sampling_rate is None or isinstance(sampling_rate, list) or isinstance(sampling_rate, np.ndarray):
# 		target 		= stim.test_set.todense()
# 		set_labels 	= stim.test_set_labels
# 	else:
# 		unfold_n 	= int(round(sampling_rate ** (-1)))
# 		target 		= input_signal.generate_square_signal()[:, ::int(unfold_n)]
# 		onset_idx 	= [[] for _ in range(target.shape[0])]
# 		offset_idx 	= [[] for _ in range(target.shape[0])]
# 		labels 		= []
# 		set_labels 	= {}
# 		for k in range(target.shape[0]):
# 			stim_idx = np.where(stim.test_set.todense()[k, :])[1]
# 			if stim_idx.shape[1]:
# 				labels.append(np.unique(np.array(stim.test_set_labels)[stim_idx])[0])
# 				iddxs 		  = np.where(target[k, :])[0]
# 				idx_diff 	  = np.diff(iddxs)
# 				onset_idx[k]  = [x for idd, x in enumerate(iddxs) if idx_diff[idd - 1] > 1 or x == 0]
# 				offset_idx[k] = [x for idd, x in enumerate(iddxs) if
# 								 idd < len(iddxs) - 1 and (idx_diff[idd] > 1 or x == len(target[k, :]))]
# 				offset_idx.append(iddxs[-1])
# 		set_labels.update({'dimensions': target.shape[0], 'labels': labels, 'onset_idx': onset_idx,
# 						   'offset_idx': offset_idx})
#
# 	if isinstance(save, dict):
# 		if save['label']:
# 			paths = save
# 			save = True
# 		else:
# 			save = False
#
# 	# state of merged populations
# 	if encoding_layer is not None:
# 		all_populations = list(itertools.chain(*[net.merged_populations, net.populations, encoding_layer.encoders]))
# 	else:
# 		all_populations = list(itertools.chain(*[net.merged_populations, net.populations]))
#
# 	for ctr, n_pop in enumerate(all_populations):
# 		if not sg.empty(n_pop.state_matrix):
# 			for idx_state, n_state in enumerate(n_pop.state_matrix):
# 				if not isinstance(n_state, list):
# 					print("\nTesting {0} readouts from Population {1} [{2}]".format(str(n_pop.decoding_pars['readout'][
# 												'N']), str(n_pop.name), str(n_pop.state_variables[idx_state])))
# 					label = n_pop.name + '-Test-StateVar{0}'.format(str(idx_state))
# 					if save:
# 						np.save(paths['activity'] + label, n_state)
# 					if debug:
# 						if save:
# 							save_path = paths['figures'] + label
# 						else:
# 							save_path = False
# 						analyse_state_matrix(n_state, set_labels, label=label, plot=plot, display=display,
# 											 save=save_path)
# 					for readout in n_pop.readouts[idx_state]:
# 						discrete_readout_test(n_state, target, readout, readout.index)
# 				else:
# 					for iddx_state, nn_state in enumerate(n_state):
# 						readout_set = n_pop.readouts[iddx_state]
# 						print("\nTesting {0} readouts from Population {1} [t = {2}]".format(
# 							str(n_pop.decoding_pars['readout'][
# 									'N']), str(n_pop.name), str(n_pop.state_sample_times[iddx_state])))
# 						label = n_pop.name + '-Test-StateVar{0}-sample{1}'.format(str(idx_state),
# 																				  str(iddx_state))
# 						if save:
# 							np.save(paths['activity'] + label, n_state)
# 						if debug:
# 							if save:
# 								save_path = paths['figures'] + label
# 							else:
# 								save_path = False
# 							analyse_state_matrix(nn_state, set_labels, label=label, plot=plot,
# 												 display=display,
# 												 save=save_path)
# 						for readout in readout_set[idx_state]:
# 							discrete_readout_test(nn_state, target, readout, readout.index)
# 			if flush:
# 				n_pop.flush_states()
