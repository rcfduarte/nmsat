import numpy as np
import scipy.signal as sp
import utils
from modules import analysis
from analog_signal import AnalogSignal

def shotnoise_fromspikes(spike_train, q, tau, dt=0.1, t_start=None, t_stop=None, array=False, eps=1.0e-8):
	"""
	Convolves the provided spike train with shot decaying exponentials yielding so called shot noise
	if the spike train is Poisson-like. Returns an AnalogSignal if array=False, otherwise (shotnoise,t)
	as numpy arrays.
	:param spike_train: a SpikeTrain object
	:param q: the shot jump for each spike
	:param tau: the shot decay time constant in milliseconds
	:param dt: the resolution of the resulting shotnoise in milliseconds
	:param t_start: start time of the resulting AnalogSignal. If unspecified, t_start of spike_train is used
	:param t_stop: stop time of the resulting AnalogSignal. If unspecified, t_stop of spike_train is used
	:param array: if True, returns (shotnoise,t) as numpy arrays, otherwise an AnalogSignal.
	:param eps: - a numerical parameter indicating at what value of the shot kernel the tail is cut.  The
	default is
	usually fine.
	Examples:
	---------
		>> stg = stgen.StGen()
		>> st = stg.poisson_generator(10.0,0.0,1000.0)
		>> g_e = shotnoise_fromspikes(st,2.0,10.0,dt=0.1)
	"""
	spike_train = spike_train
	if t_start is not None and t_stop is not None:
		assert t_stop > t_start, "t_stop must be larger than t_start"

	# time of vanishing significance
	vs_t = -tau * np.log(eps / q)

	if t_stop is None:
		t_stop = spike_train.t_stop

	# need to be clever with start time because we want to take spikes into account which occurred in
	# spikes_times
	#  before t_start
	if t_start is None:
		t_start = spike_train.t_start
		window_start = spike_train.t_start
	else:
		window_start = t_start
		if t_start > spike_train.t_start:
			t_start = spike_train.t_start

	t = np.arange(t_start, t_stop, dt)
	kern = q * np.exp(-np.arange(0.0, vs_t, dt) / tau)
	idx = np.clip(np.searchsorted(t, spike_train.spike_times, 'right') - 1, 0, len(t) - 1)
	a = np.zeros(np.shape(t), float)

	a[idx] = 1.0
	y = np.convolve(a, kern)[0:len(t)]

	if array:
		signal_t = np.arange(window_start, t_stop, dt)
		signal_y = y[-len(t):]
		return signal_y, signal_t
	else:
		result = AnalogSignal(y, dt, t_start=0.0, t_stop=t_stop - t_start)
		result.time_offset(t_start)
		if window_start > t_start:
			result = result.time_slice(window_start, t_stop)
		return result



######################################################################################################
class SpikeTrain(object):
	"""
	SpikeTrain(spikes_times, t_start=None, t_stop=None)
	This class defines a spike train as a list of times events.

	Event times are given in a list (sparse representation) in milliseconds.

	Inputs:
		spike_times - a list/numpy array of spike times (in milliseconds)
		t_start     - beginning of the SpikeTrain (if not, this is inferred)
		t_stop      - end of the SpikeTrain (if not, this is inferred)

	Examples:
		>> s1 = SpikeTrain([0.0, 0.1, 0.2, 0.5])
		>> s1.isi()
			array([ 0.1,  0.1,  0.3])
		>> s1.mean_rate()
			8.0
		>> s1.cv_isi()
			0.565685424949
	"""

	#######################################################################
	## Constructor and key methods to manipulate the SpikeTrain objects  ##
	#######################################################################
	def __init__(self, spike_times, t_start=None, t_stop=None):
		"""
		Constructor of the SpikeTrain object
		"""

		self.t_start = t_start
		self.t_stop = t_stop
		self.spike_times = np.array(spike_times, np.float32)

		# If t_start is not None, we resize the spike_train keeping only
		# the spikes with t >= t_start
		if self.t_start is not None:
			self.spike_times = np.extract((self.spike_times >= self.t_start), self.spike_times)

		# If t_stop is not None, we resize the spike_train keeping only
		# the spikes with t <= t_stop
		if self.t_stop is not None:
			self.spike_times = np.extract((self.spike_times <= self.t_stop), self.spike_times)

		# We sort the spike_times. May be slower, but is necessary for quite a
		# lot of methods
		self.spike_times = np.sort(self.spike_times, kind="quicksort")
		# Here we deal with the t_start and t_stop values if the SpikeTrain
		# is empty, with only one element or several elements, if we
		# need to guess t_start and t_stop
		# no element : t_start = 0, t_stop = 0.1
		# 1 element  : t_start = time, t_stop = time + 0.1
		# several    : t_start = min(time), t_stop = max(time)

		size = len(self.spike_times)
		if size == 0:
			if self.t_start is None:
				self.t_start = 0
			if self.t_stop is None:
				self.t_stop = 0.1
		elif size == 1:  # spike list may be empty
			if self.t_start is None:
				self.t_start = self.spike_times[0]
			if self.t_stop is None:
				self.t_stop = self.spike_times[0] + 0.1
		elif size > 1:
			if self.t_start is None:
				self.t_start = np.min(self.spike_times)
			if np.any(self.spike_times < self.t_start):
				raise ValueError("Spike times must not be less than t_start")
			if self.t_stop is None:
				self.t_stop = np.max(self.spike_times)
			if np.any(self.spike_times > self.t_stop):
				raise ValueError("Spike times must not be greater than t_stop")

		if self.t_start >= self.t_stop :
			raise Exception("Incompatible time interval : t_start = %s, t_stop = %s" % (self.t_start, self.t_stop))
		if self.t_start < 0:
			raise ValueError("t_start must not be negative")
		if np.any(self.spike_times < 0):
			raise ValueError("Spike times must not be negative")

	def __str__(self):
		return str(self.spike_times)

	def __del__(self):
		del self.spike_times

	def __len__(self):
		return len(self.spike_times)

	def __getslice__(self, i, j):
		"""
		Return a sub-list of the spike_times vector of the SpikeTrain, indexed by i,j
		"""
		return self.spike_times[i:j]

	def time_parameters(self):
		"""
		Return the time parameters of the SpikeTrain (t_start, t_stop)
		"""
		return (self.t_start, self.t_stop)

	def is_equal(self, spktrain):
		"""
		Return True if the SpikeTrain object is equal to one other SpikeTrain, i.e
		if they have same time parameters and same spikes_times

		Inputs:
			spktrain - A SpikeTrain object

		See also:
			time_parameters()
		"""
		test = (self.time_parameters() == spktrain.time_parameters())
		return np.all(self.spike_times == spktrain.spike_times) and test

	def copy(self):
		"""
		Return a copy of the SpikeTrain object
		"""
		return SpikeTrain(self.spike_times, self.t_start, self.t_stop)

	def duration(self):
		"""
		Return the duration of the SpikeTrain
		"""
		return self.t_stop - self.t_start

	def merge(self, spiketrain, relative=False):
		"""
		Add the spike times from a spiketrain to the current SpikeTrain

		Inputs:
			spiketrain - The SpikeTrain that should be added
			relative - if True, relative_times() is called on both spiketrains before merging

		Examples:
			>> a = SpikeTrain(range(0,100,10),0.1,0,100)
			>> b = SpikeTrain(range(400,500,10),0.1,400,500)
			>> a.merge(b)
			>> a.spike_times
				[   0.,   10.,   20.,   30.,   40.,   50.,   60.,   70.,   80.,
				90.,  400.,  410.,  420.,  430.,  440.,  450.,  460.,  470.,
				480.,  490.]
			>> a.t_stop
				500
		"""
		if relative:
			self.relative_times()
			spiketrain.relative_times()
		self.spike_times = np.insert(self.spike_times, self.spike_times.searchsorted(spiketrain.spike_times), \
		                             spiketrain.spike_times)
		self.t_start = min(self.t_start, spiketrain.t_start)
		self.t_stop = max(self.t_stop, spiketrain.t_stop)

	def format(self, relative=False, quantized=False):
		"""
		Return an array with a new representation of the spike times

		Inputs:
			relative  - if True, spike times are expressed in a relative
					   time compared to the previous one
			quantized - a value to divide spike times with before rounding

		Examples:
			>> st.spikes_times=[0, 2.1, 3.1, 4.4]
			>> st.format(relative=True)
				[0, 2.1, 1, 1.3]
			>> st.format(quantized=2)
				[0, 1, 2, 2]
		"""
		spike_times = self.spike_times.copy()

		if relative and len(spike_times) > 0:
			spike_times[1:] = spike_times[1:] - spike_times[:-1]

		if quantized:
			assert quantized > 0, "quantized must either be False or a positive number"
			#spike_times =  numpy.array([time/self.quantized for time in spike_times],int)
			spike_times = (spike_times/quantized).round().astype('int')

		return spike_times

	def jitter(self, jitter):
		"""
		Returns a new SpikeTrain with spiketimes jittered by a normal distribution.

		Inputs:
			  jitter - sigma of the normal distribution

		Examples:
			  >> st_jittered = st.jitter(2.0)
		"""

		return SpikeTrain(self.spike_times+jitter*(np.random.normal(loc=0.0, scale=1.0, size=self.spike_times.shape[
			0])), t_start=self.t_start, t_stop=self.t_stop)

	#######################################################################
	## Analysis methods that can be applied to a SpikeTrain object       ##
	#######################################################################

	def isi(self):
		"""
		Return an array with the inter-spike intervals of the SpikeTrain

		Examples:
			>> st.spikes_times=[0, 2.1, 3.1, 4.4]
			>> st.isi()
				[2.1, 1., 1.3]

		See also
			cv_isi
		"""
		return np.diff(self.spike_times)

	def mean_rate(self, t_start=None, t_stop=None):
		"""
		Returns the mean firing rate between t_start and t_stop, in spikes/sec

		Inputs:
			t_start - in ms. If not defined, the one of the SpikeTrain object is used
			t_stop  - in ms. If not defined, the one of the SpikeTrain object is used

		Examples:
			>> spk.mean_rate()
				34.2
		"""
		if (t_start is None) & (t_stop is None):
			t_start = self.t_start
			t_stop = self.t_stop
			idx = self.spike_times
		else:
			if t_start is None:
				t_start = self.t_start
			else:
				t_start = max(self.t_start, t_start)
			if t_stop is None:
				t_stop = self.t_stop
			else:
				t_stop = min(self.t_stop, t_stop)
			idx = np.where((self.spike_times >= t_start) & (self.spike_times <= t_stop))[0]
		return 1000. * len(idx) / (t_stop - t_start)

	def cv_isi(self):
		"""
		Return the coefficient of variation of the isis.

		cv_isi is the ratio between the standard deviation and the mean of the ISI
		  The irregularity of individual spike trains is measured by the squared
		coefficient of variation of the corresponding inter-spike interval (ISI)
		distribution normalized by the square of its mean.
		  In point processes, low values reflect more regular spiking, a
		clock-like pattern yields CV2= 0. On the other hand, CV2 = 1 indicates
		Poisson-type behavior. As a measure for irregularity in the network one
		can use the average irregularity across all neurons.

		http://en.wikipedia.org/wiki/Coefficient_of_variation

		See also
			isi, cv_kl

		"""
		isi = self.isi()
		if len(isi) > 1:
			return np.std(isi)/np.mean(isi)
		else:
			#print("Warning, a CV can't be computed because there are not enough spikes")
			return np.nan

	def cv_kl(self, bins=100):
		"""
		Provides a measure for the coefficient of variation to describe the
		regularity in spiking networks. It is based on the Kullback-Leibler
		divergence and decribes the difference between a given
		interspike-interval-distribution and an exponential one (representing
		poissonian spike trains) with equal mean.
		It yields 1 for poissonian spike trains and 0 for regular ones.

		Reference:
			http://invibe.net/LaurentPerrinet/Publications/Voges08fens

		Inputs:
			bins - the number of bins used to gather the ISI

		Examples:
			> spklist.cv_kl(100)
				0.98

		See also:
			cv_isi
		"""
		isi = self.isi() / 1000.
		if len(isi) < 2:
			#print("Warning, a CV can't be computed because there are not enough spikes")
			return np.nan
		else:
			proba_isi, xaxis = np.histogram(isi, bins=bins, normed=True)
			xaxis = xaxis[:-1]
			proba_isi /= np.sum(proba_isi)
			bin_size = xaxis[1] - xaxis[0]
			# differential entropy: http://en.wikipedia.org/wiki/Differential_entropy
			KL = - np.sum(proba_isi * np.log(proba_isi + 1e-16)) + np.log(bin_size)
			KL -= -np.log(self.mean_rate()) + 1.
			CVkl = np.exp(-KL)
			return CVkl

	def fano_factor_isi(self):
		"""
		Return the fano factor of this spike trains ISI.

		The Fano Factor is defined as the variance of the isi divided by the mean of the isi

		http://en.wikipedia.org/wiki/Fano_factor

		See also
			isi, cv_isi
		"""
		isi = self.isi()
		if len(isi) > 1:
			fano = np.var(isi)/np.mean(isi)
			return fano
		else:
			raise Exception("No spikes in the SpikeTrain !")

	def time_axis(self, time_bin=10):
		"""
		Return a time axis between t_start and t_stop according to a time_bin

		Inputs:
			time_bin - the bin width

		Examples:
			>> st = SpikeTrain(range(100),0.1,0,100)
			>> st.time_axis(10)
				[ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

		See also
			time_histogram
		"""
		axis = np.arange(self.t_start, self.t_stop+time_bin, time_bin)
		return axis

	def time_offset(self, offset, return_new=False):
		"""
		Add an offset to the SpikeTrain object. t_start and t_stop are
		shifted from offset, so does all the spike times.

		Inputs:
			offset - the time offset, in ms

		Examples:
			>> spktrain = SpikeTrain(arange(0,100,10))
			>> spktrain.time_offset(50)
			>> spklist.spike_times
				[  50.,   60.,   70.,   80.,   90.,  100.,  110.,
				120.,  130.,  140.]
		"""
		if return_new:
			return SpikeTrain(self.spike_times + offset, self.t_start+offset, self.t_stop+offset)
		else:
			self.t_start += offset
			self.t_stop += offset
			self.spike_times += offset

	def time_slice(self, t_start, t_stop):
		"""
		Return a new SpikeTrain obtained by slicing between t_start and t_stop,
		where t_start and t_stop may either be single values or sequences of
		start and stop times.

		Inputs:
			t_start - begining of the new SpikeTrain, in ms.
			t_stop  - end of the new SpikeTrain, in ms.

		Examples:
			>> spk = spktrain.time_slice(0,100)
			>> spk.t_start
				0
			>> spk.t_stop
				100
			>> spk = spktrain.time_slice([20,70], [40,90])
			>> spk.t_start
				20
			>> spk.t_stop
				90
			>> len(spk.time_slice(41, 69))
				0
		"""
		if hasattr(t_start, '__len__'):
			if len(t_start) != len(t_stop):
				raise ValueError("t_start has %d values and t_stop %d. They must be of the same length." % (len(t_start), len(t_stop)))
			mask = False
			for t0, t1 in zip(t_start, t_stop):
				mask = mask | ((self.spike_times >= t0) & (self.spike_times <= t1))
			t_start = t_start[0]
			t_stop = t_stop[-1]
		else:
			mask = (self.spike_times >= t_start) & (self.spike_times <= t_stop)
		spikes = np.extract(mask, self.spike_times)
		return SpikeTrain(spikes, t_start, t_stop)

	def interval_slice(self, interval):
		"""
		Return a new SpikeTrain obtained by slicing with an Interval. The new
		t_start and t_stop values of the returned SpikeTrain are the extrema of the Interval

		Inputs:
			interval - The interval from which spikes should be extracted

		Examples:
			>> spk = spktrain.time_slice(0,100)
			>> spk.t_start
				0
			>> spk.t_stop
				100
		"""
		times = interval.slice_times(self.spike_times)
		t_start, t_stop = interval.time_parameters()
		return SpikeTrain(times, t_start, t_stop)

	def time_histogram(self, time_bin=10, normalized=True, binary=False):
		"""
		Bin the spikes with the specified bin width. The first and last bins
		are calculated from `self.t_start` and `self.t_stop`.

		Inputs:
			time_bin   - the bin width for gathering spikes_times
			normalized - if True, the bin values are scaled to represent firing rates
						 in spikes/second, otherwise otherwise it's the number of spikes
						 per bin.
			binary     - if True, a binary matrix of 0/1 is returned

		Examples:
			>> st=SpikeTrain(range(0,100,5),0.1,0,100)
			>> st.time_histogram(10)
				[200, 200, 200, 200, 200, 200, 200, 200, 200, 200]
			>> st.time_histogram(10, normalized=False)
				[2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

		See also
			time_axis
		"""
		bins = self.time_axis(time_bin)
		hist, edges = np.histogram(self.spike_times, bins)
		hist = hist.astype(float)
		if normalized:  # what about normalization if time_bin is a sequence?
			hist *= 1000.0/float(time_bin)
		if binary:
			hist = hist.astype(bool).astype(int)
		return hist

	def fano_factor(self, time_bin=10):
		"""
		Determine the fano factor for each spike
		:param time_bin:
		:return:
		"""
		counts = self.time_histogram(time_bin=time_bin, normalized=False, binary=False)

		return np.var(counts) / np.mean(counts)

	def instantaneous_rate(self, resolution, kernel, norm, m_idx=None, t_start=None, t_stop=None, acausal=True,
	                       trim=False):
		"""
		Estimate instantaneous firing rate by kernel convolution.

		Inputs:
			resolution  - time stamp resolution of the spike times (ms). the
			              same resolution will be assumed for the kernel
			kernel      - kernel function used to convolve with
			norm        - normalization factor associated with kernel function
						  (see analysis.make_kernel for details)
			t_start     - start time of the interval used to compute the firing
						  rate
			t_stop      - end time of the interval used to compute the firing
						  rate (included)
			acausal     - if True, acausal filtering is used, i.e., the gravity
						  center of the filter function is aligned with the
						  spike to convolve
			m_idx       - index of the value in the kernel function vector that
						  corresponds to its gravity center. this parameter is
						  not mandatory for symmetrical kernels but it is
						  required when assymmetrical kernels are to be aligned
						  at their gravity center with the event times
			trim        - if True, only the 'valid' region of the convolved
						  signal are returned, i.e., the points where there
						  isn't complete overlap between kernel and spike train
						  are discarded
						  NOTE: if True and an assymetrical kernel is provided
						  the output will not be aligned with [t_start, t_stop]

		See also:
			analysis.make_kernel
		"""

		if t_start is None:
			t_start = self.t_start

		if t_stop is None:
			t_stop = self.t_stop

		if m_idx is None:
			m_idx = kernel.size / 2

		time_vector = np.zeros((t_stop - t_start)/resolution + 1)

		spikes_slice = self.spike_times[(self.spike_times >= t_start) & (self.spike_times <= t_stop)]

		for spike in spikes_slice:
			index = (spike - t_start) / resolution
			time_vector[index] = 1

		r = norm * sp.fftconvolve(time_vector, kernel, 'full')

		if acausal is True:
			if trim is False:
				r = r[m_idx:-(kernel.size - m_idx)]
				t_axis = np.linspace(t_start, t_stop, r.size)
				return t_axis, r

			elif trim is True:
				r = r[2 * m_idx:-2*(kernel.size - m_idx)]
				t_start += m_idx * resolution
				t_stop -= (kernel.size - m_idx) * resolution
				t_axis = np.linspace(t_start, t_stop, r.size)
				return t_axis, r

		if acausal is False:
			if trim is False:
				r = r[m_idx:-(kernel.size - m_idx)]
				t_axis = (np.linspace(t_start, t_stop, r.size) + m_idx * resolution)
				return t_axis, r

			elif trim is True:
				r = r[2 * m_idx:-2*(kernel.size - m_idx)]
				t_start += m_idx * resolution
				t_stop -= ((kernel.size) - m_idx) * resolution
				t_axis = (np.linspace(t_start, t_stop, r.size) + m_idx * resolution)
				return t_axis, r

	def relative_times(self):
		"""
		Rescale the spike times to make them relative to t_start.

		Note that the SpikeTrain object itself is modified, t_start
		is subtracted from spike_times, t_start and t_stop
		"""
		if self.t_start != 0:
			self.spike_times -= self.t_start
			self.t_stop -= self.t_start
			self.t_start = 0.0

	def round_times(self, resolution=0.1):
		"""
		Round the spike times to a given number of decimal places
		:param resolution:
		:return:
		"""
		decimal_places = str(resolution)[::-1].find('.')
		self.spike_times = np.array([round(n, decimal_places) for n in self.spike_times])

	def distance_victorpurpura(self, spktrain, cost=0.5):
		"""
		Function to calculate the Victor-Purpura distance between two spike trains.
		See J. D. Victor and K. P. Purpura,
			Nature and precision of temporal coding in visual cortex: a metric-space
			analysis.,
			J Neurophysiol,76(2):1310-1326, 1996

		Inputs:
			spktrain - the other SpikeTrain
			cost     - The cost parameter. See the paper for more information
		"""
		nspk_1 = len(self)
		nspk_2 = len(spktrain)
		if cost == 0:
			return abs(nspk_1-nspk_2)
		elif cost > 1e9:
			return nspk_1+nspk_2
		scr = np.zeros((nspk_1+1, nspk_2+1))
		scr[:, 0] = np.arange(0, nspk_1+1)
		scr[0, :] = np.arange(0, nspk_2+1)

		if nspk_1 > 0 and nspk_2 > 0:
			for i in xrange(1, nspk_1+1):
				for j in xrange(1, nspk_2+1):
					scr[i, j] = min(scr[i-1, j]+1, scr[i, j-1]+1)
					scr[i, j] = min(scr[i, j], scr[i-1, j-1]+cost*abs(self.spike_times[i-1]-spktrain.spike_times[j-1]))
		return scr[nspk_1, nspk_2]

	def distance_kreuz(self, spktrain, dt=0.1):
		"""
		Function to calculate the Kreuz/Politi distance between two spike trains
		See  Kreuz, T.; Haas, J.S.; Morelli, A.; Abarbanel, H.D.I. & Politi, A.
			Measuring spike train synchrony.
			J Neurosci Methods, 165:151-161, 2007

		Inputs:
			spktrain - the other SpikeTrain
			dt       - the bin width used to discretize the spike times

		Examples:
			>> spktrain.KreuzDistance(spktrain2)

		See also
			VictorPurpuraDistance
		"""
		N = (self.t_stop-self.t_start)/dt
		vec_1 = np.zeros(N, np.float32)
		vec_2 = np.zeros(N, np.float32)
		result = np.zeros(N, float)
		idx_spikes = np.array(self.spike_times/dt, int)
		previous_spike = 0
		if len(idx_spikes) > 0:
			for spike in idx_spikes[1:]:
				vec_1[previous_spike:spike] = (spike-previous_spike)
				previous_spike = spike
		idx_spikes = np.array(spktrain.spike_times/dt, int)
		previous_spike = 0
		if len(idx_spikes) > 0:
			for spike in idx_spikes[1:]:
				vec_2[previous_spike:spike] = (spike-previous_spike)
				previous_spike = spike
		idx = np.where(vec_1 < vec_2)[0]
		result[idx] = vec_1[idx]/vec_2[idx] - 1
		idx = np.where(vec_1 > vec_2)[0]
		result[idx] = -vec_2[idx]/vec_1[idx] + 1
		return np.sum(np.abs(result))/len(result)

	def psth(self, events, time_bin=2, t_min=50, t_max=50, average=True):
		"""
		Return the psth of the spike times contained in the SpikeTrain according to selected events,
		on a time window t_spikes - tmin, t_spikes + tmax

		Inputs:
			events  - Can be a SpikeTrain object (and events will be the spikes) or just a list
					  of times
			time_bin- The time bin (in ms) used to gather the spike for the psth
			t_min   - Time (>0) to average the signal before an event, in ms (default 0)
			t_max   - Time (>0) to average the signal after an event, in ms  (default 100)

		Examples:
			>> spk.psth(spktrain, t_min = 50, t_max = 150)
			>> spk.psth(spktrain, )

		See also
			SpikeTrain.spike_histogram
		"""

		if isinstance(events, SpikeTrain):
			events = events.spike_times
		assert (t_min >= 0) and (t_max >= 0), "t_min and t_max should be greater than 0"
		assert len(events) > 0, "events should not be empty and should contained at least one element"

		spk_hist = self.time_histogram(time_bin)
		count = 0
		t_min_l = np.floor(t_min / time_bin)
		t_max_l = np.floor(t_max / time_bin)
		result = np.zeros((t_min_l + t_max_l), np.float32)
		t_start = np.floor(self.t_start / time_bin)
		t_stop = np.floor(self.t_stop / time_bin)
		result = []
		for ev in events:
			ev = np.floor(ev / time_bin)
			if ((ev - t_min_l) > t_start) and (ev + t_max_l) < t_stop:
				count += 1
				result += [spk_hist[(ev - t_min_l):ev + t_max_l]]
		result = np.array(result)
		if average:
			result /= count
		return result

	def lv(self):
		"""
		coefficient of local variation
		:return:
		"""
		# convert to array, cast to float
		v = self.isi()

		# ensure we have enough entries
		if v.size < 2:
			return np.nan
		# calculate LV and return result
		else:
			#  raise error if input is multi-dimensional
			return 3. * np.mean(np.power(np.diff(v) / (v[:-1] + v[1:]), 2))

	def lv_r(self, R=2.):
		"""
		Revised local variation coefficient
		Based on Shinomoto, S. et al., PLoS Comp.Bio., Relating firing patterns to Functional differentiation...
		:param R: Refractory time
		:return:
		"""
		# convert to array, cast to float
		v = self.isi()

		# ensure we have enough entries
		if v.size < 2:
			return np.nan
		# calculate LV and return result
		else:
			sum_lvr = 0.
			for i in range(len(v) - 1):
				sum_lvr += ((v[i] - v[i + 1]) ** 2.) / ((v[i] + v[i + 1] - 2 * R) ** 2)
			return 3. / (len(v) - 1) * sum_lvr

	def isi_entropy(self, n_bins=100):
		isi = self.isi()
		if not utils.empty(self.isi()):
			log_isi = np.log(isi)
			weights = np.ones_like(log_isi) / len(log_isi)
			n, bins = np.histogram(log_isi, n_bins, weights=weights)  # , normed=True)
			ent = []
			for prob_mass in n:
				ent.append(prob_mass * np.log2(prob_mass))
			ent = np.array(ent)
			ent = ent[~np.isnan(ent)]
			H = -np.sum(ent)
			return H
		else:
			return np.nan

	def ir(self):
		"""
		Instantaneous Irregularity
		:param isi:
		:return:
		"""
		isi = self.isi()
		iR = []
		for n in range(len(isi) - 1):
			iR.append(np.abs(np.log(isi[n + 1] / isi[n])))
		if not utils.empty(self.isi()):
			return np.mean(iR)
		else:
			return np.nan

	def cv_log_isi(self):
		"""
		CV of the log isis
		:param isi:
		:return:
		"""
		isi = self.isi()
		log_isi = np.log(isi)
		if not utils.empty(self.isi()):
			return np.std(log_isi) / np.mean(log_isi)
		else:
			return np.nan

	def isi_5p(self):
		"""
		returns the 5th percentile of the isi distribution
		:return:
		"""
		if not utils.empty(self.isi()):
			return np.percentile(self.isi(), 5)
		else:
			return np.nan

	def frequency_spectrum(self, time_bin):
		"""
		Returns the frequency spectrum of the time histogram together with the
		frequency axis.
		"""
		hist = self.time_histogram(time_bin, normalized=False)
		freq_spect = analysis.simple_frequency_spectrum(hist)
		freq_bin = 1000.0 / self.duration()  # Hz
		freq_axis = np.arange(len(freq_spect)) * freq_bin
		return freq_spect, freq_axis

	def adaptation_index(self, k=2):
		"""
		Computed the isi adaptation
		:param k: discard k initial isis
		:return:
		"""
		n = len(self.isi())
		l = []
		for iddx, nn in enumerate(self.isi()):
			if iddx > k:
				l.append((nn - self.isi()[iddx - 1]) / (nn + self.isi()[iddx - 1]))

		return np.sum(l) / (n - k - 1)

	def spikes_to_states_exp(self, dt, tau, start=None, stop=None):
		"""
		Converts a spike train into an analogue variable (low-pass filters the spike train),
		by convolving it with an exponential function.
		:parameter dt: resolution
		:parameter tau: filter time constant
		:parameter start:
		:parameter stop:
		"""
		if start is None:
			start = self.t_start
		if stop is None:
			stop = self.t_stop
		SpkTimes = np.round(self.spike_times, 1)

		if not utils.empty(SpkTimes):
			(States, TimeVec) = utils.shotnoise_fromspikes(self, 1., tau, dt, array=True)
			return States

	def spikes_to_states_binary(self, dt, start=None, stop=None):
		"""
		Converts a spike train into a binary time series
		Inputs:
			dt     - time step
			tau    - decay time constant
		"""
		if start is None:
			start = self.t_start
		if stop is None:
			stop = self.t_stop
		SpkTimes = np.round(self.spike_times, 1)
		TimeVec = np.arange(start, stop, 0.1)
		sample_rate = int(dt / 0.1)
		States = np.zeros_like(TimeVec)

		if not utils.empty(SpkTimes):
			Spk_idxs = []
			for x in SpkTimes:
				if not utils.empty(np.where(round(x, 1) == np.round(TimeVec, 1))):
					Spk_idxs.append(np.where(round(x, 1) == np.round(TimeVec, 1))[0][0])
			States = np.zeros_like(TimeVec)
			state = 0.
			for i, t in enumerate(TimeVec):
				state = 0
				if i in Spk_idxs:
					state += 1.
				States[i] = state
			States = States[::sample_rate]
		return States

