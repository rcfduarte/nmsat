import numpy as np

from spike_train import SpikeTrain

########################################################################################################################
class AnalogSignal(object):
	"""
    AnalogSignal(signal, dt, t_start=0, t_stop=None)

    Return a AnalogSignal object which will be an analog signal trace

    Inputs:
        signal  - the vector with the data of the AnalogSignal
        dt      - the time step between two data points of the sampled analog signal
        t_start - begining of the signal, in ms.
        t_stop  - end of the SpikeList, in ms. If None, will be inferred from the data

    Examples:
        >> s = AnalogSignal(range(100), dt=0.1, t_start=0, t_stop=10)

    See also
        AnalogSignalList, load_currentlist, load_vmlist, load_conductancelist, load
	"""
	def __init__(self, signal, dt, t_start=None, t_stop=None):
		self.signal = np.array(signal, float)
		self.dt = float(dt)
		if t_start is None:
			t_start = 0
		self.t_start = float(t_start)
		self.t_stop = t_stop
		self.signal_length = len(signal)
		# If t_stop is not None, we test that the signal has the correct number
		# of elements
		if self.t_stop is not None:
			if abs(self.t_stop - self.t_start - self.dt * len(self.signal)) > 0.1 * self.dt:
				raise Exception("Inconsistent arguments: t_start=%g, t_stop=%g, dt=%g implies %d elements, actually %d" % (
					t_start, t_stop, dt, int(round((t_stop-t_start) / float(dt))), len(signal)))
		else:
			self.t_stop = self.t_start + len(self.signal) * self.dt

		if self.t_start >= self.t_stop:
			raise Exception("Incompatible time interval for the creation of the AnalogSignal. t_start=%s, t_stop=%s" % (self.t_start, self.t_stop))

	def __getslice__(self, i, j):
		"""
		Return a sublist of the signal vector of the AnalogSignal
		"""
		return self.signal[i:j]

	def raw_data(self):
		"""
		Return the signal
		"""
		return self.signal

	def duration(self):
		"""
        Return the duration of the SpikeTrain
		"""
		return self.t_stop - self.t_start

	def __str__(self):
		return str(self.signal)

	def __len__(self):
		return len(self.signal)

	def max(self):
		return self.signal.max()

	def min(self):
		return self.signal.min()

	def mean(self):
		return np.mean(self.signal)

	def copy(self):
		"""
        Return a copy of the AnalogSignal object
		"""
		return AnalogSignal(self.signal, self.dt, self.t_start, self.t_stop)

	def time_axis(self, normalized=False):
		"""
        Return the time axis of the AnalogSignal
		"""
		if normalized:
			norm = self.t_start
		else:
			norm = 0.
		return np.arange(self.t_start - norm, self.t_stop - norm, self.dt)

	def time_offset(self, offset):
		"""
        Add a time offset to the AnalogSignal object. t_start and t_stop are
        shifted from offset.

        Inputs:
            offset - the time offset, in ms

        Examples:
            >> as = AnalogSignal(arange(0,100,0.1),0.1)
            >> as.t_stop
                100
            >> as.time_offset(1000)
            >> as.t_stop
                1100
        """
		t_start = self.t_start + offset
		t_stop = self.t_stop + offset
		return AnalogSignal(self.signal, self.dt, t_start, t_stop)

	def time_parameters(self):
		"""
        Return the time parameters of the AnalogSignal (t_start, t_stop, dt)
		"""
		return (self.t_start, self.t_stop, self.dt)

	def time_slice(self, t_start, t_stop):
		"""
        Return a new AnalogSignal obtained by slicing between t_start and t_stop

        Inputs:
            t_start - begining of the new SpikeTrain, in ms.
            t_stop  - end of the new SpikeTrain, in ms.

        See also:
            interval_slice
        """
		assert t_start >= self.t_start
		# assert t_stop <= self.t_stop
		assert t_stop > t_start

		t = self.time_axis()
		i_start = int(round((t_start - self.t_start) / self.dt))
		i_stop = int(round((t_stop - self.t_start) / self.dt))
		signal = self.signal[i_start:i_stop]
		result = AnalogSignal(signal, self.dt, t_start, t_stop)
		return result

	def interval_slice(self, interval):
		"""
        Return only the parts of the AnalogSignal that are defined in the range of the interval.
        The output is therefor a list of signal segments

        Inputs:
            interval - The Interval to slice the AnalogSignal with

        Examples:
            >> as.interval_slice(Interval([0,100],[50,150]))

        See also:
            time_slice
		"""
		result = []
		for itv in interval.interval_data:
			result.append(self.signal[itv[0] / self.dt:itv[1] / self.dt])
		return result

	def threshold_detection(self, threshold=None, format=None, sign='above'):
		"""
        Returns the times when the analog signal crosses a threshold.
        The times can be returned as a numpy.array or a SpikeTrain object
        (default)

        Inputs:
             threshold - Threshold
             format    - when 'raw' the raw events array is returned,
                         otherwise this is a SpikeTrain object by default
             sign      - 'above' detects when the signal gets above the threshodl, 'below when it gets below the threshold'

		Returns:
			- events   - spike times as np.array
			- SpikeTrain object

        Examples:
            >> aslist.threshold_detection(-55, 'raw')
                [54.3, 197.4, 206]
		"""

		assert threshold is not None, "threshold must be provided"

		if sign is 'above':
			cutout = np.where(self.signal > threshold)[0]
		elif sign in 'below':
			cutout = np.where(self.signal < threshold)[0]

		if len(cutout) <= 0:
			events = np.zeros(0)
		else:
			take = np.where(np.diff(cutout) > 1)[0] + 1
			take = np.append(0, take)

			time = self.time_axis()
			events = time[cutout][take]

		if format is 'raw':
			return events
		else:
			return SpikeTrain(events, t_start=self.t_start, t_stop=self.t_stop)

	def event_triggered_average(self, events, average=True, t_min=0, t_max=100, with_time=False):
		"""
        Return the spike triggered averaged of an analog signal according to selected events,
        on a time window t_spikes - tmin, t_spikes + tmax
        Can return either the averaged waveform (average = True), or an array of all the
        waveforms triggered by all the spikes.

        Inputs:
            events  - Can be a SpikeTrain object (and events will be the spikes) or just a list
                      of times
            average - If True, return a single vector of the averaged waveform. If False,
                      return an array of all the waveforms.
            t_min   - Time (>0) to average the signal before an event, in ms (default 0)
            t_max   - Time (>0) to average the signal after an event, in ms  (default 100)

        Examples:
            >> vm.event_triggered_average(spktrain, average=False, t_min = 50, t_max = 150)
            >> vm.event_triggered_average(spktrain, average=True)
            >> vm.event_triggered_average(range(0,1000,10), average=False)
        """

		if isinstance(events, SpikeTrain):
			events = events.spike_times
		else:
			assert np.iterable(events), "events should be a SpikeTrain object or an iterable object"
		assert (t_min >= 0) and (t_max >= 0), "t_min and t_max should be greater than 0"
		assert len(events) > 0, "events should not be empty and should contained at least one element"

		time_axis = np.linspace(-t_min, t_max, (t_min + t_max) / self.dt)
		N = len(time_axis)
		Nspikes = 0.
		if average:
			result = np.zeros(N, float)
		else:
			result = []

		# recalculate everything into timesteps, is more stable against rounding errors
		#  and subsequent cutouts with different sizes
		events = np.floor(np.array(events) / self.dt)
		t_min_l = np.floor(t_min / self.dt)
		t_max_l = np.floor(t_max / self.dt)
		t_start = np.floor(self.t_start / self.dt)
		t_stop = np.floor(self.t_stop / self.dt)

		for spike in events:
			if ((spike - t_min_l) >= t_start) and ((spike + t_max_l) < t_stop):
				spike = spike - t_start
				if average:
					result += self.signal[(spike - t_min_l):(spike + t_max_l)]
				else:
					result.append(self.signal[(spike - t_min_l):(spike + t_max_l)])
				Nspikes += 1
		if average:
			result = result / Nspikes
		else:
			result = np.array(result)

		if with_time:
			return result, time_axis
		else:
			return result

	def slice_by_events(self, events, t_min=100, t_max=100):
		"""
        Returns a dict containing new AnalogSignals cutout around events.

        Inputs:
            events  - Can be a SpikeTrain object (and events will be the spikes) or just a list
                      of times
            t_min   - Time (>0) to cut the signal before an event, in ms (default 100)
            t_max   - Time (>0) to cut the signal after an event, in ms  (default 100)

        Examples:
            >> res = aslist.slice_by_events([100,200,300], t_min=0, t_max =100)
            >> print len(res)
                3
        """
		if isinstance(events, SpikeTrain):
			events = events.spike_times
		else:
			assert np.iterable(events), "events should be a SpikeTrain object or an iterable object"
		assert (t_min >= 0) and (t_max >= 0), "t_min and t_max should be greater than 0"
		assert len(events) > 0, "events should not be empty and should contained at least one element"

		result = {}
		for index, spike in enumerate(events):
			if ((spike - t_min) >= self.t_start) and ((spike + t_max) < self.t_stop):
				spike = spike - self.t_start
				t_start_new = (spike-t_min)
				t_stop_new = (spike+t_max)
				result[index] = self.time_slice(t_start_new, t_stop_new)
		return result

	def mask_events(self,events,t_min=100,t_max=100):
		"""
        Returns a new Analog signal which has self.signal of numpy.ma.masked_array, where the internals (t_i-t_min, t_i+t_max) for events={t_i}
        have been masked out.

        Inputs:
            events  - Can be a SpikeTrain object (and events will be the spikes) or just a list
                      of times
            t_min   - Time (>=0) to cut the signal before an event, in ms (default 100)
            t_max   - Time (>=0) to cut the signal after an event, in ms  (default 100)

        Examples:
            >> res = signal.mask_events([100,200,300], t_min=0, t_max =100)


        Author: Eilif Muller
        """
		from numpy import ma

		if isinstance(events, SpikeTrain):
			events = events.spike_times
		else:
			assert np.iterable(events), "events should be a SpikeTrain object or an iterable object"
		assert (t_min >= 0) and (t_max >= 0), "t_min and t_max should be greater than or equal to 0"
		assert len(events) > 0, "events should not be empty and should contained at least one element"

		result = AnalogSignal(self.signal, self.dt, self.t_start, self.t_stop)
		result.signal = ma.masked_array(result.signal, None)

		for index, spike in enumerate(events):
			t_start_new = np.max([spike - t_min, self.t_start])
			t_stop_new = np.min([spike + t_max, self.t_stop])

			i_start = int(round(t_start_new / self.dt))
			i_stop = int(round(t_stop_new / self.dt))
			result.signal.mask[i_start:i_stop] = True

		return result

	def slice_exclude_events(self,events,t_min=100,t_max=100):
		"""
        yields new AnalogSignals with events cutout (useful for removing spikes).

        Events should be sorted in chronological order

        Inputs:
            events  - Can be a SpikeTrain object (and events will be the spikes) or just a list
                      of times
            t_min   - Time (>0) to cut the signal before an event, in ms (default 100)
            t_max   - Time (>0) to cut the signal after an event, in ms  (default 100)

        Examples:
            >> res = aslist.slice_by_events([100,200,300], t_min=0, t_max =10)
            >> print len(res)
                4

        Author: Eilif Muller
        """
		if isinstance(events, SpikeTrain):
			events = events.spike_times
		else:
			assert np.iterable(events), "events should be a SpikeTrain object or an iterable object"
		assert (t_min >= 0) and (t_max >= 0), "t_min and t_max should be greater than 0"

		# if no events to remove, return self
		if len(events) == 0:
			yield self
			return

		t_last = self.t_start
		for spike in events:
			# skip spikes which aren't close to the signal interval
			if spike + t_min < self.t_start or spike - t_min > self.t_stop:
				continue

			t_min_local = np.max([t_last, spike - t_min])
			t_max_local = np.min([self.t_stop, spike + t_max])

			if t_last < t_min_local:
				yield self.time_slice(t_last, t_min_local)

			t_last = t_max_local

		if t_last < self.t_stop:
			yield self.time_slice(t_last, self.t_stop)

	def cov(self, signal):
		"""

        Returns the covariance of two signals (self, signal),

        i.e. mean(self.signal*signal)-mean(self.signal)*(mean(signal)


        Inputs:
            signal  - Another AnalogSignal object.  It should have the same temporal dimension
                      and dt.

        Examples:
            >> a1 = AnalogSignal(numpy.random.normal(size=1000),dt=0.1)
            >> a2 = AnalogSignal(numpy.random.normal(size=1000),dt=0.1)
            >> print a1.cov(a2)
            -0.043763817072107143
            >> print a1.cov(a1)
            1.0063757246782141

        See also:
            NeuroTools.analysis.ccf
            http://en.wikipedia.org/wiki/Covariance

        Author: Eilif Muller

		"""

		assert signal.dt == self.dt
		assert signal.signal.shape == self.signal.shape

		return np.mean(self.signal * signal.signal) - np.mean(self.signal) * (np.mean(signal.signal))

