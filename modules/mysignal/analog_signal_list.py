import re
import numpy as np

import io
from spike_list import SpikeList
from analog_signal import AnalogSignal
########################################################################################################################
class AnalogSignalList(object):
	"""
    AnalogSignalList(signals, id_list, times=None, dt=None, t_start=None, t_stop=None, dims=None)

    Return a AnalogSignalList object which will be a list of AnalogSignal objects.

    Inputs:
        signals - a list of tuples (id, value) with all the values sorted in time of the analog signals
        id_list - the list of the ids of all recorded cells (needed for silent cells)
        times   - array of sampled time points (if dt is None, it will be inferred from times)
        dt      - if dt is specified, time values should be floats
        t_start - beginning of the List, in ms.
        t_stop  - end of the List, in ms. If None, will be inferred from the data
        dims    - dimensions of the recorded population, if not 1D population

    dt, t_start and t_stop are shared for all SpikeTrains object within the SpikeList

    See also
        load_currentlist load_vmlist, load_conductancelist
	"""
	def __init__(self, signals, id_list, dt=None, times=None, t_start=None, t_stop=None, dims=None):

		if dt is None:
			assert times is not None, "dt or times must be specified"
			dt = np.mean(np.diff(np.unique(times)))
		self.dt = np.round(float(dt), 1)

		if t_start is None and times is None:
			t_start = 0.
		elif t_start is None and times is not None:
			t_start = min(times)
			if abs(np.round(min(times) - np.round(min(times)), 1)) <= self.dt:
					t_start = np.round(min(times))
		self.t_start = t_start
		#
		if t_stop is None and times is None:
			t_stop = len(signals) * self.dt
		elif t_stop is None and times is not None:
			t_stop = max(times)
			if abs(np.round(max(times) - np.round(max(times)), 1)) <= self.dt:
				t_stop = np.round(max(times))
				if t_stop == max(times) and self.dt >= 1.:
					t_stop += self.dt
		self.t_stop = round(t_stop, 1)

		self.dimensions = dims
		self.analog_signals = {}
		self.signal_length = len(signals)
		# print "%s" % str(self.signal_length)

		signals = np.array(signals)

		lens = []
		for id in id_list:
			signal = np.transpose(signals[signals[:, 0] == id, :])[1]
			lens.append(len(signal))
		lens = np.array(lens)

		for id in id_list:
			signal = np.transpose(signals[signals[:, 0] == id, :])[1]
			if len(signal) > 0 and len(signal) == max(lens):
				# print max(lens)
				self.analog_signals[id] = AnalogSignal(signal, self.dt, self.t_start, self.t_stop)
			elif len(signal) > 0 and len(signal) != max(lens):
				sig = np.zeros(max(lens))
				sig[:len(signal)] = signal.copy()
				steps_left = max(lens) - len(signal)
				# print steps_left
				sig[len(signal):] = np.ones(steps_left) * signal[-1]
				# print len(sig)
				self.analog_signals[id] = AnalogSignal(sig, self.dt, self.t_start, self.t_stop)

		signals = self.analog_signals.values()
		if signals:
			self.signal_length = len(signals[0])
			for signal in signals:
				if len(signal) != self.signal_length:
					raise Exception("Signals must all be the same length %d != %d" % (self.signal_length, len(signal)))

		if t_stop is None:
			self.t_stop = self.t_start + self.signal_length * self.dt

	def id_list(self):
		"""
        Return the list of all the cells ids contained in the
        SpikeList object
		"""
		return np.sort(np.array(self.analog_signals.keys()))

	def copy(self):
		"""
        Return a copy of the AnalogSignalList object
        """
		# Maybe not optimal, should be optimized
		aslist = AnalogSignalList([], [], self.dt, self.t_start, self.t_stop, self.dimensions)
		for id in self.id_list():
			aslist.append(id, self.analog_signals[id])
		return aslist

	def __getitem__(self, id):
		if id in self.id_list():
			return self.analog_signals[id]
		else:
			raise Exception("id %d is not present in the AnalogSignal. See id_list()" %id)

	def __setitem__(self, i, val):
		assert isinstance(val, AnalogSignal), "An AnalogSignalList object can only contain AnalogSignal objects"
		if len(self) > 0:
			errmsgs = []
			for attr in "dt", "t_start", "t_stop":
				if getattr(self, attr) == 0:
					if getattr(val, attr) != 0:
						errmsgs.append("%s: %g != %g (diff=%g)" % (attr, getattr(val, attr), getattr(self, attr),
						                                           getattr(val, attr) - getattr(self, attr)))
				elif (getattr(val, attr) - getattr(self, attr))/getattr(self, attr) > 1e-12:
					errmsgs.append("%s: %g != %g (diff=%g)" % (attr, getattr(val, attr), getattr(self, attr),
					                                           getattr(val, attr)-getattr(self, attr)))
			if len(val) != self.signal_length:
				errmsgs.append("signal length: %g != %g" % (len(val), self.signal_length))
			if errmsgs:
				raise Exception("AnalogSignal being added does not match the existing signals: "+", ".join(errmsgs))
		else:
			self.signal_length = len(val)
			self.t_start = val.t_start
			self.t_stop = val.t_stop
		self.analog_signals[i] = val

	def __len__(self):
		return len(self.analog_signals)

	def __iter__(self):
		return self.analog_signals.itervalues()

	def __sub_id_list(self, sub_list=None):
		if sub_list == None:
			return self.id_list()
		if type(sub_list) == int:
			return np.random.permutation(self.id_list())[0:sub_list]
		if type(sub_list) == list:
			return sub_list

	def append(self, id, signal):
		"""
        Add an AnalogSignal object to the AnalogSignalList

        Inputs:
            id     - the id of the new cell
            signal - the AnalogSignal object representing the new cell

        The AnalogSignal object is sliced according to the t_start and t_stop times
        of the AnalogSignallist object

        See also
            __setitem__
        """
		assert isinstance(signal, AnalogSignal), "An AnalogSignalList object can only contain AnalogSignal objects"
		if id in self.id_list():
			raise Exception("Id already present in AnalogSignalList. Use setitem instead()")
		else:
			self[id] = signal

	def time_axis(self):
		"""
        Return the time axis of the AnalogSignalList object
        """
		return np.arange(self.t_start, self.t_stop, self.dt)

	def id_offset(self, offset):
		"""
        Add an offset to the whole AnalogSignalList object. All the id are shifted
        with a offset value.

        Inputs:
            offset - the id offset

        Examples:
            >> as.id_list()
                [0,1,2,3,4]
            >> as.id_offset(10)
            >> as.id_list()
                [10,11,12,13,14]
        """
		id_list = np.sort(self.id_list())
		N = len(id_list)
		for idx in xrange(1, len(id_list) + 1):
			id = id_list[N - idx]
			spk = self.analog_signals.pop(id)
			self.analog_signals[id + offset] = spk

	def id_slice(self, id_list):
		"""
        Return a new AnalogSignalList obtained by selecting particular ids

        Inputs:
            id_list - Can be an integer (and then N random cells will be selected)
                      or a sublist of the current ids

        The new AnalogSignalList inherits the time parameters (t_start, t_stop, dt)

        See also
            time_slice
        """
		new_AnalogSignalList = AnalogSignalList([], [], dt=self.dt, t_start=self.t_start, t_stop=self.t_stop,
		                                        dims=self.dimensions)
		id_list = self.__sub_id_list(id_list)
		for id in id_list:
			try:
				new_AnalogSignalList.append(id, self.analog_signals[id])
			except Exception:
				print "id %d is not in the source AnalogSignalList" %id
		return new_AnalogSignalList

	def time_offset(self, offset):
		"""
		Shifts the time axis by offset
		:param offset:
		:return:
		"""
		new_AnalogSignalList = AnalogSignalList([], [], dt=self.dt, t_start=self.t_start+offset,
		                                        t_stop=self.t_stop+offset, dims=self.dimensions)
		for id in self.id_list():
			an_signal = self.analog_signals[id].copy()
			new_an_signal = an_signal.time_offset(offset)
			new_AnalogSignalList.append(id, new_an_signal)

		return new_AnalogSignalList

	def time_slice(self, t_start, t_stop):
		"""
        Return a new AnalogSignalList obtained by slicing between t_start and t_stop

        Inputs:
            t_start - begining of the new AnalogSignalList, in ms.
            t_stop  - end of the new AnalogSignalList, in ms.

        See also
            id_slice
        """
		new_AnalogSignalList = AnalogSignalList([], [], dt=self.dt, t_start=t_start, t_stop=t_stop,
		                                        dims=self.dimensions)
		for id in self.id_list():
			new_AnalogSignalList.append(id, self.analog_signals[id].time_slice(t_start, t_stop))
		return new_AnalogSignalList

	def select_ids(self, criteria=None):
		"""
        Return the list of all the cells in the AnalogSignalList that will match the criteria
        expressed with the following syntax.

        Inputs :
            criteria - a string that can be evaluated on a AnalogSignal object, where the
                       AnalogSignal should be named ``cell''.

        Exemples:
            >> aslist.select_ids("mean(cell.signal) > 20")
            >> aslist.select_ids("cell.std() < 0.2")
        """
		selected_ids = []
		for id in self.id_list():
			cell = self.analog_signals[id]
			if eval(criteria):
				selected_ids.append(id)
		return selected_ids

	def convert(self, format="[values, ids]"):
		"""
        Return a new representation of the AnalogSignalList object, in a user designed format.
            format is an expression containing either the keywords values and ids,
            time and id.

        Inputs:
            format    - A template to generate the corresponding data representation, with the keywords
                        values and ids

        Examples:
            >> aslist.convert("[values, ids]") will return a list of two elements, the
                first one being the array of all the values, the second the array of all the
                corresponding ids, sorted by time
            >> aslist.convert("[(value,id)]") will return a list of tuples (value, id)
        """
		is_values = re.compile("values")
		is_ids = re.compile("ids")
		values = np.concatenate([st.signal for st in self.analog_signals.itervalues()])
		ids = np.concatenate([id * np.ones(len(st.signal), int) for id, st in self.analog_signals.iteritems()])
		if is_values.search(format):
			if is_ids.search(format):
				return eval(format)
			else:
				raise Exception("You must have a format with [values, ids] or [value, id]")
		is_values = re.compile("value")
		is_ids = re.compile("id")
		if is_values.search(format):
			if is_ids.search(format):
				result = []
				for id, time in zip(ids, values):
					result.append(eval(format))
			else:
				raise Exception("You must have a format with [values, ids] or [value, id]")
			return result

	def raw_data(self):
		"""
        Function to return a N by 2 array of all values and ids.

        Examples:
            >> spklist.raw_data()
            >> array([[  1.00000000e+00,   1.00000000e+00],
                      [  1.00000000e+00,   1.00000000e+00],
                      [  2.00000000e+00,   2.00000000e+00],
                         ...,
                      [  2.71530000e+03,   2.76210000e+03]])

        See also:
            convert()
        """
		data = np.array(self.convert("[values, ids]"))
		data = np.transpose(data)
		return data

	def as_array(self):
		"""
		Return the analog signal list as an array (len(id_list) x len(time_axis))
		"""
		if len(self.analog_signals[self.id_list()[0]].raw_data()) != len(self.time_axis()):
			time_axis = self.time_axis()[:-1] # in some cases, the time is rounded and causes and error
		else:
			time_axis = self.time_axis()

		a = np.zeros((len(self.id_list()), len(time_axis)))
		#print len(self.time_axis())
		for idx, n in enumerate(self.id_list()):
			a[idx, :] = self.analog_signals[n].raw_data()

		return a

	def save(self, user_file):
		"""
        Save the AnalogSignal in a text or binary file

            user_file - The user file that will have its own read/write method
                        By default, if s tring is provided, a StandardTextFile object
                        will be created. Nevertheless, you can also
                        provide a StandardPickleFile
        Examples:
            >> a.save("data.txt")
            >> a.save(StandardTextFile("data.txt"))
            >> a.save(StandardPickleFile("data.pck"))
        """
		as_loader = io.DataHandler(user_file, self)
		as_loader.save()

	def mean(self, axis=0):
		"""
        Return the mean AnalogSignal after having performed the average of all the signals
        present in the AnalogSignalList

        Examples:
            >> a.mean()

        See also:
            std

        :param axis: [0, 1], take the mean over time [0], or over neurons [1]
        """

		if axis == 0:
			result = np.zeros(int((self.t_stop - self.t_start) / self.dt), float)
			for id in self.id_list():
				result += self.analog_signals[id].signal
			return result/len(self)

		else:
			means = []
			for n in self.analog_signals:
				means.append(self.analog_signals[n].mean())

			return np.array(means)

	def std(self, axis=0):
		"""
        Return the standard deviation along time between all the AnalogSignals contained in
        the AnalogSignalList

        Examples:
            >> a.std()
               numpy.array([0.01, 0.2404, ...., 0.234, 0.234]

        See also:
            mean
        """
		result = np.zeros((len(self), int(round((self.t_stop - self.t_start) / self.dt))), float)
		for count, id in enumerate(self.id_list()):
			try:
				result[count, :] = self.analog_signals[id].signal
			except ValueError:
				print result[count, :].shape, self.analog_signals[id].signal.shape
				raise
		return np.std(result, axis)

	def event_triggered_average(self, eventdict, events_ids=None, analogsignal_ids=None, average=True,
	                            t_min=0, t_max=100, mode='same'):
		"""
        Returns the event triggered averages of the analog signals inside the list.
        The events can be a SpikeList object or a dict containing times.
        The average is performed on a time window t_spikes - tmin, t_spikes + tmax
        Can return either the averaged waveform (average = True), or an array of all the
        waveforms triggered by all the spikes.

        Inputs:
            events  - Can be a SpikeList object (and events will be the spikes) or just a dict
                      of times
            average - If True, return a single vector of the averaged waveform. If False,
                      return an array of all the waveforms.
            mode    - 'same': the average is only done on same ids --> return {'eventids':average};
                      'all': for all ids in the eventdict the average from all ananlog signals
                      is returned --> return {'eventids':{'analogsignal_ids':average}}
            t_min   - Time (>0) to average the signal before an event, in ms (default 0)
            t_max   - Time (>0) to average the signal after an event, in ms  (default 100)
            events_ids - when given only perform average over these ids
            analogsignal_ids = when given only perform average on these ids

        Examples
            >> vmlist.event_triggered_average(spikelist, average=False, t_min = 50, t_max = 150, mode = 'same')
            >> vmlist.event_triggered_average(spikelist, average=True, mode = 'all')
            >> vmlist.event_triggered_average({'1':[200,300,'3':[234,788]]}, average=False)
        """
		if isinstance(eventdict, SpikeList):
			eventdict = eventdict.spiketrains
		if events_ids is None:
			events_ids = eventdict.keys()
		if analogsignal_ids is None:
			analogsignal_ids = self.analog_signals.keys()

		x = np.ceil(np.sqrt(len(analogsignal_ids)))
		y = x
		results = {}

		first_done = False

		for id in events_ids:
			events = eventdict[id]
			if len(events) <= 0:
				continue
			if mode is 'same':
				if self.analog_signals.has_key(id) and id in analogsignal_ids:
					results[id] = self.analog_signals[id].event_triggered_average(events, average=average,
					                                                              t_min=t_min, t_max=t_max)
			elif mode is 'all':
				results[id] = {}
				for id_analog in analogsignal_ids:
					analog_signal = self.analog_signals[id_analog]
					results[id][id_analog] = analog_signal.event_triggered_average(events, average=average,
					                                                               t_min=t_min, t_max=t_max)
		return results

	def plot_random(self):
		"""

		:return:
		"""
		import matplotlib.pyplot as plt
		idx = np.random.permutation(self.id_list())[0]
		fig = plt.figure()
		plt.plot(self.time_axis(), self.analog_signals[int(idx)].raw_data())
		fig.suptitle('Channel {0}'.format(str(idx)))
		plt.show(block=False)

		return idx


