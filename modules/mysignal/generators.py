import math
import itertools
import numpy as np

from modules import analysis

########################################################################################################################
class Interval(object):
	"""
	Interval(start_times, end_times).
	Inputs:
		start_times - A list of the start times for all the sub intervals considered, in ms
		stop_times  - A list of the stop times for all the sub intervals considered, in ms

	Examples:
		>> itv = Interval([0,100,200,300],[50,150,250,350])
		>> itv.time_parameters()
			0, 350
	"""
	def __init__(self, start_times, end_times):
		"""
		Constructor of the Interval object.
		"""
		test = isinstance(start_times, int) or isinstance(start_times, float)
		assert test, "start_times should be a number !"
		test = isinstance(end_times, int) or isinstance(end_times, float)
		assert test, "end_times should be a number !"
		self.start_times = [start_times]
		self.end_times = [end_times]

	def intersect(self, itv):
		self.interval_data = self.interval_data & itv.interval_data

	def union(self, itv):
		self.interval_data = self.interval_data | itv.interval_data

	def __str__(self):
		return str(self.interval_data)

	def __len__(self):
		return np.shape(self.interval_data)[0]

	def __getslice__(self, i, j):
		"""
		Return a sub-list of the spike_times vector of the SpikeTrain
		"""
		return self.interval_data[i:j]

	def time_parameters(self):
		"""
		Return the time parameters of the SpikeTrain (t_start, t_stop)
		"""
		bounds = self.interval_data.extrema
		return (bounds[0][0], bounds[-1][0])

	def t_start(self):
		return self.start_times[0]

	def t_stop(self):
		return self.end_times[0]

	def copy(self):
		"""
		Return a copy of the SpikeTrain object
		"""
		return Interval(self.start_times, self.end_times)

	def total_duration(self):
		"""
		Return the total duration of the interval
		"""
		tot_duration = 0
		for i in self.interval_data:
			tot_duration += i[1] - i[0]
		return tot_duration

	def slice_times(self, times):
		spikes_selector = np.zeros(len(times), dtype=np.bool)
		spikes_selector = (times >= self.t_start()) & (times <= self.t_stop())
		return np.extract(spikes_selector, times)


############################################################################################
class PairsGenerator(object):
	"""
    PairsGenerator(SpikeList, SpikeList, no_silent)
    This class defines the concept of PairsGenerator, that will be used by all
    the functions using pairs of cells. Functions get_pairs() will then be used
    to obtain pairs from the generator.

    Inputs:
        spk1      - First SpikeList object to take cells from
        spk2      - Second SpikeList object to take cells from
        no_silent - Boolean to say if only non silent cells should
                    be considered. False by default

    Examples:
        >> p = PairsGenerator(spk1, spk1, True)
        >> p.get_pairs(100)

    See also AutoPairs, RandomPairs, CustomPairs, DistantDependentPairs
	"""
	def __init__(self, spk1, spk2, no_silent=False):
		self.spk1 = spk1
		self.spk2 = spk2
		self.no_silent = no_silent
		self._get_id_lists()

	def _get_id_lists(self):
		self.ids_1 = set(self.spk1.id_list)
		self.ids_2 = set(self.spk2.id_list)
		if self.no_silent:
			n1 = set(self.spk1.select_ids("len(cell.spike_times) == 0"))
			n2 = set(self.spk2.select_ids("len(cell.spike_times) == 0"))
			self.ids_1 -= n1
			self.ids_2 -= n2

	def get_pairs(self, nb_pairs):
		"""
        Function to obtain a certain number of cells from the generator

        Inputs:
            nb_pairs - int to specify the number of pairs desired

        Examples:
            >> res = p.get_pairs(100)
        """
		# Question: where is this defined?
		return _abstract_method(self)


########################################################################################################################
class AutoPairs(PairsGenerator):
	"""
    AutoPairs(SpikeList, SpikeList, no_silent). Inherits from PairsGenerator.
    Generator that will return pairs of the same elements (contained in the
    two SpikeList) selected twice.

    Inputs:
        spk1      - First SpikeList object to take cells from
        spk2      - Second SpikeList object to take cells from
        no_silent - Boolean to say if only non silent cells should
                    be considered. False by default

    Examples:
        >> p = AutoPairs(spk1, spk1, True)
        >> p.get_pairs(4)
            [[1,1],[2,2],[4,4],[5,5]]

    See also RandomPairs, CustomPairs, DistantDependentPairs
	"""
	def __init__(self, spk1, spk2, no_silent=False):
		PairsGenerator.__init__(self, spk1, spk2, no_silent)

	def get_pairs(self, nb_pairs):
		cells = np.random.permutation(list(self.ids_1.intersection(self.ids_2)))
		N = len(cells)
		if nb_pairs > N:
			if not self.no_silent:
				print "Only %d distinct pairs can be extracted. Turn no_silent to True." % N
		try:
			pairs = np.zeros((N, 2), int)
			pairs[:, 0] = cells[0:N]
			pairs[:, 1] = cells[0:N]
			return pairs
		except Exception:
			return np.array([[],[]])


########################################################################################################################
class RandomPairs(PairsGenerator):
	"""
    RandomPairs(SpikeList, SpikeList, no_silent, no_auto). Inherits from PairsGenerator.
    Generator that will return random pairs of elements.

    Inputs:
        spk1      - First SpikeList object to take cells from
        spk2      - Second SpikeList object to take cells from
        no_silent - Boolean to say if only non silent cells should
                    be considered. False by default
        no_auto   - Boolean to say if pairs with the same element (id,id) should
                    be removed. True by default, i.e those pairs are discarded

    Examples:
        >> p = RandomPairs(spk1, spk1, True, False)
        >> p.get_pairs(4)
            [[1,3],[2,5],[1,4],[5,5]]
        >> p = RandomPairs(spk1, spk1, True, True)
        >> p.get_pairs(3)
            [[1,3],[2,5],[1,4]]


    See also RandomPairs, CustomPairs, DistantDependentPairs
	"""
	def __init__(self, spk1, spk2, no_silent=False, no_auto=True):
		PairsGenerator.__init__(self, spk1, spk2, no_silent)
		self.no_auto = no_auto

	def get_pairs(self, nb_pairs):
		cells1 = np.array(list(self.ids_1), int)
		cells2 = np.array(list(self.ids_2), int)
		pairs = np.zeros((0, 2), int)
		N1 = len(cells1)
		N2 = len(cells2)
		T = min(N1, N2)
		while len(pairs) < nb_pairs:
			N = min(nb_pairs - len(pairs), T)
			tmp_pairs = np.zeros((N, 2), int)
			tmp_pairs[:, 0] = cells1[np.floor(np.random.uniform(0, N1, N)).astype(int)]
			tmp_pairs[:, 1] = cells2[np.floor(np.random.uniform(0, N2, N)).astype(int)]
			if self.no_auto:
				idx = np.where(tmp_pairs[:, 0] == tmp_pairs[:, 1])[0]
				pairs = np.concatenate((pairs, np.delete(tmp_pairs, idx, axis=0)))
			else:
				pairs = np.concatenate((pairs, tmp_pairs))
		return pairs


########################################################################################################################
class DistantDependentPairs(PairsGenerator):
	"""
    DistantDependentPairs(SpikeList, SpikeList, no_silent, no_auto). Inherits from PairsGenerator.
    Generator that will return pairs of elements according to the distances between the cells. The
    dimensions attribute of the SpikeList should be not empty.

    Inputs:
        spk1      - First SpikeList object to take cells from
        spk2      - Second SpikeList object to take cells from
        no_silent - Boolean to say if only non silent cells should
                    be considered. False by default
        no_auto   - Boolean to say if pairs with the same element (id,id) should
                    be remove. True by default, i.e those pairs are discarded
        length    - the lenght (in mm) covered by the extend of spk1 and spk2. Currently, spk1
                    and spk2 should cover the same surface. Default is spk1.length
        d_min     - the minimal distance between cells
        d_max     - the maximal distance between cells

    Examples:
        >> p = DistantDependentPairs(spk1, spk1, True, False)
        >> p.get_pairs(4, d_min=0, d_max = 50)
            [[1,3],[2,5],[1,4],[5,5]]
        >> p = DistantDependentPairs(spk1, spk1, True, True, lenght=1)
        >> p.get_pairs(3, d_min=0.25, d_max=0.35)
            [[1,3],[2,5],[1,4]]


    See also RandomPairs, CustomPairs, AutoPairs
	"""
	def __init__(self, spk1, spk2, no_silent=False, no_auto=True, lenght=1., d_min=0, d_max=1e6):
		PairsGenerator.__init__(self, spk1, spk2, no_silent)
		self.lenght = lenght
		self.no_auto = no_auto
		self.d_min = d_min
		self.d_max = d_max

	def set_bounds(self, d_min, d_max):
		self.d_min = d_min
		self.d_max = d_max

	def get_pairs(self, nb_pairs):
		"""
		Function to obtain a certain number of cells from the generator

        Inputs:
            nb_pairs - int to specify the number of pairs desired

        The length parameter of the DistantDependentPairs should be defined correctly. It is the extent of the grid.

        Examples:
            >> res = p.get_pairs(100, 0.3, 0.5)
		"""
		cells1 = np.array(list(self.ids_1), int)
		cells2 = np.array(list(self.ids_2), int)
		pairs = np.zeros((0, 2), int)
		N1 = len(cells1)
		N2 = len(cells2)
		T = min(N1, N2)
		while len(pairs) < nb_pairs:
			N = min(nb_pairs - len(pairs), T)
			cell1 = cells1[np.floor(np.random.uniform(0, N1, N)).astype(int)]
			cell2 = cells2[np.floor(np.random.uniform(0, N1, N)).astype(int)]
			pos_cell1 = np.array(self.spk1.id2position(cell1)) * self.lenght / self.spk1.dimensions[0]
			pos_cell2 = np.array(self.spk2.id2position(cell2)) * self.lenght / self.spk2.dimensions[0]
			dist = analysis.distance(pos_cell1, pos_cell2, self.lenght)
			idx = np.where((dist >= self.d_min) & (dist < self.d_max))[0]
			N = len(idx)
			if N > 0:
				tmp_pairs = np.zeros((N, 2), int)
				tmp_pairs[:, 0] = cell1[idx]
				tmp_pairs[:, 1] = cell2[idx]
				if self.no_auto:
					idx = np.where(tmp_pairs[:, 0] == tmp_pairs[:, 1])[0]
					pairs = np.concatenate((pairs, np.delete(tmp_pairs, idx, axis=0)))
				else:
					pairs = np.concatenate((pairs, tmp_pairs))
		return pairs


########################################################################################################################
class CustomPairs(PairsGenerator):
	"""
    CustomPairs(SpikeList, SpikeList, pairs). Inherits from PairsGenerator.
    Generator that will return custom pairs of elements.

    Inputs:
        spk1  - First SpikeList object to take cells from
        spk2  - Second SpikeList object to take cells from
        pairs - A list of tuple that will be the pairs returned
                when get_pairs() function will be used.

    Examples:
        >> p = CustomPairs(spk1, spk1, [(i,i) for i in xrange(100)])
        >> p.get_pairs(4)
            [[1,1],[2,2],[3,3],[4,4]]

    See also RandomPairs, CustomPairs, DistantDependentPairs, AutoPairs
	"""
	def __init__(self, spk1, spk2, pairs=[[], []]):
		PairsGenerator.__init__(self, spk1, spk2)
		self.pairs = np.array(pairs)

	def get_pairs(self, nb_pairs):
		if nb_pairs > len(self.pairs):
			print "Trying to select too much pairs..."
		return self.pairs[0:nb_pairs]

########################################################################################################################
# Change point detection library!!https://bitbucket.org/aihara/changefinder


def LevinsonDurbin(r, lpcOrder):
	"""
	from http://aidiary.hatenablog.com/entry/20120415/1334458954
	"""
	a = np.zeros(lpcOrder + 1, dtype=np.float64)
	e = np.zeros(lpcOrder + 1, dtype=np.float64)
	a[0] = 1.0
	a[1] = - r[1] / r[0]
	e[1] = r[0] + r[1] * a[1]
	lam = - r[1] / r[0]
	for k in range(1, lpcOrder):
		lam = 0.0
		for j in range(k + 1):
			lam -= a[j] * r[k + 1 - j]
		lam /= e[k]

		U = [1]
		U.extend([a[i] for i in range(1, k + 1)])
		U.append(0)

		V = [0]
		V.extend([a[i] for i in range(k, 0, -1)])
		V.append(1)

		a = np.array(U) + lam * np.array(V)
		e[k + 1] = e[k] * (1.0 - lam * lam)

	return a, e[-1]


class _SDAR_1Dim(object):
	def __init__(self, r, order):
		self._r = r
		self._mu = np.random.random()
		self._sigma = np.random.random()
		self._order = order
		self._c = np.zeros(self._order+1)

	def update(self,x,term):
		assert len(term) >= self._order, "term must be order or more"
		term = np.array(term)
		self._mu = (1 - self._r) * self._mu + self._r * x
		for i in range(1,self._order):
			self._c[i] = (1-self._r)*self._c[i]+self._r * (x-self._mu) * (term[-i]-self._mu)
		self._c[0] = (1-self._r)*self._c[0]+self._r * (x-self._mu)*(x-self._mu)
		what, e = LevinsonDurbin(self._c,self._order)
		xhat = np.dot(-what[1:],(term[::-1]  - self._mu))+self._mu
		self._sigma = (1-self._r)*self._sigma + self._r * (x-xhat) * (x-xhat)
		return -math.log(math.exp(-0.5 *(x-xhat)**2/self._sigma)/((2 * math.pi)**0.5 * self._sigma**0.5)),xhat


class _ChangeFinderAbstract(object):
	def _add_one(self,one,ts,size):
		ts.append(one)
		if len(ts)==size+1:
			ts.pop(0)

	def _smoothing(self,ts):
		return sum(ts)/float(len(ts))


class ChangeFinder(_ChangeFinderAbstract):
	def __init__(self, r=0.5, order=1, smooth=7):
		assert order > 0, "order must be 1 or more."
		assert smooth > 2, "term must be 3 or more."
		self._smooth = smooth
		self._smooth2 = int(round(self._smooth/2.0))
		self._order = order
		self._r = r
		self._ts = []
		self._first_scores = []
		self._smoothed_scores = []
		self._second_scores = []
		self._sdar_first = _SDAR_1Dim(r,self._order)
		self._sdar_second = _SDAR_1Dim(r,self._order)

	def update(self,x):
		score = 0
		predict = x
		predict2 = 0
		if len(self._ts) == self._order:
			score, predict = self._sdar_first.update(x, self._ts)
			self._add_one(score,self._first_scores,self._smooth)
		self._add_one(x,self._ts,self._order)
		second_target = None
		if len(self._first_scores) == self._smooth:
			second_target = self._smoothing(self._first_scores)
		if second_target and len(self._smoothed_scores) == self._order:
			score, predict2 = self._sdar_second.update(second_target, self._smoothed_scores)
			self._add_one(score, self._second_scores, self._smooth2)
		if second_target:
			self._add_one(second_target,self._smoothed_scores, self._order)
		if len(self._second_scores) == self._smooth2:
			return self._smoothing(self._second_scores), predict
		else:
			return 0.0, predict


def smooth(x, window_len=11, window='hanning'):
	"""
	Smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """
	if x.ndim != 1:
		raise ValueError, "smooth only accepts 1 dimension arrays."
	if x.size < window_len:
		raise ValueError, "Input vector needs to be bigger than window size."

	if window_len<3:
		return x

	if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
		raise ValueError, "Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

	s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
	if window == 'flat': #moving average
		w = np.ones(window_len, 'd')
	else:
		w = eval('np.'+window+'(window_len)')
	y = np.convolve(w/w.sum(), s, mode='valid')
	return y[(window_len/2-1):-(window_len/2)]


def moving_window(seq, window_len=10):
	"""
	Generator for moving window intervals
	:return:
	"""
	it = iter(seq)
	result = tuple(itertools.islice(it, window_len))
	if len(result) == window_len:
		yield result
	for elem in it:
		result = result[1:] + (elem,)
		yield result
