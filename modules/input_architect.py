"""
========================================================================================================================
Input Architect Module
========================================================================================================================
(this documentation is incomplete)

Classes:
--------
Grammar             - wrapper for the generation of sequences of elements according to specific rules
StimulusSet         - wrapper to hold and manipulate all the data pertaining to the input stimuli,
					  labels, and corresponding time series
StochasticGenerator - Stochastic process generator
InputSignal         - generate and store AnalogSignal object referring to the structured input signal u(t)
InputSignalSet      - class to hold and manipulate complex sets of input signals
InputNoise          - generate and store AnalogSignal object referring to the noise to add to the input signal u(t)
Encoder             - convert continuous signal into SpikeList objects, or create a population of spiking neurons
					  from a given parameter set

Functions:
----------
pad_array					- pads an array with zeros along the time dimension
generate_template			- generates a spatio-temporal spike template
load_preset_grammar 		- build a grammar object based on parameters stored in a file
stimulus_sequence_to_binary	-
merge_signals       		- sums 2 AnalogSignals
make_simple_kernel			- simple way to create a smoothing kernel for 1D convolution

========================================================================================================================
Copyright (C) 2019  Renato Duarte, Barna Zajzon

Neural Mircocircuit Simulation and Analysis Toolkit is free software;
you can redistribute it and/or modify it under the terms of the
GNU General Public License as published by the Free Software Foundation;
either version 2 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

The GNU General Public License does not permit this software to be
redistributed in proprietary programs.
"""
# other imports
import cPickle as pickle
import numpy as np
import bisect
import itertools
import time

from scipy.sparse import coo_matrix, lil_matrix
from scipy.signal import fftconvolve
import copy
import decimal

# nmsat
import parameters
import signals
import io
import net_architect
import visualization

# nest
import nest
import nest.topology as tp

logger = io.get_logger(__name__)


def pad_array(input_array, add=10):
    """
    Pads an array with zeros along the time dimension

    :param input_array:
    :param add:
    :return:
    """
    new_shape 	= (input_array.shape[0], input_array.shape[1]+add)
    new_size 	= (new_shape[0])*(new_shape[1])
    zero_array 	= np.zeros(new_size).reshape(new_shape)
    zero_array[:input_array.shape[0], :input_array.shape[1]] = input_array
    return zero_array


def stimulus_sequence_to_binary(seq):
    """
    Convert a stimulus sequence to a binary time series
    """
    elements = list(np.unique(seq))
    dims = len(elements)

    tmp = lil_matrix(np.zeros((dims, len(seq))))

    if isinstance(seq[0], basestring) or isinstance(seq[0], int):
        for i, x in enumerate(elements):
            idx = np.where(seq == x)
            tmp[i, idx] = 1.
    elif isinstance(seq[0], list):
        for ii, xx in enumerate(seq):
            for i, x in enumerate(elements):
                if x in xx:
                    tmp[i, ii] = 1.
    return coo_matrix(tmp)


def merge_signals(signal1, signal2, operation=np.add):
    """
    Combine 2 Analog signals
    :param signal1: AnalogSignal object 1
    :param signal2: AnalogSignal object 2
    :param operation: How to combine the signals (numpy function, e.g., np.add, np.multiply, np.append)
    :return: AnalogSignal
    """
    assert isinstance(signal1, signals.AnalogSignal) and isinstance(signal2, signals.AnalogSignal), "Signals must be " \
                                                                                                    "AnalogSignal"
    assert signal1.dt == signal2.dt, "Inconsistent signal resolution"
    assert signal1.t_start == signal2.t_start, "Inconsistent t_start"
    assert signal1.t_stop == signal2.t_stop, "Inconsistent t_stop"

    signal = operation(signal1.signal, signal2.signal)

    return signals.AnalogSignal(signal, dt=signal1.dt, t_start=signal1.t_start, t_stop=signal1.t_stop)


def merge_signal_lists(sl1, sl2, operation=np.add):
    """
    Combine 2 AnalogSignalList objects
    :param sl1:
    :param sl2:
    :param operation:
    :return:
    """
    assert isinstance(sl1, signals.AnalogSignalList) and isinstance(sl2, signals.AnalogSignalList), "Signals must be AnalogSignalList"
    assert sl1.dt == sl2.dt, "Inconsistent signal resolution"
    assert sl1.t_start == sl2.t_start, "Inconsistent t_start"
    assert sl1.t_stop == sl2.t_stop, "Inconsistent t_stop"

    new_values = operation(sl1.raw_data()[:, 0], sl2.raw_data()[:, 0])
    ids = sl1.raw_data()[:, 1]
    time_data = sl1.time_axis()
    tmp = [(ids[n], new_values[n]) for n in range(len(new_values))]

    return signals.AnalogSignalList(tmp, np.unique(ids).tolist(), times=time_data)


def generate_template(n_neurons, rate, duration, resolution=0.01, rng=None, store=False):
    """
    Generates a spatio-temporal spike template
    :param n_neurons: Number of neurons that compose the pattern
    :param rate: spike rate in the template
    :param duration: [ms] total duration of template
    :param resolution:
    :param rng: random number generator state object (optional). Either None or a numpy.random.RandomState object,
        or an object with the same interface
    :param store: save the template in the provided path
    :return: SpikeList object
    """
    gen 	= StochasticGenerator(rng=rng)
    times 	= []
    ids 	= []
    rounding_precision = signals.determine_decimal_digits(resolution)
    for n in range(n_neurons):
        spk_times = gen.poisson_generator(rate, t_start=resolution, t_stop=duration, array=True)
        times.append(list(spk_times))
        ids.append(list(n * np.ones_like(times[-1])))
    ids = list(signals.iterate_obj_list(ids))
    tmp = [(ids[idx], round(n, rounding_precision)) for idx, n in enumerate(list(signals.iterate_obj_list(times)))]

    sl = signals.SpikeList(tmp, list(np.unique(ids)), t_start=resolution, t_stop=duration)
    sl.round_times(resolution)

    if store:
        sl.save(store)

    return sl


def make_simple_kernel(shape, width=3, height=1., resolution=1., normalize=False, **kwargs):
    """
    Simplest way to create a smoothing kernel for 1D convolution
    :param shape: {'box', 'exp', 'alpha', 'double_exp', 'gauss'}
    :param width: kernel width
    :param height: peak amplitude of the kernel
    :param resolution: time step
    :param normalize: [bool]
    :return: kernel k
    """
    # TODO load external kernel...
    x = np.arange(0., (width / resolution) + resolution, 1.) #resolution)

    if shape == 'box':
        k = np.ones_like(x) * height

    elif shape == 'exp':
        assert 'tau' in kwargs, "for exponential kernel, please specify tau"
        tau = kwargs['tau']
        k = np.exp(-x / tau) * height

    elif shape == 'double_exp':
        assert ('tau_1' in kwargs), "for double exponential kernel, please specify tau_1"
        assert ('tau_2' in kwargs), "for double exponential kernel, please specify tau_2"

        tau_1 = kwargs['tau_1']
        tau_2 = kwargs['tau_2']
        tmp_k = (-np.exp(-x / tau_1) + np.exp(-x / tau_2))
        k = tmp_k * (height / np.max(tmp_k))

    elif shape == 'alpha':
        assert ('tau' in kwargs), "for alpha kernel, please specify tau"

        tau = kwargs['tau']
        tmp_k = ((x / tau) * np.exp(-x / tau))
        k = tmp_k * (height / np.max(tmp_k))

    elif shape == 'gauss':
        assert ('mu' in kwargs), "for Gaussian kernel, please specify mu"
        assert ('sigma' in kwargs), "for Gaussian kernel, please specify sigma"

        sigma = kwargs['sigma']
        mu = kwargs['mu']
        tmp_k = (1. / (sigma * np.sqrt(2. * np.pi))) * np.exp(- ((x - mu) ** 2. / (2. * (sigma ** 2.))))
        k = tmp_k * (height / np.max(tmp_k))

    elif shape == 'tri':
        halfwidth = width / 2
        trileft = np.arange(1, halfwidth + 2)
        triright = np.arange(halfwidth, 0, -1)  # odd number of bins
        k = np.append(trileft, triright)
        k += height

    elif shape == 'sin':
        k = np.sin(2 * np.pi * x / width * kwargs['frequency'] + kwargs['phase_shift']) * height
        k += kwargs['mean_amplitude']
    else:
        logger.warning("Kernel not implemented, please choose {'box', 'exp', 'alpha', 'double_exp', 'gauss', 'tri'}")
        k = 0
    if normalize:
        k /= k.sum()

    return k


########################################################################################################################
class StochasticGenerator:
    """
    Stochastic process generator
    ============================
    (adapted from NeuroTools)

    Generate stochastic processes of various types and return them as SpikeTrain or AnalogSignal objects.

    Implemented types:
    ------------------
    a) Spiking Point Process - poisson_generator, inh_poisson_generator, gamma_generator, !!inh_gamma_generator!!,
    inh_adaptingmarkov_generator, inh_2Dadaptingmarkov_generator

    b) Continuous Time Process - OU_generator, GWN_generator, continuous_rv_generator (any other distribution)
    """

    def __init__(self, rng=None, seed=None):
        """
        :param rng: random number generator state object (optional). Either None or a numpy.random.RandomState object,
        or an object with the same interface
        :param seed: rng seed

        If rng is not None, the provided rng will be used to generate random numbers, otherwise StGen will create
        its own rng. If a seed is provided, it is passed to rng.seed(seed)
        """
        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng

        if seed is not None:
            self.rng.seed(seed)

    def seed(self, seed):
        """
        seed the gsl rng
        """
        self.rng.seed(seed)

    def poisson_generator(self, rate, t_start=0.0, t_stop=1000.0, array=False, debug=False):
        """
        Returns a SpikeTrain whose spikes are a realization of a Poisson process
        with the given rate (Hz) and stopping time t_stop (milliseconds).

        Note: t_start is always 0.0, thus all realizations are as if
        they spiked at t=0.0, though this spike is not included in the SpikeList.

        :param rate: the rate of the discharge (in Hz)
        :param t_start: the beginning of the SpikeTrain (in ms)
        :param t_stop: the end of the SpikeTrain (in ms)
        :param array: if True, a numpy array of sorted spikes is returned,
                      rather than a SpikeTrain object.

        :return spikes: SpikeTrain object

        Examples:
        --------
            >> gen.poisson_generator(50, 0, 1000)
            >> gen.poisson_generator(20, 5000, 10000, array=True)
        """

        n = (t_stop - t_start) / 1000.0 * rate
        number = np.ceil(n + 3 * np.sqrt(n))
        if number < 100:
            number = min(5 + np.ceil(2 * n), 100)

        if number > 0:
            isi = self.rng.exponential(1.0 / rate, int(number)) * 1000.0
            if number > 1:
                spikes = np.add.accumulate(isi)
            else:
                spikes = isi
        else:
            spikes = np.array([])

        spikes += t_start
        i = np.searchsorted(spikes, t_stop)

        extra_spikes = []
        if i == len(spikes):
            # ISI buf overrun
            t_last = spikes[-1] + self.rng.exponential(1.0 / rate, 1)[0] * 1000.0

            while t_last < t_stop:
                extra_spikes.append(t_last)
                t_last += self.rng.exponential(1.0 / rate, 1)[0] * 1000.0

            spikes = np.concatenate((spikes, extra_spikes))

            if debug:
                logger.debug("ISI buf overrun handled. len(spikes)=%d, len(extra_spikes)=%d" % (len(spikes),
                                                                                                len(extra_spikes)))

        else:
            spikes = np.resize(spikes, (i,))

        if not array:
            spikes = signals.SpikeTrain(spikes, t_start=t_start, t_stop=t_stop)

        if debug:
            return spikes, extra_spikes
        else:
            return spikes

    def gamma_generator(self, a, b, t_start=0.0, t_stop=1000.0, array=False, debug=False):
        """
        Returns a SpikeTrain whose spikes are a realization of a gamma process
        with the given shape a, b and stopping time t_stop (milliseconds).
        (average rate will be a*b)

        Note: t_start is always 0.0, thus all realizations are as if
        they spiked at t=0.0, though this spike is not included in the SpikeList.

        :param a,b: the parameters of the gamma process
        :param t_start: the beginning of the SpikeTrain (in ms)
        :param t_stop: the end of the SpikeTrain (in ms)
        :param array: if True, a numpy array of sorted spikes is returned, rather than a SpikeTrain object.

        Examples:
        --------
            >> gen.gamma_generator(10, 1/10., 0, 1000)
            >> gen.gamma_generator(20, 1/5., 5000, 10000, array=True)
        """
        n = (t_stop - t_start) / 1000.0 * (a * b)
        number = np.ceil(n + 3 * np.sqrt(n))
        if number < 100:
            number = min(5 + np.ceil(2 * n), 100)

        if number > 0:
            isi = self.rng.gamma(a, b, number) * 1000.0
            if number > 1:
                spikes = np.add.accumulate(isi)
            else:
                spikes = isi
        else:
            spikes = np.array([])

        spikes += t_start
        i = np.searchsorted(spikes, t_stop)

        extra_spikes = []
        if i == len(spikes):
            # ISI buf overrun
            t_last = spikes[-1] + self.rng.gamma(a, b, 1)[0] * 1000.0

            while t_last < t_stop:
                extra_spikes.append(t_last)
                t_last += self.rng.gamma(a, b, 1)[0] * 1000.0

            spikes = np.concatenate((spikes, extra_spikes))

            if debug:
                logger.debug("ISI buf overrun handled. len(spikes)=%d, len(extra_spikes)=%d" % (len(spikes),
                                                                                                len(extra_spikes)))
        else:
            spikes = np.resize(spikes, (i,))

        if not array:
            spikes = signals.SpikeTrain(spikes, t_start=t_start, t_stop=t_stop)

        if debug:
            return spikes, extra_spikes
        else:
            return spikes

    def OU_generator(self, dt, tau, sigma, y0, t_start=0.0, t_stop=1000.0, rectify=False, array=False, time_it=False):
        """
        Generates an Ornstein-Uhlenbeck process using the forward euler method. The function returns
        an AnalogSignal object.

        :param dt: the time resolution in milliseconds of th signal
        :param tau: the correlation time in milliseconds
        :param sigma: std dev of the process
        :param y0: initial value of the process, at t_start
        :param t_start: start time in milliseconds
        :param t_stop: end time in milliseconds
        :param array: if True, the functions returns the tuple (y,t)
                      where y and t are the OU signal and the time bins, respectively,
                      and are both numpy arrays.
        :return AnalogSignal
        """
        if time_it:
            t1 = time.time()

        t = np.arange(t_start, t_stop, dt)
        N = len(t)
        y = np.zeros(N, float)
        y[0] = y0
        fac = dt / tau
        gauss = fac * y0 + np.sqrt(2 * fac) * sigma * self.rng.standard_normal(N - 1)
        mfac = 1 - fac

        # python loop... bad+slow!
        for i in xrange(1, N):
            idx = i - 1
            y[i] = y[idx] * mfac + gauss[idx]

        if time_it:
            logger.info(time.time() - t1)
        if rectify:
            y[y < 0] = 0.

        if array:
            return (y, t)
        else:
            return signals.AnalogSignal(y, dt, t_start, t_stop)

    def GWN_generator(self, amplitude=1., mean=0., std=1., t_start=0.0, t_stop=1000.0, dt=1.0, rectify=True,
                      array=False):
        """
        Generates a Gaussian White Noise process. The function returns an AnalogSignal object.

        :param amplitude: maximum amplitude of the noise signal
        """

        t = np.arange(t_start, t_stop, dt)
        wn = amplitude * np.random.normal(loc=mean, scale=std, size=len(t))

        if rectify:
            wn[wn < 0] = 0.

        if array:
            return (wn, t)
        else:
            return signals.AnalogSignal(wn, dt, t_start, t_stop)

    def continuous_rv_generator(self, function, amplitude=1., t_start=0.0, t_stop=1000.0, dt=1.0, rectify=True,
                                array=False, **kwargs):
        """
        Generates a realization of a continuous noise process by drawing iid values from the distribution specified by
        function and parameterized by **kwargs
        :param function: distribution function (e.g. np.random.poisson)
        Note: **kwargs must correspond to the function parameters
        """

        t = np.arange(t_start, t_stop, dt)
        if isinstance(function, basestring):
            function = eval(function)
        s = function(size=len(t), **kwargs)
        s *= amplitude

        if rectify:
            s[s < 0] = 0.
        if array:
            return (s, t)
        else:
            return signals.AnalogSignal(s, dt, t_start, t_stop)


########################################################################################################################
class Grammar:
    """
    GRAMMAR:
    Class representing the set of grammatical rules underlying the
    temporal sequences (strings) to be produced.
    The rules are specified by the allowed transitions of a directed
    graph, along with corresponding transition probabilities. The
    sequences are generated by traversing the graph.
    """

    def __init__(self, grammar_pars_set, pre_set=None, pre_set_full_path=None):
        """
        :param grammar_pars_set: ParameterSet or dictionary object, should contain
        either the name of the pre_set grammar or the following fields:
        - states: a list containing the states of grammar, i.e., the
        nodes of the directed graph. For simplicity and robustness of
        implementation, the nodes should correspond to the individual
        (unique) symbols that constitute the language
        - alphabet: a list containing the unique symbols that ought to
        be represented. In many cases, these symbols are different from
        the states, given that the same symbol may correspond to
        several different nodes, in which case, the different states
        for the same symbol are numbered (see examples)
        - transition_pr: list of tuples with the structure (source_state,
        target_state, transition probability). e.g. [('a','b',0.1),
        ('a','c',0.3)]
        - start_symbols: a list containing the possible start symbols
        - end_symbols: a list containing the terminal symbols
        - pre_set: if specified will use predefined grammars, whose parameters are already specified
        """
        if isinstance(grammar_pars_set, dict):
            grammar_pars_set = parameters.ParameterSet(grammar_pars_set)
        if not (isinstance(grammar_pars_set, parameters.ParameterSet)):
            raise TypeError("grammar_pars_set should be ParameterSet object or dictionary")

        # if pre_set is not None:
        # 	assert pre_set_full_path is not None, "Path to parameters file must be provided"
        # 	self.name = pre_set
        # 	grammar_pars_set = ParameterSet(load_preset_grammar(pre_set_full_path, pre_set))
        # 	print "\n Loading {0} grammar".format(self.name)

        self.name = pre_set
        self.states = list(np.sort(grammar_pars_set.states))
        self.alphabet = list(np.sort(grammar_pars_set.alphabet))
        self.transition_pr = grammar_pars_set.transitions
        self.start_symbols = grammar_pars_set.start_states
        self.end_symbols = grammar_pars_set.end_states
        # self.n_gram, self.topological_entropy = self.compute_topological_entropy()
        self.transitionTable = []

    def print_rules(self):
        """
        Displays all the relevant information.
        """
        logger.info('***************************************************************************')
        if self.name is not None:
            logger.info(('Generative mechanism abides to %s rules' % self.name))
        logger.info('Unique states: {0}'.format(self.states))
        logger.info('Alphabet: {0}'.format(self.aplhabet))
        logger.info('Transition table: ')
        self.print_transitiontable(self.generate_transitiontable(), self.states)
    # print '\nGrammar complexity (n, TE): {0}{1}'.format(self.n_gram, self.topological_entropy)

    def generate_transitiontable(self):
        """
        Creates a look-up table with all allowed transitions and their probabilities
        """
        t = self.transition_pr

        table = np.zeros((len(self.states), len(self.states)))

        for i, ii in enumerate(self.states):
            for j, jj in enumerate(self.states):
                tmp = [v[2] for v in t if (v[0] == jj and v[1] == ii)]
                if tmp:
                    table[i, j] = tmp[0]

        self.transitionTable = table
        return table

    @staticmethod
    def print_transitiontable(tr_table, un_symb):
        """
        Print the full transition matrix in a visually understandable table
        :param tr_table: numpy array
        :param un_symb: list of unique symbols
        """

        logger.info("----------------------------------------------------------")
        header = "|    "
        for x in un_symb:
            header += "| %s " % x
        header += "|"
        logger.info(header)
        for i in range(tr_table.shape[0]):
            new_line = "| %s |" % un_symb[i]
            for j in range(tr_table.shape[1]):
                new_line += " %s " % str(np.round(tr_table[i, j], 1))
            logger.info(new_line)

    def validate(self, debug=False):
        """
        Verify that all the start and end states are members of the
        state set and if the alphabet is different from the states
        """
        assert set(self.start_symbols).issubset(set(self.states)), 'start_symbols not in states'
        assert set(self.end_symbols).issubset(set(self.states)), 'end_symbols not in states'

        if not set(self.alphabet).issubset(set(self.states)):
            TestVar = set(self.alphabet).difference(set(self.states))
            if debug:
                logger.debug(TestVar)
            return TestVar
        else:
            return True

    def generate_string(self):
        """
        Generate a single grammatical string by traversing the grammar
        :return: string as a list of symbols...
        """
        string = []

        string.append(self.start_symbols[np.random.random_integers(0, len(self.start_symbols) - 1)])
        current_state = string[-1]

        while current_state not in self.end_symbols:
            allowed_transitions = [x for i, x in enumerate(self.transition_pr) if x[0] == current_state]

            assert (allowed_transitions >= 0), 'No allowed transitions from node {0}'.format(current_state)
            if len(allowed_transitions) == 1:
                string.append(allowed_transitions[0][1])
            else:
                Pr = [n[2] for i, n in enumerate(allowed_transitions)]
                cumPr = np.cumsum(Pr)
                idx = bisect.bisect_right(cumPr, np.random.random())
                string.append(allowed_transitions[idx][1])
            current_state = string[-1]

        return string

    def generate_string_set(self, n_strings, str_range=[], debug=False):
        """
        (Generator Function)
        Generates a grammatical string set containing nStrings strings...
        :param n_strings: Total number of strings to generate
        :param str_range: [list] string length, should be specified as interval [min_len, max_len]
        :returns string_set: iterator of generated strings
        """

        if self.name is not None and debug:
            logger.debug('Generating {0} strings, according to {1} rules...'.format(n_strings, self.name))

        string_set = []

        while len(string_set) < n_strings:
            str = self.generate_string()
            if str_range:
                while not (str_range[0] < len(str) < str_range[1]):
                    str = self.generate_string()
                string_set.append(str)
            else:
                string_set.append(str)

        if debug:
            logger.debug('Example String: {0}'.format(''.join(
                string_set[np.random.random_integers(0, len(self.start_symbols) - 1)])))

        return string_set

    @staticmethod
    def concatenate_stringset(string_set, separator=''):
        """
        Concatenates a string set (list of strings) into a list of symbols, placing the
        separator symbol between strings
        :param string_set: list of strings
        :param separator: string symbol separating different strings
        """
        str_set = copy.deepcopy(string_set)
        [n.insert(0, separator) for n in list(str_set)]
        symbol_seq = np.concatenate(list(str_set)).tolist()
        return symbol_seq

    def correct_symbolseq(self, symbol_seq):
        """
        Corrects the symbol sequence. In sequences where the same symbol
        corresponds to different states (i.e. alphabet != states), the
        various states for the same symbol are specified by a numeric index (see
        pre_set grammars). This function removes the index, maintaining only
        the proper symbol.
        :param symbol_seq: concatenated sequence
        :return sequence: symbol sequence as a long, unique string, can be easily
        converted to a list if necessary (list())
        """
        # assert len(symbol_seq[0]) <= 1, 'Sequence argument should be concatenated prior to calling this function...'
        if self.validate() is not True:
            logger.info('No corrections necessary...')

        symbol_seq = list(symbol_seq)

        # list all indexed symbols
        bList = map(lambda x: x, set(self.states).difference(set(self.alphabet)))
        # corresponding non-indexed symbols
        symbol = [x[0] for x in bList]

        assert not (set(symbol).difference(set(self.alphabet))), 'Symbols in states do not match symbols in alphabet'

        # Replace sequence symbols in bList with the corresponding symbols in symbol list
        for i, n in enumerate(bList):
            idx = np.where(np.array(symbol_seq) == n)
            idx = idx[0].tolist()
            if len(idx) > 1:
                for a in idx:
                    symbol_seq[a] = symbol[i]
            elif idx:
                symbol_seq[idx[0]] = symbol[i]

        # each entry in the sequence should be one symbol
        tmp = [len(symbol_seq[n]) <= 1 for n in range(len(symbol_seq))]
        assert np.mean(tmp) == 1., 'Sequence entries not uniquely defined'

        # Remove possible trailing spaces
        sequence = ''.join(symbol_seq)

        return list(itertools.chain(sequence))

    def compute_all_ngrams(self, n, limit=10000):
        """
        Determine all possible n-grams from strings generated by the grammar
        :param n: n-gram value
        :param limit: max_number of strings to generate (to sample from)
        :return all_ngrams: list with all symbol n-grams
        :return un_ngrams: list of unique n-grams allowed by the grammar
        """
        long_seq = self.correct_symbolseq(self.concatenate_stringset(self.generate_string_set(limit)))

        all_ngrams = []
        for ii, nn in enumerate(list(long_seq)):
            ngram = ''.join(list(long_seq)[ii:ii + n])
            all_ngrams.append(ngram)  #list(long_seq)[ii:ii + n])

        un_ngrams = np.unique(all_ngrams).tolist()

        # check if all possibilities are represented (n-gram frequency):
        count = []
        for ii, nn in enumerate(un_ngrams):
            if len(nn) < n:
                un_ngrams.pop(ii)
            else:
                count.append(len(np.where(np.array(all_ngrams) == nn)[0]))
        assert np.min(count) > 10, "Increase limit..."

        return all_ngrams, un_ngrams

    def build_tr_matrix(self, n, sample=10000):
        """
        Build the transition matrix, based on the lift method
        # (ref)
        :param n: n_gram value
        :param sample: how many sample strings to use
        :return M: transition matrix
        """

        all_ngrams, un_ngrams = self.compute_all_ngrams(n, sample)

        M = np.zeros((len(un_ngrams), len(un_ngrams)))
        nGrams = np.array(all_ngrams)
        for ii, i in enumerate(un_ngrams):
            for jj, j in enumerate(un_ngrams):
                M[ii, jj] = float(any(nGrams[np.where(nGrams == i)[0][:-1] + 1] == j))

        return M

    def compute_topological_entropy(self, sample=1000):
        """
        Compute the grammar's complexity (topological entropy) - ref.
        :return: n - sufficient lift (corresponds also to best n-gram model)
        :return: TE - topological entropy (at final lift)
        """
        # Determine the lift:
        logger.info("Computing %s TE" % str(self.name))
        TE = []

        for nn in [1, 2, 3, 4, 5]:
            M = self.build_tr_matrix(nn, sample=sample)

            max_eig = np.real(np.max(np.linalg.eigvals(M)))
            TE.append(np.log(max_eig))

            if (len(TE) > 1) and (np.round(np.diff(TE), 1)[-1] == 0.):
                top_ent = TE[-2]
                n = nn - 1
                break

        return n, top_ent

    def fit_model(self, n_strings=100000, model='3_gram', test=True):
        """
        Generate a long symbolic sequence and iteratively fit different predictive models
        :param n_strings: maximum number of strings to generate (for frequency counts)
        :param model: [str] type of model to fit (fixed context -> 'n_gram', variable_context -> '')
        :return log_loss: average log loss over the test data
        """
        # TODO fit models to different grammars

        if model[-4:] == 'gram':
            n = int(model[0])

            # determine all symbol n-grams
            all_ngrams, un_ngrams = self.compute_all_ngrams(n, limit=n_strings)
            all_prefixes, un_prefixes = self.compute_all_ngrams(n - 1, limit=n_strings)

            prob = [[] for i in range(len(self.alphabet))]
            seq = [[] for i in range(len(self.alphabet))]

            # Conditional probabilities estimated by MLE, using the counts from a large enough sequence:
            # P(s1|s0) = c(s0s1)/c(s0) - counts of n_gram divided by prefix counts
            for ii, i in enumerate(self.alphabet):
                # For each symbol, find all the n_grams where it is terminal
                a = [nn for nn in un_ngrams if nn[-1] == i]

                # check the frequency of all possible prefixes
                pref = [nn[:-1] for nn in a]

                # count the occurrences of the n_gram and the prefix
                nG_counts = [len(np.where(np.array(all_ngrams) == nn)[0]) for nn in a]
                pr_counts = [len(np.where(np.array(all_prefixes) == nn)[0]) for nn in pref]

                # estimate the probabilities
                for idx in range(len(a)):
                    prob[ii].append(float(nG_counts[idx]) / float(pr_counts[idx]))
                seq[ii] = a

            if test:
                test_seq = self.correct_symbolseq(self.concatenate_stringset(self.generate_string_set(n_strings)))
                test_pr = []
                if n == 1:
                    for idxi, ii in enumerate(list(test_seq)):
                        idx = np.where(np.array(seq) == ii)[0][0]
                        test_pr.append(prob[idx])
                else:
                    for ii in range(n, len(list(test_seq))):
                        idd = test_seq[ii - n:ii]
                        ind = [(j, i) for j, x in enumerate(seq) for i, y in enumerate(x) if y == idd]
                        if ind:
                            idx = ind[0]
                        else:
                            idx = None
                        if idx is not None:
                            test_pr.append(prob[idx[0]][idx[1]])
                        else:
                            test_pr.append(0.)

                log_loss = -np.sum(np.log2(test_pr) / len(list(test_seq)))

                return prob, seq, log_loss

            else:
                return prob, seq

        else:
            n = 0


########################################################################################################################
class StimulusSet(object):
    """
    StimulusSet object is a wrapper to hold and manipulate all the data pertaining to the input stimuli,
    labels, and corresponding time series.
    """

    def __init__(self, initializer=None, unique_set=False):
        """
        Initialize the StimulusSet object
        :param initializer: global ParameterSet, dictionary or Grammar object
        :param unique_set:
        """
        logger.info("\nGenerating StimulusSet: ")
        self.grammar = None
        self.full_stim_set = None
        self.transient_stim_set = None
        self.train_stim_set = None
        self.test_stim_set = None
        self.full_set = None
        self.full_set_labels = None
        self.transient_set = None
        self.transient_set_labels = None
        self.train_set = None
        self.train_set_labels = None
        self.test_set = None
        self.test_set_labels = None
        self.dims = 0
        self.elements = 0
        if unique_set:
            self.unique_set = None
            self.unique_set_labels = None

        # initializer is dict
        if isinstance(initializer, dict) and not isinstance(initializer, parameters.ParameterSet):
            initializer = parameters.ParameterSet(initializer)
            full_parameters = parameters.ParameterSet(initializer.copy())
            initializer = full_parameters.stim_pars

        if isinstance(initializer, parameters.ParameterSet):
            full_parameters = parameters.ParameterSet(initializer.copy())
            initializer = full_parameters.stim_pars

            self.dims = initializer.n_stim
            self.elements = initializer.elements
            if initializer.grammar is not None:
                self.grammar = Grammar(initializer.grammar, pre_set=initializer.grammar.pre_set,
                                       pre_set_full_path=initializer.grammar.pre_set_path)
                self.elements = self.grammar.alphabet
                self.dims = len(self.grammar.alphabet)
                self.grammar.print_rules()

        elif isinstance(initializer, Grammar):
            self.dims = len(initializer.alphabet)
            self.elements = initializer.alphabet
            self.grammar = Grammar

        elif initializer is not None:
            raise TypeError("Initializer of StimulusSet has incorrect type")

    def load_data(self, data_mat, type='full_set'):
        """
        Load external data - can be stimulus set or stimulus labels, full, train or test
        :param data_mat: data matrix (np.ndarray or list), [N x T]!!
        :param type: data type, has to be consistent with StimulusSet attributes
        """
        logger.info("\t- Loading external data [{0}]".format(str(type)))
        assert isinstance(data_mat, list) or isinstance(data_mat, np.ndarray), "Provided signal must be a list or " \
                                                                               "numpy array"
        if isinstance(data_mat, list):
            data_mat = np.array(data_mat)
        if isinstance(data_mat, np.ndarray) and not type[-6:] == 'labels':
            if np.mean(data_mat[data_mat.nonzero()] == 1) and not isinstance(data_mat, coo_matrix):
                data_mat = coo_matrix(data_mat)
        try:
            setattr(self, type, data_mat)
        except:
            raise TypeError("data type error")

        if type[-6:] == 'labels':
            self.elements = np.unique(data_mat)
            self.dims = len(self.elements)
            data_mat2 = self.stimulus_sequence_to_binary(getattr(self, type))
            self.load_data(np.array(data_mat2.todense()), type=type[:-7])
        else:
            self.dims = data_mat.shape[0]

    def _build_stimulus_seq(self, n_strings, string_length=[], separator=''):
        """
        Build a symbolic time series based on the provided grammar or, if no grammar is specified,
        by randomly drawing the stimulus elements

        :param n_strings: number of strings to generate
        :param string_length: list with the interval of [min, max] str length
        :param separator: str used to separate the individual strings...
        :return n: iterator for full stimulus sequence
        """
        if self.grammar is not None:
            self.full_stim_set = self.grammar.generate_string_set(n_strings, string_length, debug=False)
            str_set = self.grammar.concatenate_stringset(self.full_stim_set, separator=separator)
            seq = self.grammar.correct_symbolseq(str_set)
        else:
            seq = []
            for n in range(n_strings):
                if string_length:
                    length = np.random.randint(string_length[0], string_length[1])
                else:
                    length = 1
                idxs = np.random.randint(0, len(self.elements), length)
                str_ = []
                for nn in idxs:
                    str_.append(self.elements[nn])

                seq.append(str_)
        return seq

    def stimulus_sequence_to_binary(self, seq):
        """
        Convert a stimulus sequence to a binary time series
        :param seq: symbolic sequence
        """
        tmp = lil_matrix(np.zeros((self.dims, len(seq))))

        if isinstance(seq[0], basestring) or isinstance(seq[0], int):
            for i, x in enumerate(self.elements):
                idx = np.where(np.array(seq) == x)
                tmp[i, idx] = 1.
        elif isinstance(seq[0], list):
            for i, x in enumerate(self.elements):
                for ii, xx in enumerate(np.array(seq)):
                    if x == xx:
                        tmp[i, ii] = 1.
        return coo_matrix(tmp)

    def create_set(self, set_length, string_length=[], separator=''):
        """
        Create the full stimulus set (optional)
        :param set_length:
        :param string_length:
        :param separator:
        :return:
        """
        self.full_set_labels = self._build_stimulus_seq(set_length, string_length, separator)
        if isinstance(self.full_set_labels[0], list) and self.grammar is not None:
            seq = self.grammar.concatenate_stringset(self.full_set_labels, separator)
            seq = self.grammar.correct_symbolseq(seq)
        else:
            seq = self.full_set_labels
        self.full_set = self.stimulus_sequence_to_binary(seq)
        logger.info("- Creating full stimulus sequence [{0}]".format(str(len(self.full_set_labels))))

    def create_unique_set(self, full_set_position=0):
        """
        Create a stimulus sequence where each element appears only once and append it
        to the full_set(labels) at position `full_set_position`.
        :param full_set_position: list index where the unique sequence should be inserted
        :return:
        """
        elements = self.elements
        set_length = len(elements)
        seq = elements[:set_length]

        if hasattr(self, "unique_set"):
            self.unique_set_labels = seq
            self.unique_set = self.stimulus_sequence_to_binary(seq)
            self.full_set_labels.insert(full_set_position, self.unique_set_labels)
            self.full_set_labels = list(itertools.chain(*self.full_set_labels))
            self.full_set = self.stimulus_sequence_to_binary(self.full_set_labels)
            logger.info("- Creating unique stimulus sequence [{0}]".format(str(len(self.unique_set_labels))))
        else:
            logger.info("- Skipping generation of unique stimulus sequence, not required.")

    def divide_set(self, transient_set_length, train_set_length, test_set_length):
        """
        Divide the full dataset into train and test data.
        Alternatively generate train and test data independently.
        :param transient_set_length:
        :param train_set_length: length of train set
        :param test_set_length: length of test set
        :return:
        """
        assert (self.full_set is not None), "Full set hasn't been generated"
        if hasattr(self, 'unique_set'):
            assert (len(self.full_set_labels) == transient_set_length + train_set_length + test_set_length +
                    len(self.unique_set_labels)), "Inconsistent dimensions"
            unique_set_length = len(self.unique_set_labels)
        else:
            assert (len(self.full_set_labels) == transient_set_length + train_set_length + test_set_length), \
                "Inconsistent dimensions"
            unique_set_length = 0

        if self.grammar is not None:
            self.train_set_labels = self.full_set_labels[transient_set_length + unique_set_length:train_set_length +
                                                                                                  transient_set_length + unique_set_length]
            self.test_set_labels = self.full_set_labels[transient_set_length + unique_set_length + train_set_length:]
            set_ = self.full_set.copy()
            if not isinstance(set_, np.ndarray):
                set_ = set_.toarray()
            train_set = set_[:, transient_set_length + unique_set_length:transient_set_length + unique_set_length +
                                                                         train_set_length]
            test_set = set_[:, transient_set_length + unique_set_length + train_set_length:]
        else:
            self.train_set_labels = self.full_set_labels[transient_set_length + unique_set_length:train_set_length +
                                                                                                  transient_set_length + unique_set_length]
            self.test_set_labels = self.full_set_labels[transient_set_length + unique_set_length + train_set_length:]
            set_ = self.full_set.copy()
            if not isinstance(set_, np.ndarray):
                set_ = set_.toarray()
            train_set = set_[:, transient_set_length + unique_set_length:transient_set_length + unique_set_length
                                                                         + train_set_length]
            test_set = set_[:, transient_set_length + unique_set_length + train_set_length:]

        self.train_set = coo_matrix(train_set)
        self.test_set = coo_matrix(test_set)
        logger.info("- Dividing set [train={0} / test={1}]".format(str(len(self.train_set_labels)),
                                                             str(len(self.test_set_labels))))

    def separate_transient_set(self, n_discard=0):
        """
        Isolate initial elements from the full_set (responses may be contaminated with initial transients),
        but do not remove them from full_set. These elements will be stored separately in the
        transient_set (because a signal still needs to be generated from them).
        :param n_discard: number of elements to discard
        """
        if self.grammar is not None:
            assert(self.full_stim_set is not None), "Full set hasn't been generated"
        else:
            assert (self.full_set is not None), "Full set hasn't been generated"

        transient_set = self.full_set.todense()[:, :n_discard]
        self.transient_set_labels = self.full_set_labels[:n_discard]
        self.transient_set = coo_matrix(transient_set)

        logger.info("- Creating transient set [{0}]".format(str(len(self.transient_set_labels))))

    def save(self, path, label=''):
        """
        Save StimulusSet object
        :param path: [str] primary folder to store data to
        :param label: [str] an extra label to be appended to the stored filename
        """
        with open('{0}/StimulusSet_{1}.pkl'.format(path, label), 'w') as fp:
            pickle.dump(self, fp, -1)

    def generate_datasets(self, parameters, external_sequence=None):
        """
        Generate full_set, discard and divide..
        :param parameters:
        :param external_sequence: Complete data sequence
        :return:
        """
        if external_sequence is None:
            self.create_set(parameters.full_set_length)
        else:
            self.load_data(external_sequence, type='full_set_labels')
        self.separate_transient_set(parameters.transient_set_length)
        self.create_unique_set(parameters.transient_set_length)
        self.divide_set(parameters.transient_set_length, parameters.train_set_length,
                        parameters.test_set_length)


########################################################################################################################
class InputSignal(object):
    """
    Generate and store AnalogSignal object referring to the structured input signal u(t)
    """
    def __init__(self, initializer=None, online=False, with_noise=False):
        """

        :param initializer: ParameterSet  or dictionary with all the input signal parameters
        :param online: if online generation of input signal
        :param with_noise: if there's noise to be added to the signal
        """
        self.online = online
        self.with_noise = with_noise
        if initializer is None:
            self.dimensions = 0
            self.dt = 0
            self.kernel = None
            self.peak = 0
            self.amplitudes = None
            self.base = 0
            self.durations = None
            self.intervals = None
            self.global_start = 0
            self.global_stop = 0
            self.onset_times = [[] for _ in range(self.dimensions)]
            self.offset_times = [[] for _ in range(self.dimensions)]
            self.time_data = None

        else:
            if isinstance(initializer, dict):
                initializer = parameters.ParameterSet(initializer)
            if isinstance(initializer, parameters.ParameterSet):
                self.dimensions = initializer.N
                self.dt = initializer.resolution
                self.kernel = initializer.kernel
                self.peak = initializer.max_amplitude
                self.amplitudes = None
                self.base = initializer.min_amplitude
                self.duration_parameters = initializer.durations
                self.durations = initializer.durations
                self.interval_parameters = initializer.i_stim_i
                self.intervals = initializer.i_stim_i
                self.global_start = initializer.start_time
                self.global_stop = initializer.stop_time
                self.onset_times = [[] for _ in range(self.dimensions)]
                self.offset_times = [[] for _ in range(self.dimensions)]
                self.time_data = None
        self.input_signal = []

    def load_signal(self, signal, dt=1., onset=0., inherit_from=None):
        """
        Load externally generated continuous signal
        :param signal: numpy.ndarray or list containing the values on each time step
        :param dt: time resolution
        :param onset: global signal onset time [ms]
        :param inherit_from:
        :return:
        """
        assert isinstance(signal, list) or isinstance(signal, np.ndarray) \
               or isinstance(signal, signals.AnalogSignalList), \
            "Provided signal must be a list or numpy array or AnalogSignalList"

        if isinstance(signal, list):
            signal = np.array(signal)
        elif isinstance(signal, np.ndarray):
            self.dt = dt
            if len(np.shape(signal)) > 1:
                self.dimensions = signal.shape[0]
            else:
                self.dimensions = 1
            self.kernel = None
            self.peak = np.max(signal)
            self.amplitudes = None
            self.base = np.min(signal)
            self.durations = None
            self.intervals = None
            self.global_start = onset
            if len(np.shape(signal)) > 1:
                self.global_stop = self.global_start + (signal.shape[1] / dt)
            else:
                self.global_stop = self.global_start + (len(signal) / dt)
            self.onset_times = [[] for _ in range(self.dimensions)]
            self.offset_times = [[] for _ in range(self.dimensions)]
            self.time_data = np.arange(onset, self.global_stop, dt)
            self.input_signal = []

            if self.dimensions > 1:
                for nn in range(self.dimensions):
                    self.input_signal.append(signals.AnalogSignal(signal[nn, :], self.dt, t_start=self.global_start,
                                                                  t_stop=self.global_stop))
            else:
                self.input_signal.append(signals.AnalogSignal(signal, self.dt, t_start=self.global_start,
                                                              t_stop=self.global_stop))
            self.generate()

        elif isinstance(signal, signals.AnalogSignalList):
            self.dt = dt
            self.dimensions = np.shape(signal)[0]
            self.kernel = None
            self.peak = np.max(signal.raw_data()[:, 0])
            self.amplitudes = None
            self.base = np.min(signal.raw_data()[:, 0])
            self.durations = None
            self.intervals = None
            self.global_start = onset
            self.global_stop = self.global_start + (np.shape(signal.raw_data()[:, 0])[0] / dt)
            self.onset_times = [[] for _ in range(self.dimensions)]
            self.offset_times = [[] for _ in range(self.dimensions)]
            self.time_data = np.arange(onset, self.global_stop, dt)
            self.input_signal = signal

        if inherit_from is not None:
            assert isinstance(inherit_from, InputSignal), "Class properties must be inherited from InputSignal object"

            self.kernel = inherit_from.kernel
            self.amplitudes = inherit_from.amplitudes
            self.durations = inherit_from.durations
            self.intervals = inherit_from.intervals
            self.onset_times = inherit_from.onset_times
            self.offset_times = inherit_from.offset_times

    def set_stimulus_amplitudes(self, seq):
        """
        Unfold the amplitude of each stimulus presentation
        :param seq: symbolic or binary input stimulus sequence
        :return: self.amplitudes - single stimulus amplitudes
        """
        if isinstance(seq, list):
            if isinstance(seq[0], basestring) or isinstance(seq[0], list):
                seq = stimulus_sequence_to_binary(seq)
        if isinstance(self.peak[0], float) or isinstance(self.peak[0], int):
            if len(self.peak) == 1:
                amp = np.repeat(self.peak[0], np.shape(seq)[1])
            else:
                amp = self.peak
        elif isinstance(self.peak[0], tuple):
            kwargs = self.peak[0][1]
            if isinstance(kwargs, dict) and 'size' in kwargs:
                kwargs['size'] = np.shape(seq)[1]
            amp = self.peak[0][0](**kwargs)
        else:
            logger.info("max_amplitude parameter must be a list with a single value, multiple values or a tuple of (" \
                  "function, parameters)")
            amp = 0
        self.amplitudes = [[] for _ in range(self.dimensions)]
        amplitudes = np.zeros((self.dimensions, np.shape(seq)[1]))
        for n in range(self.dimensions):
            if len(amp) == self.dimensions:
                amplitudes[n, :] = seq.toarray()[n, :] * amp[n]
            else:
                amplitudes[n, :] = seq.toarray()[n, :] * amp
            tmp = amplitudes[n, :]
            for ii in list(tmp[tmp.nonzero()]):
                self.amplitudes[n].append(ii)

    def set_stimulus_times(self, seq, spk_patterns=False):
        """
        Extract the onset and offset times for the stimulus, using the durations and intervals
        :param seq: binary or symbolic stimulus sequence
        :param spk_patterns:
        """
        if isinstance(seq, list):
            if isinstance(seq[0], basestring) or isinstance(seq[0], list):
                seq = stimulus_sequence_to_binary(seq)

        if isinstance(self.duration_parameters[0], tuple) and isinstance(self.interval_parameters[0], tuple):
            # TODO: take in a list of functions...
            kwargs = self.duration_parameters[0][1]
            if isinstance(kwargs, dict) and 'size' in kwargs:
                kwargs['size'] = np.shape(seq)[1]
            dur = np.round(self.duration_parameters[0][0](**kwargs))

            kwargs = self.interval_parameters[0][1]
            if isinstance(kwargs, dict) and 'size' in kwargs:
                kwargs['size'] = np.shape(seq)[1]
            i_stim_i = np.round(self.interval_parameters[0][0](**kwargs))

        elif (isinstance(self.duration_parameters[0], float) or isinstance(self.duration_parameters[0], int)) and (isinstance(
                self.interval_parameters[0], float) or isinstance(self.interval_parameters[0], int)):

            if len(self.duration_parameters) == 1:
                dur = np.repeat(self.duration_parameters[0], np.shape(seq)[1])
            else:
                dur = self.duration_parameters
            if len(self.interval_parameters) == 1:
                i_stim_i = np.repeat(self.interval_parameters[0], np.shape(seq)[1])  # TODO: change number of intervals!!
            else:
                i_stim_i = self.interval_parameters
        else:
            dur = 0
            i_stim_i = 0

        assert (len(dur) == seq.shape[1]), "Provided durations don't match number of elements"
        assert (len(i_stim_i) == seq.shape[1]), "Provided intervals don't match number of sequence elements"

        onsets = []
        offsets = []
        if self.online:
            onset = 0.
        else:
            onset = self.global_start
        for nn, ii in enumerate(dur):
            onsets.append(onset)
            offsets.append(onset + ii)
            onset += ii + i_stim_i[nn]

        # segregate by input signal...
        self.durations = [[] for _ in range(self.dimensions)]
        onset_times = np.zeros((self.dimensions, len(onsets)))
        offset_times = np.zeros((self.dimensions, len(offsets)))
        if self.online:
            self.onset_times = [[] for _ in range(self.dimensions)]
            self.offset_times = [[] for _ in range(self.dimensions)]
        durations = np.zeros((self.dimensions, seq.shape[1]))
        for n in range(self.dimensions):
            onset_times[n, :] = seq.toarray()[n, :] * onsets
            tmp = onset_times[n, :]
            for ii0 in list(tmp[tmp.nonzero()]):
                self.onset_times[n].append(ii0)
            if seq.toarray()[n, 0] and onset == 0. and self.online:
                self.onset_times[n].insert(0, 0.0)
            elif seq.toarray()[n, 0] and onset == 0. and not self.online and spk_patterns:
                self.onset_times[n].insert(0, 0.0)

            offset_times[n, :] = seq.toarray()[n, :] * offsets
            tmp = offset_times[n, :]
            for ii1 in list(tmp[tmp.nonzero()]):
                self.offset_times[n].append(ii1)

            durations[n, :] = seq.toarray()[n, :] * dur
            tmp = durations[n, :]
            for ii2 in list(tmp[tmp.nonzero()]):
                self.durations[n].append(ii2)

        self.intervals = i_stim_i
        self.global_stop = max(offsets)
        self.global_stop += self.dt
        self.time_data = np.arange(self.global_start, self.global_stop, self.dt)

    def apply_input_mask(self):
        """
        Expand the stimulus sequence in time by applying a mask
        :return:
        """
        time_data = self.time_data
        signal = np.zeros((self.dimensions, len(time_data)))
        s = np.zeros_like(signal)

        for nn in range(self.dimensions):
            # for each stimulus presentation, mark the mid-point..
            onsets = np.array(self.onset_times[nn])
            offsets = np.array(self.offset_times[nn])
            mid_points = ((onsets / self.dt) + (offsets / self.dt)) / 2.
            signal[nn, (mid_points - (self.global_start / self.dt)).astype(int)] = 1.

        tmp_durations = np.array(list(itertools.chain(*self.durations)))
        tmp_amplitudes = np.array(list(itertools.chain(*self.amplitudes)))

        # convolve the signal with the input mask kernel
        if (len(np.unique(tmp_durations[tmp_durations != 0.])) == 1) and (len(np.unique(tmp_amplitudes[
                                                                                            tmp_amplitudes != 0.])) == 1):
            # case when all stimuli have the same duration and amplitude
            k = make_simple_kernel(self.kernel[0], width=tmp_durations[0],
                                   height=tmp_amplitudes[0], resolution=self.dt, normalize=False, **self.kernel[1])

            for nn in range(self.dimensions):
                s[nn, :] = fftconvolve(signal[nn, :], k, 'same')

                # dirty hack to solve the 0-isi problem...
                idx = np.where(s[nn, :] > np.unique(list(itertools.chain(*self.amplitudes))))[0] # durations
                if idx.size:
                    s[nn, idx] = np.unique(list(itertools.chain(*self.amplitudes)))

                # pad with zeros (make sure) and re-scale to minimum and maximum
                idx = np.where(s[nn, :] <= 0.0001)[0]
                if idx.size:
                    s[nn, idx] = 0.
                s[nn, :] += self.base  #np.array(rescale_signal(s[nn, :], self.base, self.peak[0]))

                # dirty hack (to solve the doubled peak with box kernel with 0 isi):
                idx = np.where(s[nn, :] > tmp_amplitudes[0])[0]
                if idx.size:
                    s[nn, idx] = tmp_amplitudes[0]

                self.input_signal.append(signals.AnalogSignal(s[nn, :], self.dt, t_start=self.global_start,
                                                              t_stop=self.global_stop))
        else:
            # case when stimuli different durations and/or amplitudes
            for nn in range(self.dimensions):
                for idx, ii in enumerate(self.onset_times[nn]):
                    if len(np.unique(tmp_durations[tmp_durations != 0.])) != 1:
                        duration = self.durations[nn][idx]
                    else:
                        duration = np.unique(tmp_durations)[0]

                    if len(np.unique(tmp_amplitudes[tmp_amplitudes != 0])) != 1:
                        amplitude = self.amplitudes[nn][idx]
                    else:
                        amplitude = np.unique(tmp_amplitudes)[0]

                    k = make_simple_kernel(self.kernel[0], width=duration, height=amplitude,
                                           resolution=self.dt, normalize=False, **self.kernel[1])

                    if idx < len(self.onset_times[nn]) - 1:
                        time_window = [ii, self.onset_times[nn][idx + 1]]
                    else:
                        time_window = [ii, self.global_stop]

                    local_signal = signal[nn, int(time_window[0] / self.dt):int(time_window[1] / self.dt)]
                    s[nn, int(time_window[0] / self.dt):int(time_window[1] / self.dt)] = fftconvolve(local_signal,
                                                                                                     k, 'same')
                    tmp = np.round(s[nn, int(time_window[0] / self.dt):int(time_window[1] / self.dt)])

                    # pad with zeros (make sure) and add baseline value
                    idx = np.where(tmp <= 0.001)[0]
                    if idx.size:
                        s[nn, idx + int(time_window[0] / self.dt)] = 0.

                    s[nn, int(time_window[0] / self.dt):int(time_window[1] / self.dt)] += self.base

                self.input_signal.append(signals.AnalogSignal(s[nn, :], self.dt, t_start=self.global_start,
                                                              t_stop=self.global_stop))


    def compress_signals(self, input_signal=None):
        """
        Converts the input signal from a list of AnalogSignal objects to a single AnalogSignalList. This is a more
        efficient way because the already existing AnalogSignal objects are simply appended
        to an empty AnalogSignalList object, instead of recreating them twice.

        This function is called only through the InputSignal.load_signal() function, which does a few necessary
        initializations.

        :param: input_signal: list of AnalogSignal objects
        :return analog_signal: AnalogSignalList created from the given list (or from self.input_signal)
        """
        if input_signal is not None:
            s = input_signal
            t = input_signal[0].time_axis()
        else:
            s = self.input_signal
            t = self.time_data

        # generate empty AnalogSignalList
        compressed_signal = signals.AnalogSignalList([], [], times=t, dt=self.dt, t_start=min(t),
                                                     t_stop=max(t) + self.dt, dims=len(s))

        # append individual signals to empty AnalogSignalList
        for idx in range(len(s)):
            compressed_signal.append(idx, s[idx])

        return compressed_signal

    def generate(self):
        """
        Generate the final signal
        """
        if not signals.empty(self.input_signal):
            self.input_signal = self.compress_signals()
        elif self.online:
            self.input_signal = None
        else:
            self.apply_input_mask()
            self.input_signal = self.compress_signals()

    def as_array(self):
        """
        """
        signals = self.input_signal.analog_signals
        signal_array = np.zeros((len(signals), int(signals[0].duration())))
        for n in range(len(signals)):
            signal_array[n, :] = signals[n].signal

        return signal_array

    def time_slice(self, start, stop):
        """
        Return a new input signal, which is a temporal slice of the original signal
        :param start: start time of slice
        :param stop: end of slice
        :return:
        """
        new_signal = InputSignal()
        new_signal.load_signal(self.input_signal.time_slice(start, stop).as_array(),
                               onset=start, inherit_from=self)
        for signal_idx, n_signal in enumerate(new_signal.onset_times):
            idx1 = np.where(np.array(n_signal) >= start)[0]
            idx2 = np.where(np.array(n_signal) <= stop)[0]
            idx = np.intersect1d(idx1, idx2)
            new_signal.onset_times[signal_idx] = list(np.array(n_signal)[idx])
            new_signal.amplitudes[signal_idx] = list(np.array(new_signal.amplitudes[signal_idx])[idx])
            new_signal.durations[signal_idx] = list(np.array(new_signal.durations[signal_idx])[idx])
        for signal_idx, n_signal in enumerate(new_signal.offset_times):
            idx1 = np.where(np.array(n_signal) <= stop)[0]
            idx2 = np.where(np.array(n_signal) >= start)[0]
            idx = np.intersect1d(idx1, idx2)
            new_signal.offset_times[signal_idx] = list(np.array(n_signal)[idx])
        new_signal.intervals = new_signal.intervals[:len(list(signals.iterate_obj_list(new_signal.amplitudes)))-1]

        return new_signal

    def time_offset(self, offset):
        """
        Offset the entire signal (and all its temporal components)
        Note that this function shifts the current signal, no new signal is generated
        :param offset: value to shift by
        :return:
        """
        if self.input_signal:
            self.input_signal = self.input_signal.time_offset(offset)
        self.global_start += offset
        self.global_stop += offset
        self.time_data += offset
        for idx, a in enumerate(self.onset_times):
            if isinstance(a, list):
                self.onset_times[idx] = [b+offset for b in a]
        for idx, a in enumerate(self.offset_times):
            if isinstance(a, list):
                self.offset_times[idx] = [b+offset for b in a]

    def set_signal_online(self, stim_seq):
        """
        Sets amplitudes and times online. Useful for very large datasets, where storing the whole input time series
        would consume too much memory. This function instead returns a generator, where each stimulus step is
        generated online.

        :param stim_seq: stimulus sequence, can be
        :return:
        """
        if isinstance(stim_seq, list):
            if isinstance(stim_seq[0], basestring) or isinstance(stim_seq[0], list):
                seq = stimulus_sequence_to_binary(stim_seq)
        else:
            seq = stim_seq
        seq_dense = seq.todense()

        for nn in range(seq.shape[1]):
            self.set_stimulus_amplitudes(coo_matrix(seq_dense[:, nn]))
            self.set_stimulus_times(coo_matrix(seq_dense[:, nn]))
            yield self

    # TODO inter-stimulus intervals are not handled / are ignored here. Implement
    def generate_iterative_fast(self, stim_seq):
        """
        For very large input signals, we only return the signals
        and their ids which must be changed at the current step.

        NOTE!: 1) we assume here that multiple stimuli can be shown at each step, however,
        the ones shown simultaneously must have the same start and stop times (length)!

        :param stim_seq:
        :return [AnalogSignalList] containing usually only a few signals, the ones to be reset (active in the previous
                timestep, and the ones currently, newly active
        """
        if isinstance(stim_seq, list):
            if isinstance(stim_seq[0], basestring) or isinstance(stim_seq[0], list) or isinstance(stim_seq[0], int):
                seq = stimulus_sequence_to_binary(stim_seq)
            else:
                raise ValueError('Stimulus sequence is a list instance, but unknown elements in the list!')
        else:
            seq = stim_seq

        # at this point seq is a binary matrix (n_stim x len(stimulus_set))
        seq_dense = seq.todense()
        # list of signal (stim) ids to be reset in each step (previous ones)
        to_reset_ids = []

        # NOTE: between each 2 iterations of the loop below (generator), the amplitudes, times, and other
        # properties are updated in a different function to match the current step
        for step in range(seq.shape[1]):
            if signals.empty(self.onset_times):
                item_index = np.where(self.offset_times)[0][0]
                self.onset_times[item_index] = [self.offset_times[item_index][0] - self.durations[item_index][0]]

            in_vec = seq_dense[:, step] 			# get the column for the current step
            active_stim_ids = np.where(in_vec)[0] 	# this is a list, with possibly multiple elements
            active_signals = []  					# list with [(signal_id, AnalogSignal object)] tuples

            assert len(active_stim_ids) > 0, "Something went wrong, there must be at least one signal for each step!"

            # compute this once, here, because we assume they are the same for each signal active in this step
            first_active_id = active_stim_ids[0]
            # these values have been updated for the current step elsewhere, just use them here!
            amp = self.amplitudes[first_active_id][0]
            dur = self.durations[first_active_id][0]
            on  = self.onset_times[first_active_id][0]
            off = self.offset_times[first_active_id][0]

            time_data = np.arange(on, off, self.dt)
            signal_ = np.zeros(len(time_data))  # this will be for the new (active) signal

            mid_point = len(time_data) / 2.
            signal_[int(mid_point)] = 1.

            # TODO check if kernel could be created a single time
            if isinstance(self.kernel[1], dict):
                kern = make_simple_kernel(self.kernel[0], width=dur, height=amp, resolution=self.dt,
                                          normalize=False, **self.kernel[1])
            elif isinstance(self.kernel[1], list):
                kern = make_simple_kernel(self.kernel[0], width=dur, height=amp, resolution=self.dt,
                                          normalize=False, **self.kernel[1][first_active_id])
            else:
                raise ValueError('Wrong input kernel parameters? Only list and dictionaries are accepted!')

            s = fftconvolve(signal_, kern, mode='same')

            # create currently active analog signals and store them in a list
            for _ in active_stim_ids:
                active_signals.append(signals.AnalogSignal(np.copy(s), self.dt, t_start=on, t_stop=off))

            # generate empty AnalogSignalList
            analog_signals = signals.AnalogSignalList([], [], times=time_data, dt=self.dt,
                                                      t_start=min(time_data), t_stop=max(time_data) + self.dt)

            # create reset signal template; only first timepoint is set to 0., rest follows automatically
            reset_signal_template = np.zeros(len(time_data))
            reset_signal = signals.AnalogSignal(reset_signal_template, self.dt, t_start=on, t_stop=off)

            # if there's noise, reset all channels (except the active ones) so the noise can later be actually added
            if self.with_noise:
                to_reset_ids = np.arange(self.dimensions)

            # append signal to be reset to final signal list, which will be updated in NEST
            for id_ in to_reset_ids:
                # skip reset for signal if again active
                if id_ in active_stim_ids:
                    continue
                analog_signals.append(id_, copy.deepcopy(reset_signal))

            # clear previous signal ids, they were already reset
            to_reset_ids = []

            # append active signals to empty AnalogSignalList
            for idx, analog_signal in enumerate(active_signals):
                signal_id = active_stim_ids[idx]
                analog_signals.append(signal_id, analog_signal)
                to_reset_ids.append(signal_id)

            yield analog_signals

    def generate_square_signal(self):
        """
        Generates a binary block design array (to be used as target in some cases)
        :return:
        """
        signal = np.zeros((self.dimensions, len(self.time_data)))

        for k in range(self.dimensions):
            onsets = np.copy(self.onset_times[k])
            offsets = np.copy(self.offset_times[k])
            for idx, n in enumerate(onsets):
                idx_start = np.where(self.time_data == n)[0][0]
                if offsets[idx] > max(self.time_data):
                    offsets[idx] -= self.dt
                idx_stop = np.where(self.time_data == offsets[idx])[0][0]
                signal[k, idx_start:idx_stop] = np.ones_like(np.arange(idx_start, idx_stop, 1))
        return signal


########################################################################################################################
class InputNoise(StochasticGenerator):
    """
    Generate and store AnalogSignal object referring to the noise to add to the input signal u(t)
    """
    def __init__(self, initializer, rng=None, seed=None, start_time=None, stop_time=None):
        """

        :param initializer:
        :param rng:
        :param seed:
        :param stop_time:
        """
        StochasticGenerator.__init__(self, rng, seed)

        if isinstance(initializer, dict):
            initializer = parameters.ParameterSet(initializer)

        assert isinstance(initializer, parameters.ParameterSet), "Initializer must be a parameter dictionary or ParameterSet"
        self.N = initializer.N
        self.rectify = initializer.rectify
        self.source = initializer.noise_source
        not_allowed_keys = ['_url']
        tmp_pars = {k: v for k, v in initializer.noise_pars.as_dict().items() if k not in not_allowed_keys}
        self.parameters = tmp_pars
        self.dt = initializer.resolution
        if isinstance(initializer.start_time, list):
            self.onset_times = initializer.start_time
            self.global_start = initializer.start_time[0]
            self.offset_times = initializer.stop_time
            self.global_stop = initializer.stop_time[-1]
        else:
            self.global_start = initializer.start_time
            self.global_stop = initializer.stop_time

        if start_time is not None:
            self.global_start = start_time
        if stop_time is not None:
            self.global_stop = stop_time

        self.noise_signal = []
        self.time_data = np.arange(self.global_start, self.global_stop, self.dt)

    # TODO @comment
    def generate(self):
        """
        """
        for ii in range(self.N):
            if len(self.source) == 1:
                self.source = list(np.repeat(self.source[0], self.N))
            if not isinstance(self.source[ii], basestring):
                amplitude = self.parameters.pop('amplitude')
                self.parameters.pop('label')
                self.noise_signal.append(self.continuous_rv_generator(self.source[ii], amplitude=amplitude,
                                                                      t_start=self.global_start,
                                                                      t_stop=self.global_stop, dt=self.dt,
                                                                      rectify=self.rectify, array=False,
                                                                      **self.parameters))
            elif isinstance(self.source[ii], basestring) and self.source[ii] not in ['GWN', 'OU']:
                amplitude = self.parameters.pop('amplitude')
                self.parameters.pop('label')
                self.noise_signal.append(self.continuous_rv_generator(self.source[ii], amplitude=amplitude,
                                                                      t_start=self.global_start,
                                                                      t_stop=self.global_stop, dt=self.dt,
                                                                      rectify=self.rectify, array=False,
                                                                      **self.parameters))
            elif self.source[ii] == 'GWN':
                self.noise_signal.append(self.GWN_generator(amplitude=self.parameters['amplitude'],
                                                            mean=self.parameters['mean'],
                                                            std=self.parameters['std'],
                                                            t_start=self.global_start,
                                                            t_stop=self.global_stop,
                                                            dt=self.dt, rectify=self.rectify, array=False))
            elif self.source[ii] == 'OU':
                self.noise_signal.append(self.OU_generator(dt=self.parameters['dt'], tau=self.parameters['tau'],
                                                           sigma=self.parameters['sigma'],
                                                           y0=self.parameters['y0'],
                                                           t_start=self.global_start, t_stop=self.global_stop,
                                                           rectify=self.rectify, array=False))
            else:
                logger.warning("{0} Not currently implemented".format(self.source[ii]))

        channel_ids = []
        sigs = []
        for idx, nn in enumerate(self.noise_signal):
            channel_ids.append(idx * np.ones_like(nn.signal))
            sigs.append(nn.signal)
        sigs = list(itertools.chain(*sigs))
        channel_ids = list(itertools.chain(*channel_ids))

        # generate AnalogSignalList
        if self.N:
            tmp = [(channel_ids[n], sigs[n]) for n in range(len(channel_ids))]
            self.noise_signal = signals.AnalogSignalList(tmp, np.unique(channel_ids).tolist(), times=self.time_data)
        else:
            self.noise_signal = []

    @staticmethod
    def re_seed(global_seed):
        """
        Reset the rng seed (in case it is changed during the signal creation)
        """
        np.random.seed(global_seed)


########################################################################################################################
class InputSignalSet(object):
    """
    Class to hold and manipulate complex sets of input signals.
    """
    #TODO isn't it feasible to have the detailed version as the main constructor?
    def __init__(self, parameter_set, stim_obj=None, rng=None, online=False):
        # def __init__(self, encoding_pars, input_pars, stim_obj=None, rng=None, online=False):
        """
        Initialize member variables. If the input stimuli are spike patterns, generates a spike
        pattern template for each stimulus.

        :param parameter_set: (ParameterSet) global parameter set
        :param stim_obj: (StimulusSet) stimulus set for which the input signal set should be created
        :param rng: pseudo random number generator
        :param online: (bool) generate input signals online (memory efficient) or not
        """
        logger.info("\nGenerating Input Signals: ")
        self.online = online
        # self.parameters = parameter_set.input_pars
        self.parameters = parameter_set.input_pars
        self.transient_set_signal = None
        self.transient_set_signal_iterator = None
        self.transient_set_noise = None
        self.transient_set = None
        self.transient_stimulation_time = 0
        self.full_set_signal = None
        self.full_set_signal_iterator = None
        self.full_set_noise = None
        self.full_set = None
        self.full_stimulation_time = 0
        self.train_set_signal = None
        self.train_set_signal_iterator = None
        self.train_set_noise = None
        self.train_set = None
        self.train_stimulation_time = 0
        self.test_set_signal = None
        self.test_set_signal_iterator = None
        self.test_set_noise = None
        self.test_set = None
        self.test_stimulation_time = 0
        if online:
            self.full_set_signal_iterator = None
        if 'spike_pattern' in parameter_set.encoding_pars.generator.labels and stim_obj is not None:
            self.spike_patterns = []
            n_input_neurons 	= parameter_set.encoding_pars.encoder.n_neurons[0]
            pattern_duration 	= parameter_set.input_pars.signal.durations
            rate 				= parameter_set.input_pars.signal.max_amplitude
            resolution 			= parameter_set.input_pars.signal.resolution

            if parameter_set.encoding_pars.generator.gen_to_enc_W is None:
                for n in range(stim_obj.dims):
                    duration = pattern_duration[n] if len(pattern_duration) == stim_obj.dims else pattern_duration[0]
                    rt = rate[n] if len(rate) == stim_obj.dims else rate[0]

                    if parameter_set.encoding_pars.generator.jitter is not None and \
                            parameter_set.encoding_pars.generator.jitter[1]:
                        duration += (parameter_set.encoding_pars.generator.jitter[0] * 2)

                    spattern = generate_template(n_neurons=n_input_neurons, rate=rt, duration=duration,
                                                 resolution=resolution, rng=rng)
                    self.spike_patterns.append(spattern)
            else:
                # weighted input (modulate rates)
                w = parameter_set.encoding_pars.generator.gen_to_enc_W
                for n in range(stim_obj.dims):
                    if len(pattern_duration) == stim_obj.dims:
                        duration = pattern_duration[n]
                    else:
                        duration = pattern_duration[0]

                    if parameter_set.encoding_pars.generator.jitter is not None and parameter_set.encoding_pars.generator.jitter[1]:
                        duration += (parameter_set.encoding_pars.generator.jitter[0] * 2)

                    if len(rate) == stim_obj.dims:
                        rt = rate[n]
                    else:
                        rt = rate[0]
                    pattern = signals.SpikeList([], [], t_start=0., t_stop=duration)
                    for n_neuron in range(n_input_neurons):
                        rrt = (w[n, n_neuron] * rt) + 0.00000001
                        spk_pattern = generate_template(n_neurons=1, rate=rrt, duration=duration,
                                                        resolution=resolution, rng=rng)
                        if signals.empty(spk_pattern.spiketrains):
                            spk_train = signals.SpikeTrain([], t_start=resolution, t_stop=duration)
                        else:
                            spk_train = spk_pattern.spiketrains[0]
                        pattern.append(n_neuron, spk_train)

                    self.spike_patterns.append(pattern)
        else:
            self.spike_patterns = None
        if stim_obj is not None and hasattr(stim_obj, "unique_set"):
            self.unique_set = None
            self.unique_set_signal = None
            self.unique_set_signal_iterator = None
            self.unique_set_noise = None
            self.unique_stimulation_time = 0

    # TODO this is a workaround for now, we could have this as the main constructor
    @classmethod
    def detailed_constructor(cls, parameter_set_, encoding_pars_, input_pars_, stim_obj_=None, rng_=None,
                             online_=False):
        parameter_set = copy.deepcopy(parameter_set_)
        parameter_set.encoding_pars = encoding_pars_
        parameter_set.input_pars = input_pars_
        return cls(parameter_set, stim_obj_, rng_, online_)

    def generate_generic_set(self, stimulus_subset, set_label, timestamp_correction=None):
        """
        Generic function used by several functions (generate_full_set, generate_train_set, etc.) to generate
        or initialize input signals corresponding to the stimulus set being generated
        (full, transient, unique, train and test). Signal for stimulus sets can be generated either online
        (via generator objects) or offline.

        :param stimulus_set: StimulusSet object
        :param stimulus_subset: particular type of set, e.g., stimulus_set.full_set
        :param set_label: label for set, e.g., full_set or train_set
        :param timestamp_correction: apply time correction for signal (offset value)
        :return: tuple with 5 elements, which then update the corresponding class member objects,
                 in the calling function

                 -> self_set, self_set_signal, self_set_signal_iterator, self_set_noise, set_stimulation_time
        """
        self_set = None                     # self.full_set
        self_set_signal = None              # .full_set_signal
        self_set_signal_iterator = None     # .full_set_signal_iterator
        self_set_noise = None               # .full_set_noise
        set_stimulation_time = 0.

        assert stimulus_subset is not None, "No {0} in the provided StimulusSet object, skipping...".format(set_label)
        self_set_signal = InputSignal(self.parameters.signal, self.online, hasattr(self.parameters, "noise"))

        logger.info("- Generating {0}-dimensional input signal [{1}]".format(str(self.parameters.signal.N), set_label))
        # stimulus is generated online
        if self.online:
            logger.info("- InputSignal will be generated online. {0} is now a generator.. (no noise is added...)".format(
                set_label))
            # TODO: Noise is not added here
            self_set_signal_iterator = self_set_signal.set_signal_online(stimulus_subset)
            # self_set = self_set_signal.generate_iterative(stimulus_subset)
            self_set = self_set_signal.generate_iterative_fast(stimulus_subset)
        # stimulus is generated offline
        else:
            self_set_signal.set_stimulus_amplitudes(stimulus_subset)
            if not signals.empty(self.spike_patterns):
                self_set_signal.set_stimulus_times(stimulus_subset, spk_patterns=True)
            else:
                self_set_signal.set_stimulus_times(stimulus_subset)

            self_set_signal.generate()

            if timestamp_correction:  # correct time stamp
                self_set_signal.time_offset(timestamp_correction)

            set_stimulation_time = len(self_set_signal.time_data) * self_set_signal.dt

            if hasattr(self.parameters, "noise") and self.parameters.noise.N:
                # for full set we're not updating, otherwise it's circular update
                if set_label != "full_set":
                    self.parameters.noise.start_time = self.parameters.noise.stop_time
                    self.parameters.noise.stop_time += set_stimulation_time

                # this is only required for the full_set generation, otherwise times are deduced from noise
                stop_time = set_stimulation_time if set_label == "full_set" else None

                self_set_noise = InputNoise(self.parameters.noise, stop_time=stop_time)
                self_set_noise.generate()
                if not isinstance(self_set_noise.noise_signal, signals.AnalogSignalList):
                    self_set_noise = None
                    self_set = self_set_signal
                else:
                    merged_signal = merge_signal_lists(self_set_signal.input_signal,
                                                       self_set_noise.noise_signal,
                                                       operation=np.add)
                    self_set = InputSignal()
                    # for full set it's still okay, because start_time at that point is still 0.
                    self_set.load_signal(merged_signal.as_array(), onset=self.parameters.noise.start_time,
                                         inherit_from=self_set_signal)
                    logger.info("- Generating and adding {0}-dimensional input noise (t={1})".format(
                        str(self.parameters.signal.N), str(set_stimulation_time)))

        return self_set, self_set_signal, self_set_signal_iterator, self_set_noise, set_stimulation_time

    def generate_full_set(self, stimulus_set):
        """

        :param stimulus_set:
        :return:
        """

        result = self.generate_generic_set(stimulus_set.full_set, "full_set")
        self.full_set                  = result[0]
        self.full_set_signal           = result[1]
        self.full_set_signal_iterator  = result[2]
        self.full_set_noise            = result[3]
        self.full_stimulation_time     = result[4]

    def generate_transient_set(self, stimulus_set):
        """

        :param stimulus_set:
        :return:
        """
        result = self.generate_generic_set(stimulus_set.transient_set, "transient_set")
        self.transient_set                  = result[0]
        self.transient_set_signal           = result[1]
        self.transient_set_signal_iterator  = result[2]
        self.transient_set_noise            = result[3]
        self.transient_stimulation_time     = result[4]

    def generate_unique_set(self, stimulus_set):
        """

        :param stimulus_set:
        :return:
        """
        timestep_correction = self.transient_stimulation_time

        result = self.generate_generic_set(stimulus_set.unique_set, "unique_set", timestep_correction)
        self.unique_set                  = result[0]
        self.unique_set_signal           = result[1]
        self.unique_set_signal_iterator  = result[2]
        self.unique_set_noise            = result[3]
        self.unique_stimulation_time     = result[4]

    def generate_train_set(self, stimulus_set):
        """

        :param stimulus_set:
        :return:
        """
        timestep_correction = self.transient_stimulation_time
        if hasattr(self, "unique_set"):
            timestep_correction += self.unique_stimulation_time

        result = self.generate_generic_set(stimulus_set.train_set, "train_set", timestep_correction)
        self.train_set                  = result[0]
        self.train_set_signal           = result[1]
        self.train_set_signal_iterator  = result[2]
        self.train_set_noise            = result[3]
        self.train_stimulation_time     = result[4]

    def generate_test_set(self, stimulus_set):
        """

        :param stimulus_set:
        :return:
        """
        timestep_correction = self.transient_stimulation_time + self.train_stimulation_time
        if hasattr(self, "unique_set"):
            timestep_correction += self.unique_stimulation_time

        result = self.generate_generic_set(stimulus_set.test_set, "test_set", timestep_correction)
        self.test_set                  = result[0]
        self.test_set_signal           = result[1]
        self.test_set_signal_iterator  = result[2]
        self.test_set_noise            = result[3]
        self.test_stimulation_time     = result[4]

    def generate_datasets(self, stim):
        """
        Generate all data sets for all the different phases
        :param stim: StimulusSet object
        :return:
        """
        self.generate_full_set(stim)
        if not signals.empty(stim.transient_set_labels):
            self.generate_transient_set(stim)
        if hasattr(stim, "unique_set") and not signals.empty(stim.unique_set_labels):
            self.generate_unique_set(stim)
        self.generate_train_set(stim)
        self.generate_test_set(stim)

    def time_offset(self, offset_time=0.):
        """
        Offset the global set time (shifts all signals by offset_time)
        :return:
        """
        logger.info("Offsetting input signal set by {0}".format(str(offset_time)))
        attrs = ['full', 'transient', 'unique', 'train', 'test']

        for att in attrs:
            if hasattr(self, att + '_set') and getattr(self, att + '_set') is not None:
                set_ = getattr(self, att + '_set')
                set_.time_offset(offset_time)

    def save(self, path):
        """
        Save InputSignalSet object
        :param path: [str] primary folder to store data to
        """
        if not self.online:
            if self.transient_set is not None:
                np.save(path+'_TransientInputSignal.npy', self.transient_set.as_array())
            if hasattr(self, 'unique_set') and self.unique_set is not None:
                np.save(path + '_UniqueInputSignal.npy', self.unique_set.as_array())
            if self.train_set is not None:
                np.save(path+'_TrainInputSignal.npy', self.train_set.as_array())
            if self.test_set is not None:
                np.save(path+'TestInputSignal.npy', self.test_set.as_array())
        else:
            logger.info("InputSignals are generated online..")


########################################################################################################################
class Encoder(net_architect.Population):
    """
    Convert Continuous signal into SpikeList objects, or create a population of spiking neurons
    from a given parameter set
    """

    def __init__(self, par_set):
        net_architect.Population.__init__(self, pop_set=par_set)
        logger.info("\nCreating Input Encoder {0}".format(str(par_set.pop_names)))


########################################################################################################################
class Generator:
    """
    Generate input to the network, generator is a NEST device!
    The generator always inherits the dimensionality of its input and contains the
    connectivity features to its target (N_{input}xN_{target})
    """

    def __init__(self, initializer, input_signal=None, dims=None):
        """
        Create and setup the generators, assigning them the correct labels and ids
        :param initializer: ParameterSet or dict with generator parameters (see EncodingLayer.create_generators())
        :param input_signal: signal object to be converted (either AnalogSignalList or SpikeList) - determines the
        dimensionality of the input to the generator
        :param dims: if the generator input is an encoder, dimensionality needs to be specified
        """
        self.input_dimension = 1
        if isinstance(initializer, dict):
            initializer = parameters.ParameterSet(initializer)
        if input_signal is not None:
            if isinstance(input_signal, InputSignal):
                # 1 generator per input channel
                self.input_dimension = input_signal.dimensions
                self.time_data 		 = input_signal.time_data
            elif isinstance(input_signal, InputNoise):
                # 1 generator per channel
                self.input_dimension = input_signal.N
                self.time_data 		 = input_signal.time_data
            elif isinstance(input_signal, signals.AnalogSignalList) or isinstance(input_signal, signals.SpikeList):
                self.input_dimension = len(input_signal)
                self.time_data 		 = input_signal.time_axis()
            else:
                raise TypeError("input_signal must be InputSignal, AnalogSignalList or SpikeList")
        else:
            self.time_data = []

        # if dimensions are provided, overrides current value
        if dims is not None:
            self.input_dimension = dims

        self.gids 		= []
        self.layer_gid 	= None
        self.topology 	= initializer.topology
        self.model 		= initializer.model
        self.base_name	= None

        if len(initializer.label) == self.input_dimension:
            self.name = initializer.label
        else:
            self.name = []
            self.base_name = str(initializer.label)
            for nn in range(self.input_dimension):
                self.name.append(str(initializer.label) + '{0}'.format(str(nn)))

        for nn in range(self.input_dimension):
            model_dict = initializer.model_pars
            model_dict['model'] = initializer.model

            # nest.CopyModel(initializer.model, self.name[nn])
            # nest.SetDefaults(self.name[nn], parameters.extract_nestvalid_dict(model_dict, param_type='device'))
            # model_name = self.name[nn]

            # if multiple uniquely labeled generators given, create NEST model for each one, based on their label
            if self.base_name is None:
                nest.CopyModel(initializer.model, self.name[nn])
                nest.SetDefaults(self.name[nn], parameters.extract_nestvalid_dict(model_dict, param_type='device'))
            # only create one NEST model if one generator given and then create as many objects of
            # this model as defined by the generators dimensionality. Later only the gids are used..
            elif self.base_name not in nest.Models():
                nest.CopyModel(initializer.model, self.base_name)
                nest.SetDefaults(self.base_name, parameters.extract_nestvalid_dict(model_dict, param_type='device'))

            model_name = self.base_name

            # TODO: Test topology
            if initializer.topology:
                tp_dict = initializer.topology_dict[nn]
                gen_layer = tp.CreateLayer(tp_dict)
                self.layer_gid = gen_layer
                self.gids.append(nest.GetLeaves(gen_layer)[0])
            else:
                self.gids.append(nest.Create(model_name))

        if input_signal is not None:
            self.update_state(input_signal)

    def update_state(self, signal, prev_signal_ids=None, ids_to_update=None):
        """
        For online generation, the input signal is given iteratively
        and the state of the NEST generator objects needs to be updated.
        :param signal: (SpikeList, InputSignal, InputNoise, AnalogSignalList) object containing
                       the values (e.g., spike times) to update the generators with
        :param prev_signal_ids: [list] ids of signals that were active in the previous timestep and need to be reset now
        :param ids_to_update: [list] ids of signals that will be active in the current (next)
                              timestep and must be updated
        :return:
        """
        # for a SpikeList object, update the corresponding NEST spike generators
        if isinstance(signal, signals.SpikeList):
            rounding_precision = signals.determine_decimal_digits(signal.raw_data()[:, 0][0])
            for nn in signal.id_list:
                spike_times = [round(n, rounding_precision) for n in signal[nn].spike_times]  # to be sure
                nest.SetStatus(self.gids[nn], {'spike_times': spike_times})
        else:
            # check for proper signal format
            if isinstance(signal, InputSignal):
                signal = signal.input_signal
            elif isinstance(signal, InputNoise):
                signal = signal.noise_signal
            else:
                assert isinstance(signal, signals.AnalogSignalList), "Incorrect signal format!"

            if self.input_dimension != len(signal):
                self.input_dimension = len(signal)

            # return here model doesn't requires update to save time
            if self.model != 'step_current_generator' and self.model != 'inhomogeneous_poisson_generator':
                return

            # check if inhomogeneous_poisson_generator required and implemented in NEST
            if self.model == 'inhomogeneous_poisson_generator' and \
                    'inhomogeneous_poisson_generator' not in nest.Models():
                raise NotImplementedError("Inhomogeneous Poisson Generator not supported by NEST version!")

            if ids_to_update is not None:
                assert isinstance(ids_to_update, list), "Signal ids to update in generators must be given in list!"
                assert isinstance(prev_signal_ids, list) or prev_signal_ids is None, \
                    "Signal ids to update in generators must be given in list!"

                # in some cases, the same signal id might be updated twice, but this is not an issue
                if prev_signal_ids is not None:
                    ids_to_update += prev_signal_ids

                for nn in ids_to_update:
                    t_axis = signal.time_axis()
                    s_data = signal[nn].raw_data()
                    if len(t_axis) != len(s_data):
                        t_axis = t_axis[:-1]

                    if self.model == 'step_current_generator':
                        nest.SetStatus(self.gids[nn], {'amplitude_times': t_axis, 'amplitude_values': s_data})
                    elif self.model == 'inhomogeneous_poisson_generator':
                        nest.SetStatus(self.gids[nn], {'rate_times': t_axis, 'rate_values': s_data})
            else:
                for nn in range(self.input_dimension):
                    t_axis = signal.time_axis()
                    s_data = signal[nn].raw_data()
                    if len(t_axis) != len(s_data):
                        t_axis = t_axis[:-1]

                    if self.model == 'step_current_generator':
                        nest.SetStatus(self.gids[nn], {'amplitude_times': t_axis, 'amplitude_values': s_data})
                    elif self.model == 'inhomogeneous_poisson_generator':
                        nest.SetStatus(self.gids[nn], {'rate_times': t_axis, 'rate_values': s_data})


########################################################################################################################
class EncodingLayer:
    """
    Wrapper for all the encoders and generators involved in the input conversion process
    """
    # TODO create all instances using a prng; this means prng can't remain None... throw an error here?
    def __init__(self, initializer=None, signal=None, stim_seq=None, online=False, prng=None):
        """

        :param initializer: ParameterSet or dictionary with all parameters for Encoders and Generators
        :param signal: signal object to be converted
        :param stim_seq: [list of str] - stimulus sequence to be converted (only if input is a fixed spatiotemporal
        spike sequence)
        :param prng: random number generator for reproducibility (numpy.random)
        """
        self.prng = prng
        self.total_delay = 0.
        if initializer is not None:
            if isinstance(initializer, dict):
                initializer = parameters.ParameterSet(initializer)
            if not isinstance(initializer, parameters.ParameterSet):
                logger.info("Please provide the encoding parameters as dict or ParameterSet")
            if hasattr(initializer, 'encoder'):
                # if encoder exists, it is a population object
                self.n_encoders = initializer.encoder.N
                self.encoder_names = initializer.encoder.labels
            self.parameters = initializer
        else:
            self.n_encoders = 0
            self.encoder_names = []
            self.parameters = None

        self.n_generators = []
        if signal is not None and not online:
            if isinstance(signal, InputSignal):
                # generators convert the input signal
                self.n_generators.append(signal.dimensions)
            elif isinstance(signal, InputNoise):
                self.n_generators.append(signal.N)
            elif isinstance(signal, signals.AnalogSignalList) or isinstance(signal, signals.SpikeList):
                self.n_generators.append(len(signal))
            else:
                raise TypeError("signal must be InputSignal, AnalogSignalList or SpikeList")
        elif signal is not None and online:
            self.n_generators.append(signal.dimensions)
        elif stim_seq is not None:
            seq = list(signals.iterate_obj_list(stim_seq))
            assert isinstance(seq[0], str), "Provided stim_seq must be string sequence"
            self.n_generators.append(len(np.unique(seq)))
        else:
            self.n_generators.append(0)

        self.generator_names = []
        self.encoding_layer_gid = None
        self.generators = []
        self.encoders = []
        self.connections = []
        self.connection_types = []
        self.synaptic_weights = {}
        self.connection_delays = {}
        self.signal = signal

        def create_encoders(encoder_pars, signal=None):
            """
            Create the encoding layer
            :param encoder_pars: ParameterSet or dictionary with the encoder parameters
            :param signal: InputSignal object
            :return encoders, encoder_labels:
            """
            if isinstance(encoder_pars, dict):
                encoder_pars = parameters.ParameterSet(encoder_pars)
            assert isinstance(encoder_pars, parameters.ParameterSet), "Parameters must be provided as dictionary or ParameterSet"
            if signal is not None:
                assert (isinstance(signal, InputSignal) or isinstance(signal, InputNoise)), "Encoder Input must be InputSignal"

            encoder_labels = []
            encoders = []
            st_gen = StochasticGenerator(self.prng)
            for nn in range(encoder_pars.N):
                encoder_labels.append(encoder_pars.labels[nn])
                enc_pop_dict = {'pop_names': encoder_pars.labels[nn],
                                'n_neurons': encoder_pars.n_neurons[nn],
                                'topology': encoder_pars.topology[nn],
                                'topology_dict': encoder_pars.topology_dict[nn],
                                'gids': None,
                                'layer_gid': None,
                                'is_subpop': False}
                if hasattr(st_gen, encoder_pars.models[nn]):
                    enc_dict = parameters.copy_dict(enc_pop_dict, {})
                    encoders.append(Encoder(parameters.ParameterSet(enc_dict)))
                    logger.info("- {0} []".format(encoder_pars.models[nn]))

                elif encoder_pars.models[nn] == 'NEF':
                    neuron_dict = encoder_pars.neuron_pars[nn]
                    nest.CopyModel(neuron_dict['model'], encoder_pars.labels[nn])
                    nest.SetDefaults(encoder_pars.labels[nn], parameters.extract_nestvalid_dict(neuron_dict,
                                                                                                param_type='neuron'))

                    if encoder_pars.topology[nn]:
                        tp_dict = encoder_pars.topology_dict[nn]
                        tp_dict.update({'elements': encoder_pars.labels[nn]})
                        layer = tp.CreateLayer(tp_dict)
                        gids = nest.GetLeaves(layer)[0]
                        enc_dict = parameters.copy_dict(enc_pop_dict, {'gids': gids, 'layer_gid': layer})
                        encoders.append(Encoder(parameters.ParameterSet(enc_dict)))
                    else:
                        gids = nest.Create(encoder_pars.labels[nn], n=int(encoder_pars.n_neurons[nn]))
                        enc_dict = parameters.copy_dict(enc_pop_dict, {'gids': gids})
                        encoders.append(Encoder(parameters.ParameterSet(enc_dict)))

                    logger.info("- {0} Population of {1} neurons [{2}-{3}]".format(encoder_pars.models[nn],
                                                                             str(encoder_pars.n_neurons), str(min(gids)),
                                                                             str(max(gids))))
                    if encoder_pars.record_spikes[nn]:
                        if encoder_pars.spike_device_pars[nn].has_key('label'):
                            label = encoder_pars.spike_device_pars[nn]['label']
                        else:
                            label = 'encoder_spikes'
                        dev_gid = encoders[-1].record_spikes(parameters.extract_nestvalid_dict(
                            encoder_pars.spike_device_pars[nn], param_type='device'), label=label)
                        logger.info("- Connecting %s to %s, with label %s and id %s" % (
                            encoder_pars.spike_device_pars[nn]['model'],
                            encoders[-1].name, label, str(dev_gid)))

                    if encoder_pars.record_analogs[nn]:
                        if encoder_pars.analogue_device_pars[nn].has_key('label'):
                            label = encoder_pars.analogue_device_pars[nn]['label']
                        else:
                            label = 'encoder_analogs'
                        dev_gid = encoders[-1].record_analog(parameters.extract_nestvalid_dict(
                            encoder_pars.analogue_device_pars[nn], param_type='device'), label=label)
                        logger.info("- Connecting %s to %s, with label %s and id %s" % (
                            encoder_pars.analogue_device_pars[nn]['model'],
                            encoders[-1].name, label, str(dev_gid)))
                # TODO randomize initial variables...
                elif encoder_pars.models[nn] == 'parrot_neuron':
                    neuron_dict = encoder_pars.neuron_pars[nn]
                    nest.CopyModel(neuron_dict['model'], encoder_pars.labels[nn])
                    nest.SetDefaults(encoder_pars.labels[nn], parameters.extract_nestvalid_dict(neuron_dict,
                                                                                                param_type='neuron'))
                    if encoder_pars.topology[nn]:
                        tp_dict = encoder_pars.topology_dict[nn]
                        tp_dict.update({'elements': encoder_pars.labels[nn]})
                        layer = tp.CreateLayer(tp_dict)
                        gids = nest.GetLeaves(layer)[0]
                        enc_dict = parameters.copy_dict(enc_pop_dict, {'gids': gids, 'layer_gid': layer})
                        encoders.append(Encoder(parameters.ParameterSet(enc_dict)))
                    else:
                        gids = nest.Create(encoder_pars.labels[nn], n=int(encoder_pars.n_neurons[nn]))
                        enc_dict = parameters.copy_dict(enc_pop_dict, {'gids': gids})
                        encoders.append(Encoder(parameters.ParameterSet(enc_dict)))

                    logger.info("- {0} Population of {1} neurons [{2}-{3}]".format(encoder_pars.models[nn],
                                                                             str(encoder_pars.n_neurons[nn]),
                                                                             str(min(gids)),
                                                                             str(max(gids))))
                    if encoder_pars.record_spikes[nn]:
                        if encoder_pars.spike_device_pars[nn].has_key('label'):
                            label = encoder_pars.spike_device_pars[nn]['label']
                        else:
                            label = 'encoder_spikes'
                        dev_gid = encoders[-1].record_spikes(parameters.extract_nestvalid_dict(
                            encoder_pars.spike_device_pars[nn], param_type='device'), label=label)
                        logger.info("- Connecting %s to %s, with label %s and id %s" % (
                            encoder_pars.spike_device_pars[nn]['model'],
                            encoders[-1].name, label, str(dev_gid)))

                    if encoder_pars.record_analogs[nn]:
                        raise TypeError("Cannot record analogs from parrot neuron")
                else:
                    logger.warning("Not implemented yet!")

            return encoders, encoder_labels

        def create_generators(encoding_pars, signal_=None, input_dim=None):
            """
            Creates all necessary generator objects.
            :param encoding_pars: global encoding parameters
            :param signal_:
            :param input_dim:
            :return:
            """
            logger.info("\nCreating Generators: ")
            pars 			 = encoding_pars.generator
            generators 		 = []
            generator_labels = []

            for n in range(pars.N):
                # assess where the input to this generator comes from
                tmp = list(signals.iterate_obj_list(encoding_pars.connectivity.connections))
                targets = [x[0] for x in tmp]

                if pars.labels[n] in targets:
                    idx = targets.index(pars.labels[n])
                    source = tmp[idx][1]
                    if source in encoding_pars.encoder.labels:
                        src_idx = encoding_pars.encoder.labels.index(source)
                        tmp_input_dims = encoding_pars.encoder.n_neurons[src_idx]
                    else:
                        tmp_input_dims = None  # will be assessed from the signal
                # TODO list of possible generator types instead of label based
                elif pars.labels[n] == 'spike_pattern':
                    tmp_input_dims = encoding_pars.encoder.n_neurons[0]
                elif pars.labels[n] == 'X_noise' or 'noise' in pars.labels[n]:
                    tmp_input_dims = 1
                elif input_dim is not None:
                    tmp_input_dims = input_dim
                else:
                    tmp_input_dims = None

                # create a specific parameter set for each generator
                gen_pars_dict = {'label': pars.labels[n],
                                 'model': pars.models[n],
                                 'model_pars': pars.model_pars[n],
                                 'topology': pars.topology[n],
                                 'topology_pars': pars.topology_pars[n]}
                gen = Generator(gen_pars_dict, signal_, dims=tmp_input_dims)
                generators.append(gen)
                generator_labels.append(gen.name)
                logger.info("- {0} [{1}-{2}]".format(pars.labels[n], str(min(gen.gids)), str(max(gen.gids))))

            return generators, generator_labels

        ################################################################################
        if online:
            if hasattr(initializer, 'encoder') and initializer.encoder.N:
                self.encoders, self.encoder_names = create_encoders(initializer.encoder)
            if hasattr(initializer, 'generator') and signal is not None:
                self.generators, self.generator_names = create_generators(initializer,
                                                                          signal_=None, input_dim=signal.dimensions)
            elif hasattr(initializer, 'generator') and signal is None:
                self.generators, self.generator_names = create_generators(initializer)
        else:
            if hasattr(initializer, 'encoder'):
                self.encoders, self.encoder_names = create_encoders(initializer.encoder, signal)
            if hasattr(initializer, 'generator'):
                self.generators, self.generator_names = create_generators(initializer, signal)

    def _get_connection_type_and_parameters(self, encoding_pars, conn_pars, idx, populations, pop_objs):
        """
        Determine the type of the connecting populations and their parameters.
        :param encoding_pars:
        :param conn_pars:
        :param idx:
        :param populations:
        :param pop_objs:
        :return:
        """
        src_name = conn_pars.connections[idx][1]
        tget_name = conn_pars.connections[idx][0]

        if hasattr(encoding_pars, 'encoder') and (src_name in encoding_pars.encoder.labels):
            src_id = encoding_pars.encoder.labels.index(src_name)
            src_dims = self.encoders[src_id].size
            if conn_pars.topology_dependent[idx]:
                src_gids = self.encoders[src_id].layer_gid
            else:
                src_gids = self.encoders[src_id].gids
            src_tp = self.encoders[src_id].topology
        elif hasattr(encoding_pars, 'generator') and (src_name in encoding_pars.generator.labels):
            src_id = encoding_pars.generator.labels.index(src_name)
            src_dims = self.generators[src_id].input_dimension
            if conn_pars.topology_dependent[idx]:
                src_gids = self.generators[src_id].layer_gid
            else:
                src_gids = self.generators[src_id].gids
            src_tp = self.generators[src_id].topology
        else:
            raise TypeError("Source Population Label is not specified")

        if hasattr(encoding_pars, 'encoder') and (tget_name in encoding_pars.encoder.labels):
            tget_type = 'encoder'
            tget_id = encoding_pars.encoder.labels.index(tget_name)
            tget_dims = self.encoders[tget_id].size
            if conn_pars.topology_dependent[idx]:
                tget_gids = self.encoders[tget_id].layer_gid
            else:
                tget_gids = self.encoders[tget_id].gids
            tget_tp = self.encoders[tget_id].topology
        elif hasattr(encoding_pars, 'generator') and (tget_name in encoding_pars.generator.labels):
            tget_type = 'generator'
            tget_id = encoding_pars.generator.labels.index(tget_name)
            tget_dims = self.generators[tget_id].input_dimension
            if conn_pars.topology_dependent[idx]:
                tget_gids = self.generators[tget_id].layer_gid
            else:
                tget_gids = self.generators[tget_id].gids
            tget_tp = self.generators[tget_id].topology
        elif tget_name in populations:
            tget_type = 'population'
            tget_id = populations.index(tget_name)
            tget_dims = pop_objs[tget_id].size
            if conn_pars.topology_dependent[idx]:
                tget_gids = pop_objs[tget_id].layer_gid
            else:
                tget_gids = pop_objs[tget_id].gids
            tget_tp = pop_objs[tget_id].topology
        else:
            raise TypeError("Target Population Label is not specified or is not among the "
                            "populations: {0}".format(tget_name))

        return src_gids, src_id, src_dims, src_tp, tget_gids, tget_id, tget_dims, tget_tp

    def connect(self, encoding_pars, net_obj=None, progress=True):
        """
        Connect the generators to their target populations according to specifications
        :param encoding_pars: ParameterSet or dictionary
        :param net_obj: Network object containing the populations to connect to
        """
        logger.info("\nConnecting Encoding Layer: ")

        if isinstance(encoding_pars, dict):
            encoding_pars = parameters.ParameterSet(encoding_pars)
        assert isinstance(encoding_pars, parameters.ParameterSet), "Parameters must be dictionary or ParameterSet"
        if net_obj is not None:
            assert isinstance(net_obj, net_architect.Network), "Please provide the network object to connect to"
        else:
            logger.info("No network object provided, cannot connect input to network")

        conn_pars 	= encoding_pars.connectivity
        populations = list(signals.iterate_obj_list(net_obj.population_names))
        pop_objs 	= list(signals.iterate_obj_list(net_obj.populations))

        if not signals.empty(net_obj.merged_populations):
            merg_pop_names = [x.name for x in net_obj.merged_populations]
            merg_pops = [x for x in net_obj.merged_populations]
            populations.extend(merg_pop_names)
            pop_objs.extend(merg_pops)

        # iterate through all connections
        for idx, nn in enumerate(conn_pars.connections):
            src_name = nn[1]
            tget_name = nn[0]
            # determine the type of the connecting populations and their parameters
            try:
                src_gids, src_id, src_dims, src_tp, tget_gids, tget_id, tget_dims, tget_tp = \
                    self._get_connection_type_and_parameters(encoding_pars, conn_pars, idx, populations, pop_objs)
            except NameError:  # we just continue, nothing special or wrong here
                continue

            if "synapse_name" in conn_pars and conn_pars.synapse_name[idx] is not None:
                synapse_name = conn_pars.synapse_name[idx]
            else:
                synapse_name = src_name + '_' + tget_name

            self.connection_types.append(synapse_name)
            self.connections.append((tget_name, src_name))

            if conn_pars.models[idx] is not None and synapse_name not in nest.Models():
                nest.CopyModel(conn_pars.models[idx], synapse_name)
                nest.SetDefaults(synapse_name, conn_pars.model_pars[idx])

            if synapse_name.find('copy') > 0:  # re-connect the same neurons...
                start = time.time()
                logger.info("    - Connecting {0} (*)".format(synapse_name))

                device_models = ['spike_detector', 'spike_generator', 'multimeter']
                target_synapse_name = synapse_name[:synapse_name.find('copy')-1]
                conns = nest.GetConnections(synapse_model=target_synapse_name)

                iterate_steps = 100
                its = np.arange(0, len(conns)+1, iterate_steps).astype(int)

                for nnn, it in enumerate(its):
                    if nnn < len(its)-1:
                        con = conns[it:its[nnn+1]]
                        st = nest.GetStatus(con)
                        source_gids = [x['source'] for x in st if
                                       nest.GetStatus([x['target']])[0]['model'] not in device_models]
                        target_gids = [x['target'] for x in st if
                                       nest.GetStatus([x['target']])[0]['model'] not in device_models]
                        syn_dict = parameters.copy_dict(conn_pars.syn_specs[idx], {'model': synapse_name, 'weight':
                            conn_pars.weight_dist[idx], 'delay': conn_pars.delay_dist[idx]})

                        syn_dicts = [{'synapsemodel': list(np.repeat(synapse_name, len(source_gids)))[iddx],
                                      'source': source_gids[iddx],
                                      'target': target_gids[iddx],
                                      'weight': syn_dict['weight'],		              # TODO distributions??
                                      'delay': syn_dict['delay'],
                                      'receptor_type': syn_dict['receptor_type']} for iddx in range(len(target_gids))]
                        nest.DataConnect(syn_dicts)
                    if progress:
                        visualization.progress_bar(float(nnn) / float(len(its)))
                logger.info("\tElapsed time: {0} s".format(str(time.time()-start)))
            else:
                ## Set up connections
                # 1) pre-computed weight matrix, or function to specify w
                if conn_pars.preset_W[idx] is not None:
                    if isinstance(conn_pars.preset_W[idx], np.ndarray):
                        w = conn_pars.preset_W[idx]
                    elif isinstance(conn_pars.preset_W[idx], tuple):
                        args = parameters.copy_dict(conn_pars.preset_W[idx][1], {'size': (int(src_dims), int(tget_dims))})
                        w = conn_pars.preset_W[idx][0](**args)
                    elif isinstance(conn_pars.preset_W[idx], str):
                        w = np.load(conn_pars.preset_W[idx])
                    else:
                        raise ValueError("Incorrect W specifications")

                    logger.info("- Connecting {0} to population {1}".format(src_name, tget_name))

                    for pre_syn_matidx, pre_syn_gid in enumerate(src_gids):
                        post_syn_matidx = w[pre_syn_matidx, :].nonzero()[0]
                        post_syn_gid = list(post_syn_matidx + min(tget_gids))
                        weights = [w[pre_syn_matidx, x] for x in post_syn_matidx]

                        if isinstance(conn_pars.delay_dist[idx], dict):
                            delay = conn_pars.delay_dist[idx]
                        elif isinstance(conn_pars.delay_dist[idx], float) or \
                                isinstance(conn_pars.delay_dist[idx], int):
                            delay = np.repeat(conn_pars.delay_dist[idx], len(weights))
                        elif isinstance(conn_pars.delay_dist[idx], tuple):
                            args = parameters.copy_dict(conn_pars.delay_dist[idx][1], {'size': len(weights)})
                            delay = conn_pars.delay_dist[idx][0](**args)
                        else:
                            raise TypeError("Delays not provided in correct format")
                        visualization.progress_bar(float(pre_syn_matidx) / len(src_gids))
                        conn_dict = conn_pars.conn_specs[idx]
                        for idxx, tget in enumerate(post_syn_gid):
                            if isinstance(delay, dict):
                                d = delay
                            else:
                                d = delay[idxx]

                            syn_dict = parameters.copy_dict(conn_pars.syn_specs[idx], {'model': synapse_name,
                                                                                       'weight': weights[idxx],
                                                                                       'delay': d})

                            if conn_dict is not None:
                                nest.Connect(pre_syn_gid, [tget], conn_spec=conn_dict, syn_spec=syn_dict)
                            else:
                                nest.Connect(pre_syn_gid, [tget], 'all_to_all', syn_dict)
                    logger.info("- Connecting {0} to population {1}".format(src_name, tget_name))

                # topology connections only allowed between generator (with specified topology) and population
                elif conn_pars.topology_dependent[idx]:
                    assert (src_tp and tget_tp), "Topological connections are only possible if both pre- and post- " \
                                                 "population are topologically defined"

                    tp_dict = parameters.copy_dict(conn_pars.conn_specs[idx], {'synapse_model': synapse_name})
                    tp.ConnectLayers(src_gids, tget_gids, tp_dict)

                    logger.info("- Connecting Layer {0} to Layer {1}".format(src_name, tget_name))

                else:
                    # add extra numerical index to the src_name
                    if ((src_name + '0' in nest.Models()) and (tget_name in nest.Models())) \
                            or ((src_name in nest.Models()) and (tget_name in nest.Models())) \
                            or ((src_name + '0' in nest.Models()) and
                                (nest.GetStatus([tget_gids[0]])[0]['element_type'] == 'neuron')) \
                            or ((src_name in nest.Models()) and
                                (nest.GetStatus([tget_gids[0]])[0]['element_type'] == 'neuron')):

                        # need to use synapse with the same model for all connections from a device
                        if "synapse_name" in conn_pars and conn_pars.synapse_name[idx] is not None:
                            synapse_name = conn_pars.synapse_name[idx]
                        else:
                            synapse_name = src_name + '_' + tget_name
                        nest.SetDefaults(conn_pars.models[idx], conn_pars.model_pars[idx])
                        syn_dict = parameters.copy_dict(conn_pars.syn_specs[idx], {'model': synapse_name, 'weight':
                            conn_pars.weight_dist[idx], 'delay': conn_pars.delay_dist[idx]})
                        conn_dict = conn_pars.conn_specs[idx]
                        if isinstance(src_gids, list):
                            src_gids = tuple(x[0] for x in src_gids)
                        nest.Connect(src_gids, tget_gids, conn_spec=conn_dict, syn_spec=syn_dict)
                        logger.info("- Connecting {0} to population {1} [{2}]".format(src_name, tget_name, synapse_name))
                    else:
                        x = StochasticGenerator(self.prng)
                        if hasattr(x, encoding_pars.encoder.models[src_id]):
                            # src is a stochastic encoder... set spike times for spike generator
                            act_list = self.encoders[src_id].spiking_activity
                            if isinstance(tget_gids, list):
                                tget_gids = tuple(x[0] for x in tget_gids)

                            dt = nest.GetKernelStatus()['resolution']
                            d = decimal.Decimal(str(dt)).as_tuple()[1][0]
                            for idxx, nnn in enumerate(act_list.id_list):
                                spk_times = act_list.spiketrains[nnn].spike_times
                                spk_times[np.where(spk_times == 0.)] = dt
                                spk_t = [round(tt, d) for tt in spk_times]
                                nest.SetStatus([tget_gids[idxx]], {'spike_times': spk_t})

                            logger.info("- Connecting {0} to generator {1}".format(src_name, tget_name))

                        elif (src_name in nest.Models()) and (tget_name in nest.Models()):
                            if "synapse_name" in conn_pars and conn_pars.synapse_name[idx] is not None:
                                synapse_name = conn_pars.synapse_name[idx]
                            else:
                                synapse_name = src_name + '_' + tget_name
                            nest.SetDefaults(conn_pars.models[idx], conn_pars.model_pars[idx])
                            syn_dict = parameters.copy_dict(conn_pars.syn_specs[idx], {'model': synapse_name, 'weight':
                                conn_pars.weight_dist[idx], 'delay': conn_pars.delay_dist[idx]})
                            conn_dict = conn_pars.conn_specs[idx]
                            if isinstance(src_gids, list):
                                src_gids = tuple(x[0] for x in src_gids)
                            nest.Connect(src_gids, tget_gids, conn_spec=conn_dict, syn_spec=syn_dict)
                            logger.info("- Connecting {0} to population {1} [{2}]".format(src_name, tget_name, synapse_name))
        self.signal = None

    # def connect_clone(self, encoding_pars, network=None, clone=None): TODO safely remove!
    # 	"""
    # 	Replicate EncodingLayer so that both network and clone network receive the exact same input
    # 	:param encoding_pars:
    # 	:param network:
    # 	:param clone:
    # 	:return:
    # 	"""
    # 	native_population_names = list(signals.iterate_obj_list(network.population_names))
    # 	native_populations = list(signals.iterate_obj_list(network.populations))
    # 	target_populations = [n[0] for n in encoding_pars.connectivity.connections if n[0] in native_population_names]
    # 	target_populations.extend([n+'_clone' for n in target_populations])
    #
    # 	if hasattr(encoding_pars, "encoder") and encoding_pars.encoder.N:
    # 		encoding_pars.encoder.labels.extend(['{0}_parrots'.format(n) for n in
    # 		                                                    native_population_names if n in target_populations])
    # 		new_encoders = encoding_pars.encoder.labels
    # 		encoder_size = encoding_pars.encoder.n_neurons.extend([n.size for idx, n in enumerate(
    # 				native_populations) if native_population_names[idx] in target_populations])
    # 		encoder_pars = encoding_pars.encoder.neuron_pars.extend([{'model': 'parrot_neuron'} for n in native_population_names if n in target_populations])
    # 		enc_models = encoding_pars.encoder.models.extend(['parrots' for _ in range(len(new_encoders))])
    # 		enc_model_pars = encoding_pars.encoder.model_pars.extend([{} for _ in range(len(new_encoders))])
    # 		topologies = encoding_pars.encoder.topology.extend([False for _ in range(len(new_encoders))])
    # 		tp_dicts = encoding_pars.encoder.topology_dict.extend([None for _ in range(len(new_encoders))])
    # 		rc_spikes = encoding_pars.encoder.record_spikes.extend([False for _ in range(len(new_encoders))])
    # 		spk_dvc_pars = encoding_pars.encoder.spike_device_pars.extend([None for _ in range(len(new_encoders))])
    # 		rc_analogs = encoding_pars.encoder.record_analogs.extend([False for _ in range(len(new_encoders))])
    # 		analog_dev_pars = encoding_pars.encoder.analog_device_pars.extend([None for _ in range(len(new_encoders))])
    # 	else:
    # 		new_encoders = ['{0}_parrots'.format(n) for n in native_population_names if n in target_populations]
    # 		encoder_size = [n.size for n in native_populations]
    # 		encoder_pars = [{'model': 'parrot_neuron'} for n in native_population_names if n in target_populations]
    # 		enc_models = ['NEF' for _ in range(len(new_encoders))]
    # 		enc_model_pars = [{} for _ in range(len(new_encoders))]
    # 		topologies = [False for _ in range(len(new_encoders))]
    # 		tp_dicts = [None for _ in range(len(new_encoders))]
    # 		rc_spikes = [False for _ in range(len(new_encoders))]
    # 		spk_dvc_pars = [None for _ in range(len(new_encoders))]
    # 		rc_analogs = [False for _ in range(len(new_encoders))]
    # 		analog_dev_pars = [None for _ in range(len(new_encoders))]
    #
    # 	connections = []
    # 	connections_clone = []
    # 	conn_specs = []
    # 	conn_specs_clone = []
    # 	syn_names = []
    # 	syn_names_clone = []
    # 	syn_specs = []
    # 	syn_specs_clone = []
    # 	models = []
    # 	models_clone = []
    # 	model_pars = []
    # 	model_pars_clone = []
    # 	weights = []
    # 	weights_clone = []
    # 	delays = []
    # 	delays_clone = []
    # 	pre_w = []
    # 	pre_w_clone = []
    # 	for idx, n_connection in enumerate(encoding_pars.connectivity.connections):
    # 		if n_connection[0] in native_population_names:
    # 			connections.append(('{0}'.format(str(n_connection[0]+'_parrots')), '{0}'.format(str(n_connection[1]))))
    # 			#connections_clone.append(('{0}'.format(str(n_connection[0]+'_parrots')), '{0}'.format(str(
    # 			# n_connection[1]))))
    # 			conn_specs.append({'rule': 'all_to_all'})
    # 			syn_names.append('{0}Parrots'.format(str(n_connection[1])))
    # 			syn_specs.append({})
    # 			models.append('static_synapse')
    # 			model_pars.append({})
    # 			weights.append(1.)
    # 			delays.append(0.1)
    # 			pre_w.append(None)
    #
    # 			connections.append(('{0}'.format(str(n_connection[0])), '{0}'.format(str(n_connection[0]+'_parrots'))))
    # 			connections_clone.append(('{0}_clone'.format(str(n_connection[0])), '{0}'.format(str(n_connection[0]+'_parrots'))))
    # 			conn_specs.append({'rule': 'one_to_one'})
    # 			conn_specs_clone.append({'rule': 'one_to_one'})
    # 			syn_names.append(encoding_pars.connectivity.synapse_name[idx])
    # 			syn_names_clone.append(encoding_pars.connectivity.synapse_name[idx])
    # 			syn_specs.append(encoding_pars.connectivity.syn_specs[idx])
    # 			syn_specs_clone.append(encoding_pars.connectivity.syn_specs[idx])
    # 			models.append(encoding_pars.connectivity.models[idx])
    # 			models_clone.append(encoding_pars.connectivity.models[idx])
    # 			model_pars.append(encoding_pars.connectivity.model_pars[idx])
    # 			model_pars_clone.append(encoding_pars.connectivity.model_pars[idx])
    # 			weights.append(encoding_pars.connectivity.weight_dist[idx])
    # 			weights_clone.append(encoding_pars.connectivity.weight_dist[idx])
    # 			delays.append(encoding_pars.connectivity.delay_dist[idx])
    # 			delays_clone.append(encoding_pars.connectivity.delay_dist[idx])
    # 			pre_w.append(encoding_pars.connectivity.preset_W[idx])
    # 			pre_w_clone.append(encoding_pars.connectivity.preset_W[idx])
    # 		else:
    # 			connections.append(n_connection)
    # 			#connections_clone.append(n_connection)
    # 			conn_specs.append(encoding_pars.connectivity.conn_specs[idx])
    # 			syn_names.append(encoding_pars.connectivity.synapse_name[idx])
    # 			syn_specs.append(encoding_pars.connectivity.syn_specs[idx])
    # 			models.append(encoding_pars.connectivity.models[idx])
    # 			model_pars.append(encoding_pars.connectivity.model_pars[idx])
    # 			weights.append(encoding_pars.connectivity.weight_dist[idx])
    # 			delays.append(encoding_pars.connectivity.delay_dist[idx])
    # 			pre_w.append(encoding_pars.connectivity.preset_W[idx])
    #
    # 	tp = [False for _ in range(len(connections))]
    # 	extra_encoder = {
    # 		'N': len(new_encoders),
    # 		'labels': new_encoders,
    # 		'models': enc_models,
    # 		'model_pars': enc_model_pars,
    # 		'n_neurons': encoder_size,
    # 		'neuron_pars': encoder_pars,
    # 		'topology': topologies,
    # 		'topology_dict': tp_dicts,
    # 		'record_spikes': rc_spikes,
    # 		'spike_device_pars': spk_dvc_pars,
    # 		'record_analogs': rc_analogs,
    # 		'analog_device_pars': analog_dev_pars
    # 	}
    # 	extra_connectivity_native = {
    # 			'synapse_name': syn_names,
    # 			'connections': connections,
    # 			'topology_dependent': tp,
    # 			'conn_specs': conn_specs,
    # 			'syn_specs': syn_specs,
    # 			'models': models,
    # 			'model_pars': model_pars,
    # 			'weight_dist': weights,
    # 			'delay_dist': delays,
    # 			'preset_W': pre_w}
    # 	extra_connectivity_clone = {
    # 			'synapse_name': syn_names_clone,
    # 			'connections': connections_clone,
    # 			'topology_dependent': [False for _ in range(len(connections_clone))],
    # 			'conn_specs': conn_specs_clone,
    # 			'syn_specs': syn_specs_clone,
    # 			'models': models_clone,
    # 			'model_pars': model_pars_clone,
    # 			'weight_dist': weights_clone,
    # 			'delay_dist': delays_clone,
    # 			'preset_W': pre_w_clone}
    # 	native_encoding_pars = parameters.ParameterSet({'encoder': extra_encoder, 'generator': encoding_pars.generator.as_dict(),
    # 	                                     'connectivity': extra_connectivity_native})
    # 	clone_encoding_pars = parameters.ParameterSet({'encoder': extra_encoder, 'generator': encoding_pars.generator.as_dict(),
    # 	                                     'connectivity': extra_connectivity_clone})
    # 	self.__init__(native_encoding_pars)
    # 	self.connect(native_encoding_pars, network)
    # 	self.connect(clone_encoding_pars, clone)

    @staticmethod
    def replicate_connections(net, clone, progress=True): # TODO - correct or remove
        """
        Replicate the connectivity from the encoding layer to the clone network
        :param net:
        :param clone:
        :param progress:
        :return:
        """
        start = time.time()
        target_population_names = clone.population_names
        target_population_gids = [n.gids for n in clone.populations]
        source_population_names = [n.split('_')[0] for n in target_population_names]
        source_population_gids = [net.populations[net.population_names.index(n)].gids for n in source_population_names]
        device_models = ['spike_detector', 'multimeter'] # 'spike_generator',

        for idx, n_pop in enumerate(source_population_gids):
            logger.info("\n Replicating Encoding Layer connections to {0}".format(clone.population_names[idx]))
            conns 			= nest.GetConnections(target=n_pop)
            iterate_steps 	= 100
            its 			= np.arange(0, len(conns) + 1, iterate_steps).astype(int)
            if len(its) > 1:
                for nnn, it in enumerate(its):
                    if nnn < len(its) - 1:
                        con = conns[it:its[nnn + 1]]
                        st = nest.GetStatus(con)

                        source_gids = [x['source'] for x in st if nest.GetStatus([x['source']])[0]['model'] not in
                                       device_models]

                        target_gids = [target_population_gids[idx][0] for x in st
                                       if nest.GetStatus([x['source']])[0]['model'] not in device_models]

                        weights 	= [x['weight'] for x in st if nest.GetStatus([x['source']])[0]['model'] not in device_models]
                        delays 		= [x['delay'] for x in st if nest.GetStatus([x['source']])[0]['model'] not in device_models]
                        models 		= [str(x['synapse_model']) for x in st if nest.GetStatus([x['source']])[0]['model'] not in device_models]
                        receptors 	= [x['receptor'] for x in st if nest.GetStatus([x['source']])[0]['model'] not in device_models]
                        syn_dicts 	= [{'synapsemodel': models[iddx],
                                         'source': source_gids[iddx],
                                         'target': target_gids[iddx],
                                         'weight': weights[iddx],
                                         'delay': delays[iddx],
                                         'receptor_type': receptors[iddx]} for iddx in range(len(target_gids))]
                        nest.DataConnect(syn_dicts)
                    if progress:
                        visualization.progress_bar(float(nnn) / float(len(its)))

                logger.info("\tElapsed time: {0} s".format(str(time.time() - start)))
            else:
                st = nest.GetStatus(conns)
                logger.info([nest.GetStatus([x['source']])[0]['model'] for x in st])

                source_gids = [x['source'] for x in st if nest.GetStatus([x['source']])[0]['model'] not in
                               device_models]
                target_gids = [target_population_gids[idx][0] for x in st if nest.GetStatus([x['source']])[0][
                    'model'] not in device_models]
                weights 	= [x['weight'] for x in st if nest.GetStatus([x['source']])[0]['model'] not in device_models]
                delays 		= [x['delay'] for x in st if nest.GetStatus([x['source']])[0]['model'] not in device_models]
                models 		= [str(x['synapse_model']) for x in st if
                                 nest.GetStatus([x['source']])[0]['model'] not in device_models]
                receptors = [x['receptor'] for x in st if
                             nest.GetStatus([x['source']])[0]['model'] not in device_models]

                syn_dicts = [{'synapsemodel': models[iddx],
                              'source': source_gids[iddx],
                              'target': target_gids[iddx],
                              'weight': weights[iddx],
                              'delay': delays[iddx],
                              'receptor_type': receptors[iddx]} for iddx in range(len(target_gids))]
                nest.DataConnect(syn_dicts)
                logger.info("\tElapsed time: {0} s".format(str(time.time() - start)))

    def connect_decoders(self, decoding_pars):
        """
        Connect decoders to encoding population
        :param decoding_pars:
        :return:
        """
        if isinstance(decoding_pars, dict):
            decoding_pars = parameters.ParameterSet(decoding_pars)
        assert isinstance(decoding_pars, parameters.ParameterSet), "DecodingLayer must be initialized with ParameterSet or " \
                                                                   "dictionary"
        decoder_params = {}
        # initialize state extractors:
        if hasattr(decoding_pars, "state_extractor"):
            pars_st = decoding_pars.state_extractor
            logger.info("\nConnecting Decoders: ")
            for ext_idx, n_src in enumerate(pars_st.source_population):
                assert(n_src in self.encoder_names), "State extractor must connect to encoder population"
                if n_src in self.encoder_names:
                    pop_index = self.encoder_names.index(n_src)
                    src_obj = self.encoders[pop_index]
                else:
                    raise TypeError("No source populations in Encoding Layer")

                decoder_params.update({'state_variable': pars_st.state_variable,
                                       'state_specs': pars_st.state_specs,
                                       'reset_states': pars_st.reset_states,
                                       'average_states': pars_st.average_states,
                                       'standardize': pars_st.standardize})

                if hasattr(decoding_pars, "readout"):
                    pars_readout = decoding_pars.readout
                    assert(len(decoding_pars.readout) == decoding_pars.state_extractor.N), "Specify one readout dictionary " \
                                                                                           "per state extractor"
                    decoder_params.update({'readout': pars_readout})
                src_obj.connect_decoders(parameters.ParameterSet(decoder_params))
        else:
            raise IOError("DecodingLayer requires the specification of state extractors")

    def extract_synaptic_weights(self, src_gids=None, tget_gids=None, syn_name=None, progress=True):
        """
        Determine the connection weights between src_gids and tget_gids for synapses syn_name
        :param src_gids:
        :param tget_gids:
        :param syn_name:
        :param progress:
        :return:
        """
        # TODO remove the use of 'copy' keyword (this is now done with the multiport synapse in NEST)
        if src_gids is None and tget_gids is None:
            for con in list(np.unique(self.connection_types)):
                if con[-4:] != 'copy':
                    status_dict = nest.GetStatus(nest.GetConnections(synapse_model=con))
                    src_gids = [status_dict[n]['source'] for n in range(len(status_dict))]
                    tget_gids = [status_dict[n]['target'] for n in range(len(status_dict))]
                    self.synaptic_weights.update({con: net_architect.extract_weights_matrix(list(np.unique(src_gids)),
                                                                                            list(np.unique(tget_gids)),
                                                                                            progress=progress)})

        elif src_gids and tget_gids:
            if syn_name is None:
                syn_name = str(nest.GetStatus(nest.GetConnections([src_gids[0]], [tget_gids[0]]))[0]['synapse_model'])
            self.synaptic_weights.update({syn_name: net_architect.extract_weights_matrix(list(np.unique(src_gids)),
                                                                                         list(np.unique(tget_gids)),
                                                                                         progress=progress)})
        else:
            logger.warning("Provide gids!!")

    def extract_synaptic_delays(self, src_gids=None, tget_gids=None, syn_name=None, progress=True):
        """
        Determine the synaptic delays between src_gids and tget_gids for synapses syn_name
        :param src_gids: source NEST ids
        :param tget_gids: target NEST ids
        :param syn_name: name of synapse
        :param progress: show progress bar
        :return:
        """
        if src_gids is None:
            for con in list(np.unique(self.connection_types)):
                status_dict = nest.GetStatus(nest.GetConnections(synapse_model=con))
                src_gids = [status_dict[n]['source'] for n in range(len(status_dict))]
                tget_gids = [status_dict[n]['target'] for n in range(len(status_dict))]
                self.connection_delays.update({con: net_architect.extract_delays_matrix(list(np.unique(src_gids)),
                                                                                        list(np.unique(tget_gids)),
                                                                                        progress=progress)})
        elif src_gids and tget_gids:
            if syn_name is None:
                syn_name = str(nest.GetStatus(nest.GetConnections([src_gids[0]], [tget_gids[0]]))[0]['synapse_model'])
            self.connection_delays.update({syn_name: net_architect.extract_delays_matrix(list(np.unique(src_gids)),
                                                                                         list(np.unique(tget_gids)),
                                                                                         progress=progress)})
        else:
            logger.warning("Provide gids!!")

    def extract_encoder_activity(self, t_start=None, t_stop=None):
        """
        Read the spiking activity of the encoders present in the EncodingLayer
        :return:
        """
        if not signals.empty(self.encoders):
            logger.info("\nExtracting and storing recorded activity from encoders: ")
        for n_enc in self.encoders:
            if n_enc.attached_devices:
                logger.info("- Encoder {0}".format(n_enc.name))
                for n_dev in n_enc.attached_devices:
                    if nest.GetStatus(n_dev)[0]['to_memory']:
                        # status = nest.GetStatus(n_dev)[0]['events']
                        # t_start, t_stop = np.min(status['times']), np.max(status['times'])
                        n_enc.load_activity(n_dev, t_start=t_start, t_stop=t_stop)
                    elif nest.GetStatus(n_dev)[0]['to_file']:
                        n_enc.load_activity(list(nest.GetStatus(n_enc)[0]['filenames']), t_start=t_start, t_stop=t_stop)

    def flush_records(self, decoders=False):
        """
        Delete all data from all devices connected to the encoders
        :return:
        """
        if not signals.empty(self.encoders):
            logger.info("\nClearing device data: ")
        for n_enc in self.encoders:
            if n_enc.attached_devices:
                devices = n_enc.attached_device_names
                for idx, n_dev in enumerate(n_enc.attached_devices):
                    logger.info(" - {0} {1}".format(devices[idx], str(n_dev)))
                    nest.SetStatus(n_dev, {'n_events': 0})
                    if nest.GetStatus(n_dev)[0]['to_file']:
                        io.remove_files(nest.GetStatus(n_dev)[0]['filenames'])
        if decoders:
            for n_enc in self.encoders:
                if n_enc.decoding_layer is not None:
                    n_enc.decoding_layer.flush_records()

    def update_state(self, signal, prev_signal_ids=None, ids_to_update=None):
        """
        Update state of encoders and generators. If ids_to_update is given, then only the appropriate generators
        will be updated for efficiency reasons.

        :param signal: [AnalogSignalList] signal used to update state of encoders and generators
        :param prev_signal_ids: [list] with signal ids that were active/updated in the previous step
        :param ids_to_update: [list] with signal ids for which the generators should be updated
        :return:
        """
        for n_gen in self.generators:
            n_gen.update_state(signal, prev_signal_ids, ids_to_update)

    # TODO this could use a bit of rethinking
    def extract_connectivity(self, net, sub_set=False, progress=False):
        """
        Extract encoding layer connections.

        :param net: Network object
        :param sub_set:
        :param progress: display progress bar
        :return:
        """
        n_items = 10
        if sub_set:
            tgets = list(signals.iterate_obj_list([list(signals.iterate_obj_list(n.gids)) for n in
                                                   net.populations]))[:n_items]
            srces = list(itertools.chain(*[n.gids for n in self.generators][0]))[:n_items]
            self.extract_synaptic_weights(srces, tgets, syn_name='Gen_Net', progress=progress)
            self.extract_synaptic_delays(srces, tgets, syn_name='Gen_Net', progress=progress)

        # if there's only one connection type
        elif len(np.unique(self.connection_types)) == 1:
            tgets = list(signals.iterate_obj_list([list(signals.iterate_obj_list(n.gids)) for n in net.populations]))
            srces = list(itertools.chain(*[n.gids for n in self.generators][0]))
            self.extract_synaptic_weights(srces, tgets, syn_name='Gen_Net', progress=progress)
            self.extract_synaptic_delays(srces, tgets, syn_name='Gen_Net', progress=progress)

        elif len(self.connections) != len(np.unique(self.connection_types)):
            for con_idx, n_con in enumerate(self.connections):
                src_name = n_con[1]
                tget_name = n_con[0]
                syn_name = self.connection_types[con_idx] + str(con_idx)
                if src_name in net.population_names:
                    src_gids = net.populations[net.population_names.index(src_name)].gids
                elif src_name in list([n.name for n in net.merged_populations]):
                    merged_populations = list([n.name for n in net.merged_populations])
                    src_gids = net.merged_populations[merged_populations.index(src_name)].gids
                elif src_name in self.encoder_names:
                    src_gids = self.encoders[self.encoder_names.index(src_name)].gids
                else:
                    gen_names = [x.name[0].split('0')[0] for x in self.generators]
                    src_gids = list(itertools.chain(*self.generators[gen_names.index(src_name)].gids))
                if tget_name in net.population_names:
                    tget_gids = net.populations[net.population_names.index(tget_name)].gids
                elif tget_name in list([n.name for n in net.merged_populations]):
                    merged_populations = list([n.name for n in net.merged_populations])
                    tget_gids = net.merged_populations[merged_populations.index(tget_name)].gids
                elif tget_name in self.encoder_names:
                    tget_gids = self.encoders[self.encoder_names.index(tget_name)].gids
                else:
                    gen_names = [x.name[0].split('0')[0] for x in self.generators]
                    tget_gids = list(itertools.chain(*self.generators[gen_names.index(tget_name)].gids))
                self.extract_synaptic_weights(src_gids, tget_gids, syn_name=syn_name, progress=progress)
                self.extract_synaptic_delays(src_gids, tget_gids, syn_name=syn_name, progress=progress)
        else:
            self.extract_synaptic_weights(progress=progress)
            self.extract_synaptic_delays(progress=progress)

    def determine_total_delay(self):
        """
        Determine the connection delays involved in the encoding layer
        :return:
        """
        assert (not signals.empty(self.connection_delays)), "Please run extract_connectivity first..."

        for k, v in self.connection_delays.items():
            delay = np.unique(np.array(v[v.nonzero()].todense()))
            assert (len(delay) == 1), "Heterogeneous delays in encoding layer are not supported.."

        self.total_delay = float(delay)
        logger.info("\n- total delays in EncodingLayer: {0} ms".format(str(self.total_delay)))