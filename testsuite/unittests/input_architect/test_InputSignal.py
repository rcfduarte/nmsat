"""
Test the `InputSignal` class in input_architect.py and all its methods.
"""

import sys
sys.path.append('../../../')
sys.path.append('../')

import numpy as np

# nest.init(sys.argv + ['--verbosity=QUIET', '--quiet'])  # turn off all NEST messages

from modules import parameters
from modules import input_architect as ia
from modules import signals as sg


########################################################################################################################
# Test functions
########################################################################################################################

def test_compress_signals():
	"""
	:return:
	"""
	from datasets import dataset1 as data

	dt 	= 0.1
	amp = [120., 100., 50., 25.]
	dur = 200.
	on 	= 0.0

	input_signals = []

	for sig_idx in range(4):
		off = on + dur
		s = np.ones(int((off - on) / dt)) * amp[sig_idx]
		input_signals.append(sg.AnalogSignal(s, dt, t_start=on, t_stop=off))

	def __validate(signal_):
		assert isinstance(signal_, sg.AnalogSignalList)
		assert signal_.t_start == on
		assert signal_.t_stop == dur
		assert signal_.signal_length == 2000
		assert signal_.analog_signals[0].t_stop == signal_.analog_signals[2].t_stop
		assert np.array_equal(signal_.analog_signals[0].signal, np.ones(2000) * amp[0])
		assert np.array_equal(signal_.analog_signals[3].signal, np.ones(2000) * amp[3])

	#############################
	# call function with argument
	signal = ia.InputSignal(data.input_pars['signal'])
	signal = signal.compress_signals(input_signals)

	# validate results
	__validate(signal)

	################################
	# call function without argument
	signal = ia.InputSignal(data.input_pars['signal'])
	# need to prepare some member variables here which would otherwise be done before the call
	signal.input_signal = input_signals
	signal.time_data = np.arange(on, on + dur, dt)

	signal = signal.compress_signals()

	# validate results
	__validate(signal)


# TODO
def test_generate_iterative():
	pass


# TODO
def test_connect_devices():
	pass


# TODO
def test_connect_populations():
	pass


# TODO
def test_mirror():
	pass

if __name__ == "__main__":
	# test_compress_signals()
	# test_generate_iterative_fast()
	pass