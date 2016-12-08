import itertools
import numpy as np
import nest
import io
from modules import helper

from spike_list import SpikeList
from analog_signal_list import AnalogSignalList

def empty(seq):
	if isinstance(seq, np.ndarray):
		return not bool(seq.size) #seq.any() # seq.data
	elif isinstance(seq, list) and seq:
		if helper.isiterable(seq):
			result = np.mean([empty(n) for n in list(itertools.chain(seq))])
		else:
			result = np.mean([empty(n) for n in list(itertools.chain(*[seq]))])
		if result == 0. or result == 1.:
			return result.astype(bool)
		else:
			return not result.astype(bool)
	else:
		return not seq

def reject_outliers(data, m=2.):
	"""
	Remove outliers from data
	:param data:
	:param m:
	:return:
	"""
	d = np.abs(data - np.median(data))
	mdev = np.median(d)
	s = d/mdev if mdev else 0.
	return data[s < m]


def convert_activity(initializer):
	"""
	Extract recorded activity from devices, convert it to SpikeList or AnalogList
	objects and store them appropriately
	:param initializer: can be a string, or list of strings containing the relevant filenames where the
	raw data was recorded or be a gID for the recording device, if the data is still in memory
	"""
	# TODO: save option!
	# if object is a string, it must be a file name; if it is a list of strings, it must be a list of filenames
	if isinstance(initializer, basestring) or isinstance(initializer, list):
		data = io.extract_data_fromfile(initializer)
		if data is not None:
			if len(data.shape) != 2:
				data = np.reshape(data, (int(len(data) / 2), 2))
			if data.shape[1] == 2:
				spk_times = data[:, 1]
				neuron_ids = data[:, 0]
				tmp = [(neuron_ids[n], spk_times[n]) for n in range(len(spk_times))]
				return SpikeList(tmp, np.unique(neuron_ids).tolist())
			else:
				neuron_ids = data[:, 0]
				times = data[:, 1]
				for nn in range(data.shape[1]):
					if nn > 1:
						signals = data[:, nn]
						tmp = [(neuron_ids[n], signals[n]) for n in range(len(neuron_ids))]
						return AnalogSignalList(tmp, np.unique(neuron_ids).tolist(), times=times)

	elif isinstance(initializer, tuple) or isinstance(initializer, int):
		status = nest.GetStatus(initializer)[0]['events']
		if len(status) == 2:
			spk_times = status['times']
			neuron_ids = status['senders']
			tmp = [(neuron_ids[n], spk_times[n]) for n in range(len(spk_times))]
			return SpikeList(tmp, np.unique(neuron_ids).tolist())
		elif len(status) > 2:
			times = status['times']
			neuron_ids = status['senders']
			idxs = np.argsort(times)
			times = times[idxs]
			neuron_ids = neuron_ids[idxs]
			rem_keys = ['times', 'senders']
			new_dict = {k: v[idxs] for k, v in status.iteritems() if k not in rem_keys}
			analog_activity = []
			for k, v in new_dict.iteritems():
				tmp = [(neuron_ids[n], v[n]) for n in range(len(neuron_ids))]
				analog_activity.append(AnalogSignalList(tmp, np.unique(neuron_ids).tolist(), times=times))
			return analog_activity
	else:
		print "Incorrect initializer..."


##############################################################################
def rescale_list(OldList, NewMin, NewMax):
	NewRange = float(NewMax - NewMin)
	OldMin = min(OldList)
	OldMax = max(OldList)
	OldRange = float(OldMax - OldMin)
	ScaleFactor = NewRange / OldRange
	NewList = []
	for OldValue in OldList:
		NewValue = ((OldValue - OldMin) * ScaleFactor) + NewMin
		NewList.append(NewValue)
	return NewList


#################################################################################
def hammingDistance(s1, s2):
    """
    Return the Hamming distance between equal-length sequences
    (from wikipedia)
    """
    if len(s1) != len(s2):
        raise ValueError("Undefined for sequences of unequal length")
    return sum(bool(ord(ch1) - ord(ch2)) for ch1, ch2 in zip(s1, s2))