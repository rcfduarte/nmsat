__author__ = 'duarte'
import nest
import numpy as np
np.set_printoptions(threshold=np.nan, suppress=True)

nest.ResetKernel()
nest.SetKernelStatus({'resolution': 0.1})
sg = nest.Create('spike_generator', 1, {'precise_times': True})#, {'allow_offgrid_spikes': True})#


start = 1e+06

# Setting all at one doesn't seem to cause any problem (except some of the times are not the same
all_times = np.sort()

# times = np.round(np.arange(1000000., 10000000., 0.1), rounding_precision)
# nest.SetStatus(sg, {'spike_times': np.round(times, rounding_precision)})
# print all(np.round(times, rounding_precision) == np.round(nest.GetStatus(sg, 'spike_times')[0], rounding_precision))

# neuron = nest.Create('iaf_cond_exp', 10)
# nest.Connect(sg, neuron)
# times = np.linspace(1000000., 100000000., 10000)
# times = np.arange(1000000000., 2000000000., 10000000.)
# for n_trial in range(len(times)-1):
# 	tt = np.arange(times[n_trial], times[n_trial+1], 0.1)
# 	nest.SetStatus(sg, {'spike_times': np.round(tt, rounding_precision)})
# 	print np.min(tt)
# 	print all(np.round(tt, rounding_precision) == np.round(nest.GetStatus(sg, 'spike_times')[0], rounding_precision))
# 	print all(np.round(tt, rounding_precision) == nest.GetStatus(sg, 'spike_times')[0])


start = 1000000.
tt = np.sort(np.random.sample(250) * 200.)
starts = []
comp = []
while True:
	times = np.arange(start, start+2000., 200.)

	for n_trial in range(len(times)-1):
		# tt = np.sort(times[n_trial] + np.random.sample(250) * (times[n_trial+1] - times[n_trial])) # np.arange(times[
		# n_trial],
		# times[n_trial+1], 0.01)
		tt += start
		# ttt = np.array([eval('%f' % np.round(x, rounding_precision)) for x in tt])
		times = [round(x, rounding_precision) for x in tt]
		nest.SetStatus(sg, {'spike_times': times})
		# nest.Simulate(200.)
		print np.min(nest.GetStatus(sg, 'spike_times')[0])
		print nest.GetStatus(sg, 'spike_times')[0][0]
		print all(np.round(tt, rounding_precision) == np.round(nest.GetStatus(sg, 'spike_times')[0], rounding_precision))
		# print all(np.round(tt, rounding_precision) == nest.GetStatus(sg, 'spike_times')[0])

		comp.append(all(np.round(tt, rounding_precision) == np.round(nest.GetStatus(sg, 'spike_times')[0], rounding_precision)))
		starts.append(np.min(tt))
	start += 10000.
	if start > 1e+15:
		break
	# starts.append(start)