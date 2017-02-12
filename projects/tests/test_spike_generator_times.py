__author__ = 'duarte'
import nest
import numpy as np
from modules.signals import determine_decimal_digits

dt = 0.001
rounding_precision = determine_decimal_digits(dt)

nest.ResetKernel()
# nest.set_verbosity('M_WARNING')
nest.SetKernelStatus({'resolution': dt})

sg = nest.Create('spike_generator', 1)#, {'allow_offgrid_spikes': True})

# Setting all at one doesn't seem to cause any problem (except some of the times are not the same
# times = np.round(np.arange(100000., 500000., 0.1), 1)
# nest.SetStatus(sg, {'spike_times': times})
# print all(times == nest.GetStatus(sg, 'spike_times')[0])


times = np.linspace(100000., 500000., 1000)
for n_trial in range(1000-1):
	tt = np.arange(times[n_trial], times[n_trial+1], 0.1)
	nest.SetStatus(sg, {'spike_times': np.round(tt, rounding_precision)})
	print all(np.round(tt, rounding_precision) == np.round(nest.GetStatus(sg, 'spike_times')[0], rounding_precision))