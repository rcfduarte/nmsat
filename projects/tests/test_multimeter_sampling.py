import nest
import numpy as np
import matplotlib.pyplot as pl
from modules.signals import AnalogSignalList

dt = 0.1

nest.SetKernelStatus({'resolution': dt})


neurons = nest.Create('iaf_psc_exp', 10, {'V_m': -60., 'I_e': 400.})

mm1 = nest.Create('multimeter', 1, {'start': 0.1, 'interval': dt, 'record_from': ['V_m']})
mm2 = nest.Create('multimeter', 1, {'start': 0.5, 'interval': 10*dt, 'record_from': ['V_m']})
mm3 = nest.Create('multimeter', 1, {'start': 0.1, 'interval': 100.*dt, 'record_from': ['V_m']})

nest.Connect(mm1, neurons)
nest.Connect(mm2, neurons)
nest.Connect(mm3, neurons)

nest.Simulate(1000. + dt)


events1 = nest.GetStatus(mm1)[0]['events']
events2 = nest.GetStatus(mm2)[0]['events']
events3 = nest.GetStatus(mm3)[0]['events']


def plot_events(events, ax, linestyle='-', c='b'):
	"""
	"""
	times = events['times']
	print min(times), max(times), np.mean(np.diff(np.unique(times)))
	tmp = [(events['senders'][n], events['V_m'][n]) for n in range(len(events['senders']))]
	responses = AnalogSignalList(tmp, np.unique(events['senders']).tolist(), times=times)
	print min(responses.time_axis()), max(responses.time_axis()), responses.dt
	ax.plot(responses.time_axis(), responses.as_array()[0, :], linestyle, c=c)

fig, ax = pl.subplots()
plot_events(events1, ax, '-', 'b')
plot_events(events2, ax, 'o', 'g')
plot_events(events3, ax, 'o', 'r')

pl.show()
