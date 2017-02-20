import nest
import time
import pylab

import nest.raster_plot
nest.ResetKernel()

neuron = nest.Create("iaf_psc_exp")

nest.SetStatus(neuron, {"C_m":   250.0, 
			"tau_m": 10.0, 
			"V_th":  -55.0, 
			"I_e":   375.0001})

voltmeter = nest.Create("multimeter")
nest.SetStatus(voltmeter, {"withtime": True, "record_from": ['V_m'], "interval": 0.1})
sd = nest.Create("spike_detector")

nest.Connect(voltmeter, neuron)
nest.Connect(neuron, sd)

nest.Simulate(10000.0)

dVm = nest.GetStatus(voltmeter)[0]
Vms = dVm['events']['V_m']
ts = dVm['events']['times']



pylab.figure(1)


pylab.clf()
pylab.plot(ts,Vms)
pylab.show(block=False)

nest.raster_plot.from_device(sd, hist = False)

pylab.show()