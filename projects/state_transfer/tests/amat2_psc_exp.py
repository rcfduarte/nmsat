"""
Yamauchi et al.
- 	The model is therefore fully specified by the parameters {taum, R, tauR, tauj , alphaj,(j =1, 2,..., L), omega}.
	We inherit many of the parameter values adopted in the original MAT model of L=2, as R=50MOhm, tauR =2 ms,
	tau1=10 ms, and tau2=200 ms, which

- 	The only one parameter that we modified from the original MAT model is the membrane time constant;
	we changed it from taum =5 to 10 ms.

- 		alpha1	alpha2	beta	omega						Ic			Desc
	A	10		0		-0.3	5 (== V_rest + 5 = -65)		0.08nA		Phasic spiking
	B 	-0.5	0.35	-0.3	5 (== V_rest + 5 = -65)		0.08nA		Phasic bursting


================================
Stuff that's wrong in the model:

1) Comments: *) tbd

"""

import nest
import time
import pylab

import nest.raster_plot
nest.ResetKernel()

########################################################################################################################
# A
# neuron_A 	= nest.Create("amat2_psc_exp")
# mm_A 		= nest.Create("multimeter")
# sd_A 		= nest.Create("spike_detector")
#
# nest.SetStatus(mm_A, {"withtime": True, "record_from": ['V_m', 'V_th'], "interval": 0.1})
#
# nest.SetStatus(neuron_A, {'alpha_1': 10.,
# 						  'alpha_2': 0.,
# 						  'beta': -0.3,
# 						  'omega': -65., # in the paper it's 5., but the model comment states the paper assumes omega
# 						  				 # to be relative to the resting V_m
# 						  'I_e': 90.,
# 						  'tau_m': 10.,
# 						  'tau_1': 10.,
# 						  'tau_2': 200.,
# 						  't_ref': 2.,
# 						  'C_m': 200.,})
#
#
# nest.Connect(mm_A, neuron_A)
# nest.Connect(neuron_A, sd_A)
# nest.Simulate(100.0)
#
# dVm_A 	= nest.GetStatus(mm_A)[0]
# Vms_A 	= dVm_A['events']['V_m']
# V_th_A 	= dVm_A['events']['V_th']
# ts_A 	= dVm_A['events']['times']
#
# pylab.figure(1)
# pylab.clf()
# pylab.plot(ts_A, Vms_A)
#
# pylab.figure(2)
# pylab.clf()
# pylab.plot(ts_A, V_th_A)
#
# nest.raster_plot.from_device(sd_A, hist = False)
# pylab.show(block=False)

########################################################################################################################
# B
neuron_B 	= nest.Create("amat2_psc_exp")
mm_B 		= nest.Create("multimeter")
sd_B 		= nest.Create("spike_detector")

nest.SetStatus(mm_B, {"withtime": True, "record_from": ['V_m', 'V_th'], "interval": 0.1})

nest.SetStatus(neuron_B, {'alpha_1': -0.5,
						  'alpha_2': 0.35,
						  'beta': -0.3,
						  'omega': -65., # in the paper it's 5., but the model comment states the paper assumes omega
						  				 # to be relative to the resting V_m
						  'I_e': 91.,
						  'tau_m': 10.,
						  'tau_1': 10.,
						  'tau_2': 200.,
						  't_ref': 2.,
						  'C_m': 200.,})


nest.Connect(mm_B, neuron_B)
nest.Connect(neuron_B, sd_B)
nest.Simulate(100.0)

dVm_B 	= nest.GetStatus(mm_B)[0]
Vms_B 	= dVm_B['events']['V_m']
V_th_B 	= dVm_B['events']['V_th']
ts_B 	= dVm_B['events']['times']

pylab.figure(1)
pylab.clf()
pylab.plot(ts_B, Vms_B)

pylab.figure(2)
pylab.clf()
pylab.plot(ts_B, V_th_B)

nest.raster_plot.from_device(sd_B, hist = False)
pylab.show(block=False)
