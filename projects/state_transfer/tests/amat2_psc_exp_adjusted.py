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
import matplotlib.pyplot as plt

import nest.raster_plot
nest.ResetKernel()

params = {
	'A': {'alpha_1': 10.,
		  'alpha_2': 0.,
		  'beta': -0.3,
		  'omega': -65., # in the paper it's 5., but the model comment states the paper assumes omega
		  # to be relative to the resting V_m
		  'I_e': 91.,
		  'tau_m': 10.,
		  'tau_1': 10.,
		  'tau_2': 200.,
		  't_ref': 2.,
		  'C_m': 200.,},

	'B': {'alpha_1': -0.5,
		  'alpha_2': 0.35,
		  'beta': -0.3,
		  'omega': -65., # in the paper it's 5., but the model comment states the paper assumes omega
		  # to be relative to the resting V_m
		  'I_e': 91.,
		  'tau_m': 10.,
		  'tau_1': 10.,
		  'tau_2': 200.,
		  't_ref': 2.,
		  'C_m': 200.,},

	'C': {'alpha_1': 10.,
		  'alpha_2': 0.,
		  'beta': -1.,
		  'omega': -65., # in the paper it's 5., but the model comment states the paper assumes omega
		  # to be relative to the resting V_m
		  'I_e': 0.,
		  'tau_m': 10.,
		  'tau_1': 10.,
		  'tau_2': 200.,
		  't_ref': 2.,
		  'C_m': 200.,},

	'D': {'alpha_1': 10.,
		  'alpha_2': 0.,
		  'beta': -2.5,
		  'omega': -65., # in the paper it's 5., but the model comment states the paper assumes omega
		  # to be relative to the resting V_m
		  'I_e': 0.,
		  'tau_m': 10.,
		  'tau_1': 10.,
		  'tau_2': 200.,
		  't_ref': 2.,
		  'C_m': 200.,},

	'E': {'alpha_1': -0.5,
		  'alpha_2': 0.35,
		  'beta': -2.5,
		  'omega': -65., # in the paper it's 5., but the model comment states the paper assumes omega
		  # to be relative to the resting V_m
		  'I_e': 0.,
		  'tau_m': 10.,
		  'tau_1': 10.,
		  'tau_2': 200.,
		  't_ref': 2.,
		  'C_m': 200.,},

	'F': {'alpha_1': 10.,
		  'alpha_2': 0.,
		  'beta': -0.1,
		  'omega': -65., # in the paper it's 5., but the model comment states the paper assumes omega
		  # to be relative to the resting V_m
		  'I_e': 0.,
		  'tau_m': 10.,
		  'tau_1': 10.,
		  'tau_2': 200.,
		  't_ref': 2.,
		  'C_m': 200.,},
}






fig = plt.figure(figsize=(15, 9), dpi=100,)
fig.suptitle("Overview using adjusted parameters")
cnt = 1

########################################################################################################################
# A
nest.ResetKernel()
p 		= params['A']
neuron 	= nest.Create("amat2_psc_exp")
mm 		= nest.Create("multimeter")
sd 		= nest.Create("spike_detector")

nest.SetStatus(mm, {"withtime": True, "record_from": ['V_m', 'V_th'], "interval": 0.1})

nest.SetStatus(neuron, p)

nest.Connect(mm, neuron)
nest.Connect(neuron, sd)
nest.Simulate(200.0)

dVm 	= nest.GetStatus(mm)[0]
Vms 	= dVm['events']['V_m']
V_th 	= dVm['events']['V_th']
ts 	= dVm['events']['times']

plt.subplot(2,3,cnt)
plt.plot(ts, Vms, 'b')
plt.plot(ts, V_th, 'r')
plt.title('Phasic Spiking (91pA vs 80pA)')
spikes	= nest.GetStatus(sd)[0]['events']['times']
for x in spikes:
	plt.axvline(x, ymin=0., ymax=0.2)
cnt += 1


########################################################################################################################
# B
nest.ResetKernel()
p 		= params['B']
neuron 	= nest.Create("amat2_psc_exp")
mm 		= nest.Create("multimeter")
sd 		= nest.Create("spike_detector")

nest.SetStatus(mm, {"withtime": True, "record_from": ['V_m', 'V_th'], "interval": 0.1})

nest.SetStatus(neuron, p)

nest.Connect(mm, neuron)
nest.Connect(neuron, sd)
nest.Simulate(200.0)

dVm 	= nest.GetStatus(mm)[0]
Vms 	= dVm['events']['V_m']
V_th 	= dVm['events']['V_th']
ts 		= dVm['events']['times']

plt.subplot(2,3,cnt)
plt.plot(ts, Vms, 'b')
plt.plot(ts, V_th, 'r')

spikes	= nest.GetStatus(sd)[0]['events']['times']
for x in spikes:
	plt.axvline(x, ymin=0., ymax=0.2)
plt.title('Phasic Bursting (91 vs 80)')

# nest.raster_plot.from_device(sd_B, hist = False)
cnt += 1

########################################################################################################################
# C
nest.ResetKernel()
p 		= params['C']
neuron 	= nest.Create("amat2_psc_exp")
mm 		= nest.Create("multimeter")
sd 		= nest.Create("spike_detector")
dc 		= nest.Create("dc_generator")

nest.SetStatus(mm, {"withtime": True, "record_from": ['V_m', 'V_th'], "interval": 0.1})
# nest.SetStatus(dc, {"start": 20., "stop": 20.5, "amplitude": 580.})
nest.SetStatus(dc, {"start": 20., "stop": 20.5, "amplitude": 940.})

nest.SetStatus(neuron, p)

nest.Connect(mm, neuron)
nest.Connect(dc, neuron)
nest.Connect(neuron, sd)
nest.Simulate(200.0)

dVm 	= nest.GetStatus(mm)[0]
Vms 	= dVm['events']['V_m']
V_th 	= dVm['events']['V_th']
ts 		= dVm['events']['times']

plt.subplot(2,3,cnt)
plt.plot(ts, Vms, 'b')
plt.plot(ts, V_th, 'r')
plt.title('Latency (940 vs 580)')

spikes	= nest.GetStatus(sd)[0]['events']['times']
for x in spikes:
	plt.axvline(x, ymin=0., ymax=0.2)
cnt += 1

########################################################################################################################
# D
nest.ResetKernel()
p 		= params['D']
neuron 	= nest.Create("amat2_psc_exp")
mm 		= nest.Create("multimeter")
sd 		= nest.Create("spike_detector")
dc 		= nest.Create("dc_generator")

nest.SetStatus(mm, {"withtime": True, "record_from": ['V_m', 'V_th'], "interval": 0.1})
# nest.SetStatus(dc, {"start": 20., "stop": 21., "amplitude": -600.})
nest.SetStatus(dc, {"start": 20., "stop": 21., "amplitude": -1000.})

nest.SetStatus(neuron, p)

nest.Connect(mm, neuron)
nest.Connect(dc, neuron)
nest.Connect(neuron, sd)
nest.Simulate(200.0)

dVm 	= nest.GetStatus(mm)[0]
Vms 	= dVm['events']['V_m']
V_th 	= dVm['events']['V_th']
ts 		= dVm['events']['times']

plt.subplot(2,3,cnt)
plt.plot(ts, Vms, 'b')
plt.plot(ts, V_th, 'r')
plt.title('Rebound spiking (1000 vs 600)')

spikes	= nest.GetStatus(sd)[0]['events']['times']
for x in spikes:
	plt.axvline(x, ymin=1., ymax=0.8)

cnt += 1

########################################################################################################################
# E
nest.ResetKernel()
p 		= params['E']
neuron 	= nest.Create("amat2_psc_exp")
mm 		= nest.Create("multimeter")
sd 		= nest.Create("spike_detector")
dc 		= nest.Create("dc_generator")

nest.SetStatus(mm, {"withtime": True, "record_from": ['V_m', 'V_th'], "interval": 0.1})
# nest.SetStatus(dc, {"start": 20., "stop": 21., "amplitude": -600.})
nest.SetStatus(dc, {"start": 20., "stop": 21., "amplitude": -1100.})

nest.SetStatus(neuron, p)

nest.Connect(mm, neuron)
nest.Connect(dc, neuron)
nest.Connect(neuron, sd)
nest.Simulate(200.0)

dVm 	= nest.GetStatus(mm)[0]
Vms 	= dVm['events']['V_m']
V_th 	= dVm['events']['V_th']
ts 		= dVm['events']['times']

plt.subplot(2,3,cnt)
plt.plot(ts, Vms, 'b')
plt.plot(ts, V_th, 'r')
plt.title('Rebound bursting (1100 vs 600)')
spikes	= nest.GetStatus(sd)[0]['events']['times']
for x in spikes:
	plt.axvline(x, ymin=0., ymax=0.2)

cnt += 1

########################################################################################################################
# F
nest.ResetKernel()
p 		= params['F']
neuron 	= nest.Create("amat2_psc_exp")
mm 		= nest.Create("multimeter")
sd 		= nest.Create("spike_detector")
dc1		= nest.Create("dc_generator")
dc2		= nest.Create("dc_generator")
dc3		= nest.Create("dc_generator")

nest.SetStatus(mm, {"withtime": True, "record_from": ['V_m', 'V_th'], "interval": 0.1})
# nest.SetStatus(dc1, {"start": 20., "stop": 22., "amplitude": 200.})
# nest.SetStatus(dc2, {"start": 140., "stop": 142., "amplitude": -200.})
# nest.SetStatus(dc3, {"start": 155., "stop": 157., "amplitude": 200.})
nest.SetStatus(dc1, {"start": 20., "stop": 22., "amplitude": 510.})
nest.SetStatus(dc2, {"start": 140., "stop": 142., "amplitude": -510.})
nest.SetStatus(dc3, {"start": 155., "stop": 157., "amplitude": 510.})

nest.SetStatus(neuron, p)

nest.Connect(mm, neuron)
nest.Connect(dc1, neuron)
nest.Connect(dc2, neuron)
nest.Connect(dc3, neuron)
nest.Connect(neuron, sd)
nest.Simulate(200.0)

dVm 	= nest.GetStatus(mm)[0]
Vms 	= dVm['events']['V_m']
V_th 	= dVm['events']['V_th']
ts 		= dVm['events']['times']

plt.subplot(2,3,cnt)
plt.plot(ts, Vms, 'b')
plt.plot(ts, V_th, 'r')
plt.title('Threshold variability \n(510, -510, 510 vs 200..)')
spikes	= nest.GetStatus(sd)[0]['events']['times']
for x in spikes:
	plt.axvline(x, ymin=0., ymax=0.2)

cnt += 1


fig.savefig('overview_adjusted.pdf')
