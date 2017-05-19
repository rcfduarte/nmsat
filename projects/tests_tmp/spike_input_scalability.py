__author__ = 'duarte'
from modules.input_architect import generate_template
from modules import signals
import numpy as np
import pylab as pl
import nest
import time

# Parameters
parrot_layer = True
nStim = 10
nU = 10000
nC = 10000
ppns = [1, 2, 4, 8, 16]

times = []
for ppn in ppns:
	nest.ResetKernel()
	nest.SetKernelStatus({'total_num_virtual_procs': ppn})
	###################################################################
	sg = nest.Create('spike_generator', nU)
	circuit = nest.Create('aeif_cond_exp', nC)

	if parrot_layer:
		parrots = nest.Create('parrot_neuron', nU)
		nest.Connect(sg, parrots, {'rule': 'one_to_one'})
		nest.Connect(parrots, circuit, {'rule': 'pairwise_bernoulli', 'p': 0.1})
	else:
		nest.Connect(sg, circuit, {'rule': 'pairwise_bernoulli', 'p': 0.1})

	start_main = time.time()
	for n in range(nStim):
		print "\n\nSTEP " + str(n)
		spks = generate_template(n_neurons=nU, rate=20., duration=200., resolution=0.1)
		rounding_precision = signals.determine_decimal_digits(spks.raw_data()[:, 0][0])
		check_timing = []
		for nn in spks.id_list:
			spk_times = [round(n, rounding_precision) for n in spks[nn].spike_times]  # to be sure
			nest.SetStatus([sg[nn]], {'spike_times': spk_times})
			check_timing.append(all(spk_times == np.round(nest.GetStatus([sg[nn]], 'spike_times')[0],
			                                              rounding_precision)))
		print(all(check_timing))

		nest.Simulate(200.)

	stop_main = time.time() - start_main
	print stop_main
	times.append(stop_main)


# times_0 = [82.67961096763611, 45.74026393890381, 27.544389963150024, 27.580265045166016, 24.396315097808838]
# times_1 = [13.148087978363037, 17.19286799430847, 23.62993884086609, 256.60288095474243]
times_1 = [501.62666296958923, 318.2111392021179, 238.5142068862915,
 188.40600991249084, 223.6072759628296]

fig, ax = pl.subplots()
ax.plot(ppns, times, 'r-', label='SG-P-C')
ax.plot(ppns, times_1, 'g--', label='SG-C')
ax.set_xlabel('ppn')
ax.set_ylabel('Time [s]')
pl.legend()
pl.show()
