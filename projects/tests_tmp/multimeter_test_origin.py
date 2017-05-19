import nest

import nest.raster_plot

nest.ResetKernel()

neuron = nest.Create("iaf_psc_exp")

nest.SetStatus(neuron, {"C_m":   250.0, 
			"tau_m": 10.0, 
			"V_th":  -55.0, 
			"I_e":   375.0001})

# TODO play with origin here
voltmeter_shifted = nest.Create("multimeter", 1, {"interval": 202., 'start': 10.,
												  "withtime": True, "record_from": ['V_m'], })

nest.SetStatus(voltmeter_shifted, {'origin': 3.})
nest.Connect(voltmeter_shifted, neuron)
nest.Simulate(1000.0)
dVm_s = nest.GetStatus(voltmeter_shifted)[0]
Vms_s = dVm_s['events']['V_m']
ts_s = dVm_s['events']['times']

print "shifted len:" + str(len(ts_s))
print str(ts_s[:10])
