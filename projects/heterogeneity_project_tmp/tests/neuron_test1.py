__author__ = 'duarte'
import numpy as np
import nest
import pylab as pl
import nest.raster_plot

nest.ResetKernel()
nest.set_verbosity('M_WARNING')
nest.SetKernelStatus({'print_time': False, 'resolution': 0.1})
# create neurons
params = {
		'C_m': 116.51835339476165,
		'E_L': -76.42845423785847,
		'I_e': 0.,
		'V_m': -46.52785619857699,
		'V_th': -44.4502581905932,
		'V_reset': -54.18105341258776,
		'a': 4.0,
		'b': 30.0,
		'g_L': 4.63933405717297,
		'rec_bal': [1., 1., 0., 0.8],
		'rec_cond': [1., 1., 1., 1.],
		'rec_type': [1., -1., 2., -1.],
		'rec_reversal': [0., -75., 0., -90.],
		't_ref': 2.054115827009855,
		'tau_minus': 20.,
		'tau_minus_triplet': 200.,
		'tau_w': 100.,
		'tau_syn_d1': [2., 6., 0.1, 200.],
		'tau_syn_d2': [0.1, 0.1, 100., 600.],
		'tau_syn_rise': [0.3, 0.25, 1., 30.],
	}

ntot = 1
r_ext = 5.
ex_cells = nest.Create('iaf_cond_mtime', 1, params)

nest.SetStatus(ex_cells, {'V_m': -46.52785619857699})

# create generators
#ex_st = np.arange(10., 100., 10.)
ex_st = np.array([0.2, 1.5])
ex_noise = nest.Create('spike_generator', 1, params={'spike_times': ex_st})
in_st = np.array([1.0, 3.])
in_noise = nest.Create('spike_generator', 1, params={'spike_times': in_st})

#ex_noise = nest.Create('poisson_generator', 1, {'rate': 5.*1000.})
#in_noise = nest.Create('poisson_generator', 1, {'rate': 5.*1000.})
ex_input = nest.Create('parrot_neuron', 1)
nest.Connect(ex_noise, ex_input, syn_spec={'weight': 1., 'delay': 0.1})

in_input = nest.Create('parrot_neuron', 1)
nest.Connect(in_noise, in_input, syn_spec={'weight': 1., 'delay': 0.1})

# CONNECTIVITY ----------------------- #
nest.CopyModel('multiport_synapse', 'Glu')
nest.SetDefaults('Glu', {'receptor_types': [1., 3.]})
nest.Connect(ex_input, ex_cells, syn_spec = {'receptor_type': 1, "model": 'Glu'})

nest.CopyModel('multiport_synapse', 'GABA')
nest.SetDefaults('GABA', {'receptor_types': [2., 4.]})
nest.Connect(in_input, ex_cells, syn_spec={'weight': 1., 'delay': 0.1, 'receptor_type': 1, "model": 'GABA'})


#nest.Connect(ex_noise, ex_cells, syn_spec = {'weight': 1., 'delay': 0.1, 'receptor_type': 3, "model": 'multiport_synapse'})
#nest.Connect(in_noise, ex_cells, syn_spec = {'weight': 1., 'receptor_type': 3, "model": 'static_synapse'})
#nest.Connect(in_noise, ex_cells, syn_spec = {'weight': 1., 'receptor_type': 4, "model": 'static_synapse'})

ex_mm = nest.Create('multimeter')
nest.SetStatus(ex_mm, {'interval': .1, 'record_from': ['V_m','C1','S11','S12','S13','S14','C2','S21','S22','S23','S24','w', 'I_ex', 'I_in']})
nest.Connect(ex_mm, ex_cells) # connect multimeter

ex_spkdet = nest.Create('spike_detector')
nest.Connect(ex_cells, ex_spkdet)

nest.Simulate(10.1)

ex_events = nest.GetStatus(ex_mm)[0]['events']
ex_t = ex_events['times']  # time axis

#def syn_func(t,t0,params,i):#
#	dt = t-t0
#	if dt<0:
#		return 0
#	return (1.-np.exp(-dt/params['tau_syn_rise'][i])) * (
#			params['rec_bal'][i] * np.exp(-dt/params['tau_syn_d1'][i]) +
#			(1.-params['rec_bal'][i]) * np.exp(-dt/params['tau_syn_d2'][i])
#		)

#sol_1 = np.zeros_like(ex_t)
#f#or t0 in ex_st:
#	sol_1 += [syn_func(t,t0+1,params,0) for t in ex_t]
#sol_2 = np.zeros_like(ex_t)
#for t0 in in_st:#
#	sol_2 += [syn_func(t,t0+1,params,1) for t in ex_t]

pl.figure(1)
pl.clf()
pl.subplot(311)
pl.plot(ex_t, ex_events['V_m'], 'k', lw=2.)
pl.legend(['V_m'])
pl.ylabel('V [mV]')

pl.subplot(312)
pl.plot(ex_t, ex_events['C1'], '#0000ff',lw=2.)
pl.plot(ex_t, ex_events['S11'], '#3333ff')
pl.plot(ex_t, ex_events['S12'], '#6666ff')
pl.plot(ex_t, ex_events['S13'], '#9999ff')
pl.plot(ex_t, ex_events['S14'], '#ccccff')
#pl.plot(ex_t, sol_1, '--',c='#000000')
pl.plot(ex_t, ex_events['C2'], '#ff0000',lw=2.)
pl.plot(ex_t, ex_events['S21'], '#ff3333')
pl.plot(ex_t, ex_events['S22'], '#ff6666')
pl.plot(ex_t, ex_events['S23'], '#ff9999')
pl.plot(ex_t, ex_events['S24'], '#ffcccc')
#pl.plot(ex_t, sol_2, '--',c='#000000', lw=2.)
pl.legend(['e','e_rise','e_d1','e_d2','e_d3','i','i_rise','i_d1','i_d2','i_d3'],prop={'size':7})
pl.ylabel('Conductance [nS]')

#pl.subplot(313)
#pl.plot(ex_t, ex_events['w'], '#555555', lw=2.)
#pl.legend(['w'])
#pl.ylabel('Adaptation [pA]')


pl.subplot(313)
pl.plot(ex_t, ex_events['I_ex'], '#0000ff', lw=2.)
pl.plot(ex_t, ex_events['I_in'], '#ff0000', lw=2.)
pl.legend(['I_ex', 'I_in'])
pl.ylabel('Current [pA]')

#nest.raster_plot.from_device(ex_spkdet, hist=True)


pl.show()