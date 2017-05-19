__author__ = 'duarte'
__all__ = ['noise_driven_dynamics']

from modules.signals import SpikeList
from modules.visualization import SpikePlots
from defaults.paths import paths
import matplotlib.pyplot as pl

def plot_twopool_activity(net, label, start_t=1000., stop_t=2000.):
	# ######################################################################################################################
	# Let the plotting begin
	# ======================================================================================================================
	sl_1 	= net.populations[0].spiking_activity.id_slice(list(net.populations[0].gids[:1000]))
	sl_i1 	= net.populations[1].spiking_activity.id_slice(list(net.populations[1].gids[:250]))
	sl_2 	= net.populations[2].spiking_activity.id_slice(list(net.populations[2].gids[:1000]))
	sl_i2 	= net.populations[3].spiking_activity.id_slice(list(net.populations[3].gids[:250]))

	spikelist = []
	id_list = []
	missing = []
	offset = 7000

	def merge_spikes(pop, offs):
		for id_ in pop.id_list:
			if len(pop.spiketrains[id_]) > 0:
				id_list.append(offs + id_)
				for spike in pop.spiketrains[id_].spike_times:
					spikelist.append((offs + id_, spike))
			else:
				missing.append(offs + id_)

	merge_spikes(sl_1, 0)
	merge_spikes(sl_i1, -offset)
	merge_spikes(sl_2, 0)
	merge_spikes(sl_i2, -offset)

	merged_spikelist = SpikeList(spikelist, id_list)
	merged_spikelist.complete(missing)

	sp = SpikePlots(merged_spikelist, start=start_t, stop=stop_t)

	fig = pl.figure(figsize=(30, 20))
	fig.suptitle("Spiking activity for two reservoirs")
	ax1 = pl.subplot2grid((80, 1), (10, 0), rowspan=25, colspan=1)
	ax0 = pl.subplot2grid((80, 1), (0, 0), rowspan=6, colspan=1, sharex=ax1)
	ax2 = pl.subplot2grid((80, 1), (40, 0), rowspan=25, colspan=1)
	ax3 = pl.subplot2grid((80, 1), (68, 0), rowspan=6, colspan=1, sharex=ax1)

	ax3.set(xlabel='Time [ms]', ylabel='Rate')
	ax1.set(ylabel='P1 Neurons')
	ax2.set(ylabel='P2 Neurons')

	sp.dot_display(gids_colors=[(sl_1.id_list, 'b'), (np.array([x - offset for x in sl_i1.id_list]), 'r')],
				   ax=[ax1, ax0],
				   with_rate=True, display=False)

	sp.dot_display(gids_colors=[(sl_2.id_list, 'b'), (np.array([x - offset for x in sl_i2.id_list]), 'r')],
				   ax=[ax2, ax3],
				   with_rate=True, display=False)

	fig.savefig("{0}/Raster_{1}_P1_P2.pdf".format(paths['figures'], label))