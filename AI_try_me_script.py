from modules import analysis
from modules import visualization as viz

from modules import signals
import numpy as np

neurons = 50
all_spikes = []

T = 140
nspk = 30

for n in range(neurons):
	a = np.arange(T)
	np.random.shuffle(a)
	spikes = a[:nspk]
	spikes = [(n, float(x)) for x in spikes]
	all_spikes += list(spikes)

spikelist = signals.SpikeList(all_spikes, range(neurons))

ai = viz.ActivityIllustrator(spikelist, populations=None, vm_list=[], ids=None)
ai.animate_activity(time_interval=100, time_window=100, fps=60, frame_res=0.25, save=True,
					filename="animation_test1.py", activities=["raster", "rate"])

ai = viz.ActivityIllustrator(spikelist, populations=None, vm_list=[], ids=[range(35), range(35,50)])
ai.animate_activity(time_interval=100, time_window=100, fps=60, frame_res=0.25, save=True,
					filename="animation_test2.py", activities=["raster", "rate"], colors=['b','r'])

