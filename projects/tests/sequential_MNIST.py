__author__ = 'duarte'
import numpy as np
import matplotlib.pyplot as pl

n_steps = 10

(x, y) = (0., 0.)
x_coord = []
y_coord = []
train_labels = np.loadtxt('../../data/sequences/trainlabels.txt')

for n in range(n_steps):
	stim = np.loadtxt('../../data/sequences/trainimg-{0}-inputdata.txt'.format(n))
	label = np.loadtxt('../../data/sequences/trainimg-{0}-targetdata.txt'.format(n))[:, :-4]
	points = np.loadtxt('../../data/sequences/trainimg-{0}-points.txt'.format(n), skiprows=1, delimiter=',')
	print np.argmax(label), np.argmax(np.mean(label, 0))

	reject = stim[:, 2]
	eos = stim[:, 3]

	pl.plot(points[~reject.astype(bool), 0], points[~reject.astype(bool), 1])









	# (x, y) = (0., 0.)
	x_coord = []
	y_coord = []
	for dx in range(len(stim[:, 0])):
		if dx == 0:
			(x, y) = (stim[dx, 0], stim[dx, 1])
		if not stim[dx, 2] and not stim[dx, 3]:
			# (x, y) = (x+stim[dx, 0], y+stim[dx, 1])
			(x, y) = (points[dx, 0] + stim[dx, 0], points[dx, 1] + stim[dx, 1])
			x_coord.append(x)
			y_coord.append(y)
	print x_coord, y_coord, len(stim[:, 0])

	fig, ax = pl.subplots()
	ax.plot(x_coord, y_coord, '.-', lw=4)
	ax.set_xlim([0., 40.])
	ax.set_ylim([0., 40.])#
	ax.set_title(str(train_labels[n]))
	# fig, ax = pl.subplots()
	# ax.plot(points[:, 0], points[:, 1], '.')
	pl.show()
	# pl.plot(stim[n, 0])