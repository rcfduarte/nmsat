__author__ = 'duarte'
import numpy as np
import scipy.stats as st
import pylab as pl
import itertools


def generate_connectivity(n_pre, n_post, p, d_in, d_out, weights, delays, corrIn, corrOut):
	"""

	:param n_pre: Number of presynaptic neurons
	:param n_post: Number of postsynaptic neurons
	:param p: Connection probability
	:param d_in: Parameter to skew in-degree distribution
	:param d_out: Parameter to skew out-degree distribution
	:param weights: List or array of synpatic weights
	:param delays: List or array of synaptic delays
	:param corrOut: Output weight correlation
	:param corrIn: Input weight correlation
	:return:
	"""
	mu_in = -1. * (corrIn ** 2) / 2.
	# if corrIn == 0:
	# 	corrIn = 0.0000001
	alpha_post = st.lognorm.rvs(s=0.3, loc=mu_in, scale=corrIn, size=n_post)

	mu_out = -1. * (corrOut ** 2) / 2.
	# if corrOut == 0:
	# 	corrOut = 0.0000001
	alpha_pre = st.lognorm.rvs(s=0.3, loc=mu_out, scale=corrOut, size=n_pre)

	c_out = np.sum([np.exp((-n * d_out) / n_pre) for n in range(int(n_pre))])
	probs_pre = [(1./c_out) * np.exp((-n * d_out) / n_pre) for n in range(int(n_pre))]
	sources = np.arange(0, n_pre, 1).astype(int)

	c_in = np.sum([np.exp((-n * d_in) / n_post) for n in range(int(n_post))])
	probs_post = [(1./c_in) * np.exp((-n * d_in) / n_post) for n in range(int(n_post))]
	targets = np.arange(0, n_post, 1).astype(int)

	maxC = int(np.floor(n_pre * n_post * p))

	w = weights[:maxC]
	d = delays[:maxC]

	tmpList = np.array([], [('pre', int), ('post', int), ('w', float), ('d', np.ndarray)])
	tmpList.resize(maxC)

	A = np.zeros((n_pre, n_post))

	for c in np.arange(0, maxC).astype(int):
		source = np.random.choice(sources, replace=True, p=probs_pre)
		target = np.random.choice(targets, replace=True, p=probs_post)

		timeout = 100
		decCount = 0
		decisor = np.random.randint(2)

		while A[source, target] > 0:
			decCount += 1
			if decisor:
				if decCount > timeout:
					target = np.random.choice(targets, p=probs_post)
					decCount = 0
				else:
					source = np.random.choice(sources, p=probs_pre)
			else:
				if decCount > timeout:
					source = np.random.choice(sources, p=probs_pre)
					decCount = 0
				else:
					target = np.random.choice(targets, p=probs_post)
		wT = w[c]

		if corrOut > 0:
			wT *= alpha_pre[source]
		if corrIn > 0:
			wT *= alpha_post[target]

		tmpList[c] = ((source), (target), wT, d[c])
		A[source, target] = 1
	pl.imshow(A)
	pl.show()

	return tmpList

N = 1000
p = 0.1

weights = st.lognorm.rvs(s=1., loc=1., scale=1., size=N*N)
delays = st.lognorm.rvs(s=1., loc=0.1, scale=0.5, size=N*N)


for (d, corr) in zip([0, 5], [0, 1]):
	conn_list = generate_connectivity(N, N, p, d, d, weights, delays, corr, corr)
	globals()['d{0}'.format(str(d))] = conn_list

fig = pl.figure()
ax1 = fig.add_subplot(121)
ax1.set_title("Original W")
ax2 = fig.add_subplot(122)
ax2.set_title("Original d")

fig2 = pl.figure()
ax21 = fig2.add_subplot(121)
ax21.set_title("Modified W")
ax22 = fig2.add_subplot(122)
ax22.set_title("Modified d")

out_degrees = []
in_degrees = []
for i in range(N):
	out_degrees.append(len(np.where(d0['pre'] == i)[0]))
	in_degrees.append(len(np.where(d0['post'] == i)[0]))
ax1.hist(d0['w'], bins=100)
degrees = list(itertools.chain(*[in_degrees, out_degrees]))
ax2.hist(degrees, bins=100)

out_degrees = []
in_degrees = []
for i in range(N):
	out_degrees.append(len(np.where(d5['pre'] == i)[0]))
	in_degrees.append(len(np.where(d5['post'] == i)[0]))
ax21.hist(d5['w'], bins=100)
degrees = list(itertools.chain(*[in_degrees, out_degrees]))
ax22.hist(degrees, bins=100)
pl.show()
