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

	return tmpList

######################################################################################################################
# Test

nE = 10000
pEE = 0.168

connections = {'EE': {'source': nE, 'target': nE, 'p': pEE, 'w': [], 'd': [], 'corr_in': 1, 'corr_out': 0, 'd_in': 5,
	                'd_out': 5, 'conn_list': []}}

# connections = {'EE': {'source': nE, 'target': nE, 'p': pEE, 'w': [], 'd': [], 'corr_in': 0, 'corr_out': 0, 'd_in': 0,
# 	                'd_out': 0, 'conn_list': []}}

for connection, conn_dict in connections.items():
	conn_dict['w'] = st.lognorm.rvs(s=1., loc=1., scale=1., size=conn_dict['source']*conn_dict['target'])
	conn_dict['d'] = st.lognorm.rvs(s=1., loc=0.1, scale=0.5, size=conn_dict['source']*conn_dict['target'])

original_connections = connections.copy()

for connection, conn_dict in connections.items():
	conn_dict['conn_list'] = generate_connectivity(conn_dict['source'], conn_dict['target'], conn_dict['p'],
	                                                conn_dict['d_in'], conn_dict['d_out'], conn_dict['w'],
	                                                conn_dict['d'], conn_dict['corr_in'], conn_dict['corr_out'])

	maxC = int(np.floor(conn_dict['source'] * conn_dict['target'] * conn_dict['p']))
	w = conn_dict['w'][:maxC]
	d = conn_dict['d'][:maxC]
	sources = np.arange(0, conn_dict['source'], 1).astype(int)
	targets = np.arange(0, conn_dict['target'], 1).astype(int)

	tmpList = np.array([], [('pre', int), ('post', int), ('w', float), ('d', np.ndarray)])
	tmpList.resize(maxC)
	for c in range(maxC):
		src, tgt = (np.random.randint(0, conn_dict['source'], 1), np.random.randint(0, conn_dict['target'], 1))
		tmpList[c] = ((sources[src]), (targets[tgt]), conn_dict['w'][c], conn_dict['d'][c])
	original_connections[connection]['conn_list'] = tmpList



fig = pl.figure()
ax1 = fig.add_subplot(121)
ax1.set_title("Original W")
ax2 = fig.add_subplot(122)
ax2.set_title("Original d")
out_degrees = []
in_degrees = []
for connection, conn_dict in original_connections.items():
	for j in range(int(conn_dict['source'])):
		out_degrees.append(len(np.where(conn_dict['conn_list']['pre'] == j)[0]))
	for i in range(int(conn_dict['target'])):
		in_degrees.append(len(np.where(conn_dict['conn_list']['post'] == i)[0]))
w_out = []
w_in = []
weights = []
for Eneuron in np.arange(0, nE):
	for connection, conn_dict in original_connections.items():
		if connection[-1] == 'E':
			w_out.append(conn_dict['conn_list']['w'][np.where(conn_dict['conn_list']['pre'] == Eneuron)])
		elif connection[0] == 'E':
			w_in.append(conn_dict['conn_list']['w'][np.where(conn_dict['conn_list']['post'] == Eneuron)])
	w_o = list(itertools.chain(*w_out))
	w_i = list(itertools.chain(*w_in))
	weights.append(np.mean(list(itertools.chain(w_o, w_i))))
ax1.hist(weights, bins=100)

degrees = list(itertools.chain(*[in_degrees, out_degrees]))
ax2.hist(degrees, bins=100)

###############################################
fig2 = pl.figure()
ax21 = fig2.add_subplot(121)
ax21.set_title("Modified W")
ax22 = fig2.add_subplot(122)
ax22.set_title("Modified d")
out_degrees = []
in_degrees = []
for connection, conn_dict in connections.items():
	for j in range(int(conn_dict['source'])):
		out_degrees.append(len(np.where(conn_dict['conn_list']['pre'] == j)[0]))
	for i in range(int(conn_dict['target'])):
		in_degrees.append(len(np.where(conn_dict['conn_list']['post'] == i)[0]))
w_out = []
w_in = []
weights = []
for Eneuron in np.arange(0, nE):
	for connection, conn_dict in connections.items():
		if connection[-1] == 'E':
			w_out.append(conn_dict['conn_list']['w'][np.where(conn_dict['conn_list']['pre'] == Eneuron)])
		elif connection[0] == 'E':
			w_in.append(conn_dict['conn_list']['w'][np.where(conn_dict['conn_list']['post'] == Eneuron)])
	w_o = list(itertools.chain(*w_out))
	w_i = list(itertools.chain(*w_in))
	weights.append(np.mean(list(itertools.chain(w_o, w_i))))
ax21.hist(weights, bins=100)

degrees = list(itertools.chain(*[in_degrees, out_degrees]))
ax22.hist(degrees, bins=100)

pl.show()