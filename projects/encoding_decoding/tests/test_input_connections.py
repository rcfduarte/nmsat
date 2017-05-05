__author__ = 'duarte'
import numpy as np
import matplotlib.pyplot as pl
from sklearn.preprocessing import normalize
import sys
sys.path.append('../')
from read_data.auxiliary_functions import generate_input_connections

########################################################################################################################
N = 10000
n_stim = 4

# gamma_in = 0.6
# N_aff = gamma_in * N
#
# r = 0.
#
# mu_w = 1.
# sig_w = 0.5
#
# W_in = generate_input_connections(n_stim, gamma_in, N, r, mu_w, sig_w)
# print "density = {0} / gamma_u = {1}".format(str(float(len(np.nonzero(W_in)[0])) / float(W_in.size)), str(gamma_in))
# pl.imshow(W_in, interpolation='nearest', aspect='auto')
# pl.show()
#
#
# ###
# for gamma_in in np.arange(0.0, 1.1, 0.1):
# 	mu_w = 1.
# 	sig_w = 0.
# 	W_in = generate_input_connections(n_stim, gamma_in, N, r, mu_w, sig_w)
# 	print "density = {0} / gamma_u = {1}".format(str(float(len(np.nonzero(W_in)[0])) / float(W_in.size)), str(gamma_in))
# 	pl.imshow(W_in, interpolation='nearest', aspect='auto')
# 	pl.show()
#
# ###
# for r in np.arange(0.0, 1.1, 0.1):
# 	mu_w = 1.
# 	sig_w = 0.
# 	W_in = generate_input_connections(n_stim, gamma_in, N, r, mu_w, sig_w)
# 	print "density = {0} / gamma_u = {1} / r = {2}".format(str(float(len(np.nonzero(W_in)[0])) / float(W_in.size)), str(gamma_in), str(r))
# 	pl.imshow(W_in, interpolation='nearest', aspect='auto')
# 	pl.show()

####
fig = pl.figure()
ax_ctr = 1

for r in np.arange(0.0, 1.1, 0.1):
	for gamma_in in np.arange(0.0, 1.1, 0.1):
		ax = fig.add_subplot(11, 11, ax_ctr)
		mu_w = 1.
		sig_w = 0.
		try:
			W_in = generate_input_connections(n_stim, gamma_in, N, r, mu_w, sig_w)
			ax.imshow(W_in, interpolation='nearest', aspect='auto', cmap='gray_r')
		except:
			continue
		ax_ctr += 1
pl.show()