import os
import numpy as np
import sys
def cancel_range(init, end):
	rang = np.arange(init, end)
	for n in rang:
		os.system('scancel '+ str(n))

if __name__=='__main__':
	cancel_range(int(sys.argv[1]), int(sys.argv[2]))