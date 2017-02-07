import os
import sys

def submit_jobs(start_idx=0, stop_idx=None):
	with open('./job_list.txt') as fp:
		for idx, line in enumerate(fp):
			if stop_idx is not None:
				if (idx>=start_idx) and (idx<=stop_idx):
					os.system('sbatch {0}'.format(line))
			else:
				os.system('sbatch {0}'.format(line))

if __name__=='__main__':
	if len(sys.argv)>1:
		submit_jobs(int(sys.argv[1]), int(sys.argv[2]))