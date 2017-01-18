#!/usr/bin/python
import matplotlib
matplotlib.use('Agg')
import importlib
from modules.parameters import ParameterSpace
from optparse import OptionParser
from os import path
import sys


def run_experiment(params_file_full_path, computation_function="noise_driven_dynamics", **parameters):
	"""

	:param params_file_full_path: full path to parameter file
	:param computation_function: which experiment to run
	:param parameters: other CLI input parameters
	:return:
	"""
	try:
		# experiment = importlib.import_module("Computation." + computation_function)
		# changed this to account for the project folder..
		project_dir, _ = path.split(path.split(params_file_full_path)[0])
		sys.path.append(project_dir)
		experiment = importlib.import_module("computations." + computation_function)
	except:
		print("Could not find experiment `%s`. Is it in the project's ./computations/ directory?" %
		      computation_function)
		exit(-1)

	if 'keep_all' in parameters.keys():
		pars = ParameterSpace(params_file_full_path, keep_all=parameters['keep_all'])
		parameters.pop('keep_all')
	else:
		pars = ParameterSpace(params_file_full_path)

	pars.save(pars[0].kernel_pars.data_path+pars[0].kernel_pars.data_prefix+'_ParameterSpace.py')

	if pars[0].kernel_pars.system['local']:
		results = pars.run(experiment.run, **parameters)
		return results
	else:
		pars.run(experiment.run, project_dir, **parameters)


def create_parser():
	parser_ = OptionParser()
	parser_.add_option("-f", "--param-file", dest="p_file", help="parameter file", metavar="FILE")
	parser_.add_option("-c", "--computation-function", dest="c_function", help="computation function to execute")

	return parser_


def print_welcome_message():
	print("""
*** Neural Microcircuit Testbed ***

Version 0.1

This program is provided AS IS and comes with
NO WARRANTY. See the file LICENSE for details.
""")


if __name__ == "__main__":
	print_welcome_message()
	(options, args) = create_parser().parse_args()

	# we need minimum 2 arguments (3 including the script)
	if len(sys.argv) < 3 or options.p_file is None or options.c_function is None:
		print("At least two arguments (parameter file and computation function) are required! Exiting..")
		exit(-1)

	# TODO do we allow random arguments or should we include all possible params as options, would be nice in the help
	d = dict([arg.split('=', 1) for arg in args])
	for k, v in d.items():
		if v == 'False':
			d.update({k: False})
		elif v == 'True':
			d.update({k: True})
		elif v == 'None':
			d.update({k: None})
	run_experiment(options.p_file, computation_function=options.c_function, **d)