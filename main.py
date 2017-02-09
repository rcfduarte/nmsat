#!/usr/bin/python
import sys
import copy

cmdl_parse = copy.copy(sys.argv)

import matplotlib
matplotlib.use('Agg')
from argparse import ArgumentParser
from os import path
import importlib
from modules.parameters import ParameterSpace


def run_experiment(params_file_full_path, computation_function="noise_driven_dynamics", **parameters):
	"""

	:param params_file_full_path: full path to parameter file
	:param computation_function: which experiment to run
	:param parameters: other CLI input parameters
	:return:
	"""
	try:
		# determine the project folder and add it to sys.path..
		project_dir, _ = path.split(path.split(params_file_full_path)[0])
		sys.path.append(project_dir)
		experiment = importlib.import_module("computations." + computation_function)
	except Exception as err:
		print("Could not find experiment `{0}`. Is it in the project's ./computations/ directory? \nError: {1}".format(
			computation_function, str(err)))
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
	parser_ = ArgumentParser(prog="main.py")
	parser_.add_argument("-f", dest="p_file", nargs=1, default=None, metavar="parameter_file",
					  help="absolute path to parameter file", required=True)
	parser_.add_argument("-c", dest="c_function", nargs=1, default=None, metavar="computation_function",
					   help="computation function to execute", required=True)
	parser_.add_argument("--cluster", dest="cluster", nargs=1, default=None, metavar="Blaustein",
					   help="name of cluster entry in default paths dictionary")
	return parser_


def print_welcome_message():
	print("""
   *** Neural Microcircuit Testbed ***

\t      Version 0.1

This program is provided AS IS and comes with
NO WARRANTY. See the file LICENSE for details.
""")


if __name__ == "__main__":
	print_welcome_message()
	args = create_parser().parse_args(cmdl_parse[1:])  # avoids pynest sys.argv pollution

	# # TODO do we allow random arguments or should we include all possible params as options, would be nice in the help
	# d = dict([arg.split('=', 1) for arg in args[1:]])
	# for k, v in d.items():
	# 	if v == 'False':
	# 		d.update({k: False})
	# 	elif v == 'True':
	# 		d.update({k: True})
	# 	elif v == 'None':
	# 		d.update({k: None})
	run_experiment(args.p_file[0], computation_function=args.c_function[0])