#!/usr/bin/python

import matplotlib
matplotlib.use('Agg')
import pickle
import importlib
from modules.parameters import *
from optparse import OptionParser


def run_experiment(params_file_full_path, computation_function="noise_driven_dynamics", **parameters):
	"""

	:param params_file_full_path: full path to parameter file
	:param computation_function: which experiment to run
	:param parameters: other CLI input parameters
	:return:
	"""
	try:
		experiment = importlib.import_module("Computation." + computation_function)
	except:
		print("Could not find experiment `%s`. Is it in ./Computation/ directory?" % computation_function)
		exit(-1)

	if 'keep_all' in parameters.keys():
		pars = ParameterSpace(params_file_full_path, keep_all=parameters['keep_all'])
		parameters.pop('keep_all')
	else:
		pars = ParameterSpace(params_file_full_path)

	pars.save(pars[0].kernel_pars.data_path+pars[0].kernel_pars.data_prefix+'_ParameterSpace.py')
	# if hasattr(pars[0], "report_pars"):
	# 	pars.compile_parameters_table()
	if pars[0].kernel_pars.system['local']:
		results = pars.run(experiment.run, **parameters)
		return results
	else:
		pars.run(experiment.run, **parameters)


def run_emoo(params_file_full_path, computation_function="noise_driven_dynamics",
			 results_subfields=['spiking_activity', 'Global'], operation=np.mean,
			 objectives={'cv_isis': [0.8, 2.0], 'ccs': [-0.1, 0.1], 'mean_rate': 10.}, **parameters):
	"""

	:param params_file_full_path: full path to parameter file
	:param computation_function: which experiment to run
	:param results_subfields:
	:param operation:
	:param objectives:
	:param parameters: other CLI input parameters
	:return:
	"""
	try:
		experiment = importlib.import_module("Computation." + computation_function)
	except Exception as error:
		print("Could not find experiment `%s`. Is it in ./Computation/ directory?" % computation_function)
		exit(-1)

	pars = ParameterSpace(params_file_full_path, emoo=True)
	pars.save(pars[0].kernel_pars.data_path+pars[0].kernel_pars.data_prefix+'_ParameterSpace.py')
	optimization_parameters = {'n_generations': 20,
	                           'n_individuals': 10,
	                           'pop_capacity': 20,
	                           'eta_m_0': 20,
	                           'eta_c_0': 20,
	                           'p_m': 0.5}
	emoo_obj, results = pars.run_emoo(experiment.run, objectives, optimization_parameters,
	                                  results_subfields=results_subfields, operation=operation, **parameters)

	with open(pars[0].kernel_pars.data_path + pars[0].kernel_pars.data_prefix + '_EMOO_Results.pck', 'w') as fp:
		pickle.dump(results, fp)


def create_parser():
	parser_ = OptionParser()
	parser_.add_option("-f", "--param-file", dest="p_file", help="parameter file", metavar="FILE")
	parser_.add_option("-c", "--computation-function", dest="c_function", help="computation function to execute")

	return parser_


def print_welcome_message():
	print("""
	  Welcome to Network Simulation Testbed!

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