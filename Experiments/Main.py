import sys
sys.path.append('../')
import matplotlib
matplotlib.use('Agg')
from Modules.parameters import *
import pickle
import importlib

def run_experiment(params_file_full_path, computation_function="noise_driven_dynamics", **parameters):
	"""

	:param params_file_full_path:
	:param computation_function:
	:param parameters:
	:return:
	"""
	try:
		experiment = importlib.import_module("Computation." + computation_function)
	except Exception as error:
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

	:param params_file_full_path:
	:param computation_function:
	:param parameters:
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


if __name__ == "__main__":
	# we need minimum 2 arguments (3 including the script)
	if len(sys.argv) < 3:
		print("At least two arguments are required! Exiting..")
		exit(-1)

	for arg in sys.argv[3:]:
		print arg
	d = dict([arg.split('=', 1) for arg in sys.argv[3:]])
	for k, v in d.items():
		if v == 'False':
			d.update({k: False})
		elif v == 'True':
			d.update({k: True})
		elif v == 'None':
			d.update({k: None})
	run_experiment(sys.argv[1], computation_function=sys.argv[2], **d)