import sys
sys.path.append('../../')
import matplotlib
matplotlib.use('Agg')
import pprint

from Modules.parameters import *
pp = pprint.PrettyPrinter(indent=2)

def print_parameter_set(fn, ps):
	with open(fn, 'w') as f:
		f.write(ps.pretty())

def test_single_neuron_dcinput_parameterspace(ref_fn, test_fn):
	"""
	Test function for ParameterSpace object. Input is 'single_neuron_dcinput', make sure there is no random stuff
	left in the input script. Below are the currently hard-coded variables + values which can be used for testing.
	// np_seed = 100
	// msd = 500
	:return:
	"""

	# compute current parameter space
	ref_pars 	= ParameterSpace(ref_fn)
	test_pars 	= ParameterSpace(test_fn, new_config=True)

	assert sorted([hash(elem.pretty()) for elem in ref_pars.parameter_sets]) == \
		   sorted([hash(elem.pretty()) for elem in test_pars.parameter_sets])

	# if sorted([hash(elem.pretty()) for elem in ref_pars.parameter_sets]) !=\
	# 		sorted([hash(elem.pretty()) for elem in test_pars.parameter_sets]):
		# print_parameter_set("tmp_old", ref_pars.parameter_sets[1])
		# print_parameter_set("tmp_new", test_pars.parameter_sets[1])

	assert str(ref_pars.dimensions) == str(test_pars.dimensions)
	assert str(ref_pars.label) == str(test_pars.label)



# ######################################################################################################################
# Run tests
test_single_neuron_dcinput_parameterspace("../../Experiments/ParameterSets/single_neuron_dcinput.py",
										  "../../Experiments/ParameterSets/new_template.py")
