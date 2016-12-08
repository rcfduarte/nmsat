import sys
sys.path.append('../../')
sys.path.append('../../experiments/')
sys.path.append('../../experiments/parameter_sets/')
import matplotlib
matplotlib.use('Agg')
import pprint
import subprocess

from modules.parameters import *
pp = pprint.PrettyPrinter(indent=2)

def print_parameter_set(fn, ps):
	with open(fn, 'w') as f:
		f.write(ps.pretty())

def cmp_setVSspace(ref_fn, test_fn):
	ref_pars 	= ParameterSet(set_params_dict(ref_fn), label='global')
	ref_pars 	= ref_pars.clean(termination="pars")
	test_pars 	= ParameterSpace(test_fn, new_config=True)
	if hash(ref_pars.pretty()) != hash(test_pars.parameter_sets[0].pretty()):
		print("ERROR\n")
		with open("set.tmp", 'w') as f:
			f.write(ref_pars.pretty())
		with open("space.tmp", 'w') as f:
			f.write(test_pars.parameter_sets[0].pretty())

		subprocess.call(["meld", "set.tmp", "space.tmp"])
		assert False


def cmp_spaceVSspace(ref_fn, test_fn):
	ref_pars 	= ParameterSpace(ref_fn)
	test_pars 	= ParameterSpace(test_fn, new_config=True)

	if ref_pars.parameter_sets[0] != test_pars.parameter_sets[0]:
		print("ERROR\n")
		with open("set.tmp", 'w') as f:
			f.write(ref_pars.parameter_sets[0].pretty())
		with open("space.tmp", 'w') as f:
			f.write(test_pars.parameter_sets[0].pretty())

		subprocess.call(["meld", "set.tmp", "space.tmp"])
		assert False

	assert str(ref_pars.dimensions) == str(test_pars.dimensions)
	assert str(ref_pars.label) == str(test_pars.label)



# ######################################################################################################################
# Run tests
# test_single_neuron_dcinput_parameterspace("../../Experiments/ParameterSets/single_neuron_dcinput.py",
# 										  "../../Experiments/ParameterSets/new_template.py")

# def test_spike_pattern_input_sequence():
	# cmp_setVSspace("../../Experiments/ParameterSets/_originals/X_spike_pattern_input_sequence.py",
	# 			   "../../Experiments/ParameterSets/spike_pattern_input_sequence.py")

def test_spike_pattern_input_sequence():
	cmp_spaceVSspace("../../Experiments/ParameterSets/_originals/X_spike_pattern_input_sequence.py",
					 "../../Experiments/ParameterSets/spike_pattern_input_sequence.py")

if __name__ == "__main__":
	test_spike_pattern_input_sequence()