import sys
sys.path.append('/home/neuro/Desktop/CODE/network_simulation_testbed/projects/encoding_decoding')
sys.path.append('/home/neuro/Desktop/CODE/network_simulation_testbed')
import matplotlib
matplotlib.use('Agg')
from modules.parameters import *
from modules.analysis import *
from computations import stimulus_processing

stimulus_processing.run('./ED_spikepatterninput_training_parameters_lexicon_size=360_T=10000.txt', **{'plot': False, 'save': True, 'display': True})