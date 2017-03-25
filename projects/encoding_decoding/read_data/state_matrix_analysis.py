__author__ = 'duarte'
from modules.parameters import ParameterSet, copy_dict
from modules.signals import empty
from modules.visualization import *
from modules.io import set_project_paths
from modules.analysis import analyse_state_matrix
from defaults.paths import paths
import numpy as np
sys.path.append('../')
from stimulus_generator import StimulusPattern

"""
state_matrix_analysis
- read data recorded from stimulus_processing experiments
- read state matrix and analyse it
"""

# data parameters
project = 'encoding_decoding'
data_type = 'spikepatterninput' # 'dcinput' #
data_path = '/media/neuro/Data/EncodingDecoding_OUT/nStimStudy/'
state_variable = 'V_m'
data_label = 'ED_{0}_nStimStudy'.format(data_type)
dataset = '{0}_lexicon_size=10_trial=0'.format(data_label)

# set defaults and paths
set_project_paths(project)
set_global_rcParams(paths['local']['matplotlib_rc'])

# reconstruct ParameterSpace

# retrieve ParameterSet
pars = ParameterSet(data_path + data_label + '/Parameters/Parameters_' + dataset)

# retrieve input sequence (it's not the right one!!)
Language, Input, Output = StimulusPattern(pars.task_pars).load(FileName=data_path + data_label +
                                                     '/Inputs/Task_identity mapping_Delay_3_LexSize_10.txt',
                                            as_index=False)
full_input_seq = Input[pars.stim_pars.transient_set_length:pars.stim_pars.transient_set_length +
                                                           pars.stim_pars.train_set_length]

# retrieve state matrix
state_matrix = np.load(data_path + data_label + '/Activity/' + dataset + '_populationEI_state{0}_train.npy'.format(
state_variable))



manifold_learning(state_matrix, 100, standardize=False, plot=True, display=True, save=False)
# analyse_state_matrix(state_matrix, Input, label='', plot=True, display=True, save=False)


# re-create ParameterSpace
# pars_file = data_path + data_label + '_ParameterSpace.py'
# pars = ParameterSpace(pars_file)



# print the full nested structure of the results dictionaries (to harvest only specific results)
# pars.print_stored_keys(data_path + data_label + '/Results/')

# retrieve input sequence

# retrieve state matrix
