__author__ = 'duarte'
from modules.parameters import ParameterSet, ParameterSpace, copy_dict
from modules.signals import empty, iterate_obj_list
from modules.visualization import *
from modules.io import set_project_paths
from modules.analysis import dimensionality_reduction
from defaults.paths import paths
import numpy as np
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.manifold import LocallyLinearEmbedding, Isomap, SpectralEmbedding, MDS, TSNE
sys.path.append('../')
from stimulus_generator import StimulusPattern
import pickle
from auxiliary_functions import rebuild_stimulus_sequence
import time

"""
state_matrix_analysis
- read data recorded from stimulus_processing experiments
- read state matrix and analyse it
"""

# data parameters
project = 'encoding_decoding'
data_type = 'spikepatterninput' # 'dcinput' #
data_path = '/media/neuro/Data/EncodingDecoding_OUT/nStimStudy/'
state_variable = 'spikes' #'V_m'
data_label = 'ED_{0}_nStimStudy'.format(data_type)
dataset = '{0}_lexicon_size=10_trial=0'.format(data_label)
results_path = data_path + data_label + '/Results/'

# set defaults and paths
set_project_paths(project)
set_global_rcParams(paths['local']['matplotlib_rc'])
colormap = 'Accent'

# reconstruct ParameterSpace
pars_file = data_path + data_label + '_ParameterSpace.py'
pars_space = ParameterSpace(pars_file)

# load results
with open(results_path + 'Results_' + dataset, 'r') as fp:
	results = pickle.load(fp)

# retrieve ParameterSet
pars = ParameterSet(data_path + data_label + '/Parameters/Parameters_' + dataset)

# retrieve input sequence (it's not the right one!!)
# Language, Input, Output = StimulusPattern(pars.task_pars).load(FileName=data_path + data_label +
#                                                      '/Inputs/Task_identity mapping_Delay_3_LexSize_10.txt',
#                                             as_index=False)
# full_input_seq = Input[pars.stim_pars.transient_set_length:pars.stim_pars.transient_set_length +
#                                                            pars.stim_pars.train_set_length]
# unique_labels = np.unique(full_input_seq)
if data_type == 'spikepatterninput':
	stim_seq = rebuild_stimulus_sequence(results['epochs'])
else:
	with open(data_path + data_label + '/Inputs/StimulusSet.pkl', 'r') as fp:
		stim = pickle.load(fp)
	stim_seq = stim.full_set_labels

label_seq = np.array(list(iterate_obj_list(stim_seq[10:-2000])))
n_elements = np.unique(label_seq)

# retrieve state matrix
state_matrix = np.load(data_path + data_label + '/Activity/' + dataset + '_populationEI_state{0}_train.npy'.format(
state_variable))


# #######################################################################
# analyse state matrix - project to a 3d space
# =======================================================================
# 1) PCA
dimensionality_reduction(state_matrix, data_label=dataset, labels=label_seq, metric='PCA', standardize=False, plot=True,
                             colormap=colormap, display=True, save=data_path + data_label +
                                                                   '/Figures/' + state_variable)

# 2) Factor Analysis
dimensionality_reduction(state_matrix, data_label=dataset, labels=label_seq, metric='FA', standardize=False, plot=True,
                             colormap=colormap, display=True, save=data_path + data_label +
                                                                   '/Figures/' + state_variable)

# 3) Locally-Linear Embedding
dimensionality_reduction(state_matrix, data_label=dataset, labels=label_seq, metric='LLE', standardize=False, plot=True,
                             colormap=colormap, display=True, save=data_path + data_label + '/Figures/' + state_variable)

# 4) Isomap embedding
dimensionality_reduction(state_matrix, data_label=dataset, labels=label_seq, metric='IsoMap', standardize=False,
                         plot=True, colormap=colormap, display=True, save=data_path + data_label +
                                                                          '/Figures/' + state_variable)

# 5) Spectral Embedding
dimensionality_reduction(state_matrix, data_label=dataset, labels=label_seq, metric='Spectral', standardize=False,
                         plot=True, colormap=colormap, display=True, save=data_path + data_label +
                                                                          '/Figures/' + state_variable)

# 6) MDS
dimensionality_reduction(state_matrix, data_label=dataset, labels=label_seq, metric='MDS', standardize=False, plot=True,
                             colormap=colormap, display=True, save=data_path + data_label +
                                                                   '/Figures/' + state_variable)

# 7) t-SNE
dimensionality_reduction(state_matrix, data_label=dataset, labels=label_seq, metric='t-SNE', standardize=False,
                         plot=True, colormap=colormap, display=True, save=data_path + data_label +
                                                                          '/Figures/' + state_variable)





