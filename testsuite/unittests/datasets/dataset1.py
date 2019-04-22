"""

"""

from preset import *

n_stim = 3

stim_pars = dict(
    n_stim=n_stim,
    elements= np.arange(0, n_stim, 1),
    grammar= None,
    full_set_length=int(11),
    transient_set_length=int(1),
    train_set_length=int(8),
    test_set_length=int(2))

# Input Parameters
inp_resolution = .1
inp_amplitude = 1500.
inp_duration = 200.
inter_stim_interval = 0.

lexicon_size = n_stim

input_pars = {
    'signal': {
        'N': lexicon_size,
        'durations': [inp_duration],
        'i_stim_i': [inter_stim_interval],
        'kernel': ('box', {}),
        'start_time': 0.,
        'stop_time': sys.float_info.max,
        'max_amplitude': [inp_amplitude],
        'min_amplitude': 0.,
        'resolution': inp_resolution},
    # 'noise': {
    # 	'N': lexicon_size,
    # 	'noise_source': ['GWN'],
    # 	'noise_pars': {'amplitude': 5., 'mean': 1., 'std': 0.25},
    # 	'rectify': False,
    # 	'start_time': 0.,
    # 	'stop_time': sys.float_info.max,
    # 	'resolution': inp_resolution, }
}

encoding_pars = {
    "connectivity": {
        "weight_dist": [1.0],
        "syn_specs": [{}],
        "models": ['static_synapse'],
        "delay_dist": [0.1],
        "conn_specs": [{'p': 0.1, 'rule': 'pairwise_bernoulli'}],
        "label": "global",
        "connections": [('E1I1', 'inhomogeneous_poisson')],
        "model_pars": [{}],
        "topology_dependent": [False],
        "synapse_name": ['my_synapse'],
        "preset_W": [None],
    },
    "label": "global",
    "generator": {
        "models": ['inhomogeneous_poisson_generator'],
        "labels": ['inhomogeneous_poisson'],
        "N": 1,
        "model_pars": [{'origin': 0.0, 'start': 0.0, 'stop': 1.7976931348623157e+308}],
        "topology_pars": [None],
        "label": "global",
        "topology": [False],
    },
    "encoder": {
        "labels": [],
        "record_analogs": [],
        "models": [],
        "n_neurons": [],
        "topology_dict": [],
        "label": "global",
        "model_pars": [],
        "neuron_pars": [],
        "spike_device_pars": [],
        "record_spikes": [],
        "N": 0,
        "analog_device_pars": [],
        "topology": [],
    }
}