__author__ = 'duarte'
from modules.parameters import ParameterSet, extract_nestvalid_dict
from modules.input_architect import EncodingLayer, InputNoise
from modules.net_architect import Network
from modules.io import set_storage_locations
from modules.signals import iterate_obj_list
from modules.analysis import single_neuron_responses
from modules import visualization as vis
import cPickle as pickle
import numpy as np
import nest


def run(parameter_set, plot=False, display=False, save=True):
    """
    Compute single neuron fI curves
    :param parameter_set: must be consistent with the computation
    :param plot: plot results - either show them or save to file
    :param display: show figures/reports
    :param save: save results
    :return results_dictionary:
    """
    if not isinstance(parameter_set, ParameterSet):
        if isinstance(parameter_set, basestring) or isinstance(parameter_set, dict):
            parameter_set = ParameterSet(parameter_set)
        else:
            raise TypeError("parameter_set must be ParameterSet, string with full path to parameter file or "
                            "dictionary")

    # ##################################################################################################################
    # Setup extra variables and parameters
    # ==================================================================================================================
    if plot:
        vis.set_global_rcParams(parameter_set.kernel_pars['mpl_path'])
    paths = set_storage_locations(parameter_set, save)

    np.random.seed(parameter_set.kernel_pars['np_seed'])

    # ##################################################################################################################
    # Set kernel and simulation parameters
    # ==================================================================================================================
    print('\nRuning ParameterSet {0}'.format(parameter_set.label))
    nest.ResetKernel()
    nest.set_verbosity('M_WARNING')
    nest.SetKernelStatus(extract_nestvalid_dict(parameter_set.kernel_pars.as_dict(), param_type='kernel'))

    # ##################################################################################################################
    # Build network
    # ==================================================================================================================
    net = Network(parameter_set.net_pars)

    # ##################################################################################################################
    # Randomize initial variable values
    # ==================================================================================================================
    for idx, n in enumerate(list(iterate_obj_list(net.populations))):
        if hasattr(parameter_set.net_pars, "randomize_neuron_pars"):
            randomize = parameter_set.net_pars.randomize_neuron_pars[idx]
            for k, v in randomize.items():
                n.randomize_initial_states(k, randomization_function=v[0], **v[1])

    ####################################################################################################################
    # Build Input Signal Sets
    # ==================================================================================================================
    assert hasattr(parameter_set, "input_pars")

    total_stimulation_time = parameter_set.kernel_pars.sim_time + parameter_set.kernel_pars.transient_t

    # Current input (need to build 2 separate noise signals for the 2 input channels)
    # Generate input for channel 1
    input_noise_ch1 = InputNoise(parameter_set.input_pars.noise, rng=np.random, stop_time=total_stimulation_time)
    input_noise_ch1.generate()
    input_noise_ch1.re_seed(parameter_set.kernel_pars.np_seed)

    # Generate input for channel 2
    input_noise_ch2 = InputNoise(parameter_set.input_pars.noise, rng=np.random, stop_time=total_stimulation_time)
    input_noise_ch2.generate()
    input_noise_ch2.re_seed(parameter_set.kernel_pars.np_seed)

    if plot:
        inp_plot = vis.InputPlots(stim_obj=None, input_obj=None, noise_obj=input_noise_ch1)
        inp_plot.plot_noise_component(display=display, save=paths['figures'] + "/InputNoise_CH1")

        inp_plot = vis.InputPlots(stim_obj=None, input_obj=None, noise_obj=input_noise_ch2)
        inp_plot.plot_noise_component(display=display, save=paths['figures'] + "/InputNoise_CH2")

    # ##################################################################################################################
    # Build and connect input
    # ==================================================================================================================
    enc_layer_ch1 = EncodingLayer(parameter_set.encoding_ch1_pars, signal=input_noise_ch1)
    enc_layer_ch1.connect(parameter_set.encoding_ch1_pars, net)

    enc_layer_ch2 = EncodingLayer(parameter_set.encoding_ch2_pars, signal=input_noise_ch2)
    enc_layer_ch2.connect(parameter_set.encoding_ch2_pars, net)

    # ##################################################################################################################
    # Connect Devices
    # ==================================================================================================================
    net.connect_devices()

    # ##################################################################################################################
    # Simulate
    # ==================================================================================================================
    if parameter_set.kernel_pars.transient_t:
        net.simulate(parameter_set.kernel_pars.transient_t)
        net.flush_records()

    net.simulate(parameter_set.kernel_pars.sim_time + nest.GetKernelStatus()['resolution'])

    # ##################################################################################################################
    # Extract and store data
    # ==================================================================================================================
    net.extract_population_activity(t_start=parameter_set.kernel_pars.transient_t,
                                    t_stop=parameter_set.kernel_pars.sim_time + parameter_set.kernel_pars.transient_t)
    net.extract_network_activity()

    # ##################################################################################################################
    # Analyse / plot data
    # ==================================================================================================================
    results = dict()

    analysis_interval = [parameter_set.kernel_pars.transient_t,
                         parameter_set.kernel_pars.transient_t + parameter_set.kernel_pars.sim_time]
    for idd, nam in enumerate(net.population_names):
        results.update({nam: {}})
        results[nam] = single_neuron_responses(net.populations[idd],
                                               parameter_set, pop_idx=idd,
                                               start=analysis_interval[0],
                                               stop=analysis_interval[1],
                                               plot=plot, display=display,
                                               save=paths['figures'] + paths['label'])
        if results[nam]['rate']:
            print('Output Rate [{0}] = {1} spikes/s'.format(str(nam), str(results[nam]['rate'])))

    # ##################################################################################################################
    # Save data
    # ==================================================================================================================
    if save:
        with open(paths['results'] + 'Results_' + parameter_set.label, 'w') as f:
            pickle.dump(results, f)
        parameter_set.save(paths['parameters'] + 'Parameters_' + parameter_set.label)
