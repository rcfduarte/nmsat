"""
========================================================================================================================
Auxiliary Module
========================================================================================================================
Utility functions that are frequently used in specific experiments

Functions:
------------

Classes:
------------

========================================================================================================================
Copyright (C) 2018  Renato Duarte, Barna Zajzon

Neural Mircocircuit Simulation and Analysis Toolkit is free software;
you can redistribute it and/or modify it under the terms of the
GNU General Public License as published by the Free Software Foundation;
either version 2 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

The GNU General Public License does not permit this software to be
redistributed in proprietary programs.

"""
import signals
import input_architect
import analysis
import numpy as np
import itertools
import time
import nest

from multiprocessing.dummy import Pool as ThreadPool


########################################################################################################################
# Auxiliary Functions
########################################################################################################################
# TODO comment
def set_sampling_parameters(sampling_times, input_signal_set, input_signal):
    """

    :param sampling_times:
    :param input_signal_set:
    :param input_signal:
    :return:
    """
    # TODO remove when implemented
    if sampling_times is not None:
        raise NotImplementedError('Sampling times is not yet available, use sampling_offsets until then!')

    if sampling_times is None and not input_signal_set.online:
        t_samp = np.sort(list(signals.iterate_obj_list(input_signal.offset_times)))  # extract stimulus offset times
        sub_sampling_times = None
    elif sampling_times is None and input_signal_set.online:
        t_samp = [round(nest.GetKernelStatus()['time'])]  # offset times will be specified online, in the main
    # iteration
        sub_sampling_times = None
    elif sampling_times is not None and input_signal_set.online:
        t_samp = [round(nest.GetKernelStatus()['time'])]
        sub_sampling_times = sampling_times
    else:
        t_samp = sampling_times
        sub_sampling_times = None
    return t_samp, sub_sampling_times


def set_decoder_times(enc_layer, input_pars, encoding_pars, decoding_pars,
                      sampling_interval=None, correct_offset=None):
    """
    This is a rewritten version of the 'set_decoder_times' function below, with proper offsetting
    and cleaner if case handling.

    :param enc_layer: [EncodingLayer]
    :param input_pars:
    :param encoding_pars: [ParameterSet]
    :param decoding_pars: [ParameterSet]
    :param sampling_interval: [float]?
    :param correct_offset: [float or int] sets the sampling offset for the extractors
    :return:
    """
    # set flag that this function was called
    decoding_pars.update({'decoder_times_set': True})

    enc_layer.determine_total_delay()
    encoder_delay   = enc_layer.total_delay
    stim_duration   = input_pars.signal.durations
    stim_isi        = input_pars.signal.i_stim_i

    # be careful here, bool is also int
    if isinstance(correct_offset, float) or isinstance(correct_offset, int):
        add_offset = correct_offset
    else:
        add_offset = 0.

    if sampling_interval is not None:
        duration = sampling_interval
    else:
        # TODO - other variants
        if (len(stim_duration) != 1 and np.mean(stim_duration) != stim_duration[0]) or all(stim_isi):
            raise NotImplementedError("Stimulus durations should be fixed and constant and "
                                      "inter-stimulus intervals == 0.")
        else:
            duration = stim_duration[0]

    # iterate through each decoded variable and set the appropriate offset
    for extractor_idx, extractor_pars in enumerate(decoding_pars.state_extractor.state_specs):
        state_variable = decoding_pars.state_extractor.state_variable[extractor_idx]
        offset = add_offset + encoder_delay

        # decoding the filtered spiketrains implies an extra delay of 0.1 due to the delta neurons
        if state_variable == 'spikes':
            offset += 0.1

        extractor_pars.update({'offset': offset, 'interval': duration})
        print("Extractor {0}: \n- offset = {1} ms\n- interval = {2}".format(state_variable, str(offset), str(duration)))

    if not signals.empty(enc_layer.encoders) and hasattr(encoding_pars, "input_decoder") and \
                    encoding_pars.input_decoder is not None:
        for extractor_idx, extractor_pars in enumerate(encoding_pars.input_decoder.state_extractor.state_specs):
            state_variable = encoding_pars.input_decoder.state_extractor.state_variable[extractor_idx]
            offset = add_offset

            if state_variable == 'spikes':
                offset += 0.1

            extractor_pars.update({'offset': offset, 'interval': duration})
            print("Encoder Extractor {0}: \n- offset = {1} ms\n- interval = {2}".format(state_variable, str(offset),
                                                                                        str(duration)))


def retrieve_data_set(set_name, stimulus_set, input_signal_set):
    """
    Extract the properties of the dataset to be used
    :param set_name:
    :param stimulus_set:
    :param input_signal_set:
    :return:
    """
    if set_name is None:
        set_name = "full"
    all_labels = getattr(stimulus_set, "{0}_set_labels".format(set_name))
    if isinstance(all_labels[0], list):
        labels = np.unique(list(itertools.chain(*all_labels)))
        set_labels = list(itertools.chain(*all_labels))
    else:
        labels = np.unique(all_labels)
        set_labels = all_labels
    set_size = len(set_labels)
    input_signal = getattr(input_signal_set, "{0}_set_signal".format(set_name))
    stimulus_seq = getattr(stimulus_set, "{0}_set".format(set_name))
    if input_signal_set.online:
        signal_iterator         = getattr(input_signal_set, "{0}_set_signal_iterator".format(set_name))
        analog_signal_iterator  = getattr(input_signal_set, "{0}_set".format(set_name))
    else:
        signal_iterator         = None
        analog_signal_iterator  = None
    return labels, set_labels, set_size, input_signal, stimulus_seq, signal_iterator, analog_signal_iterator


def retrieve_stimulus_timing(input_signal_set, stim_seq_pos, set_size, signal_iterator,
                             t_samp, state_sample_time, input_signal):
    """
    Extract all relevant timing information from the current signal.

    :param input_signal_set: [InputSignalSet]
    :param stim_seq_pos: [int]  position
    :param set_size: [int] size of current stimulus set (full, train or test)
    :param signal_iterator: []
    :param t_samp: [list] ???
    :param state_sample_time: [
    :param input_signal: [InputSignal]

    :return: local signal, stimulus_duration, stimulus_onset, t_samp, state_sample_time, simulation_time
    """
    if input_signal_set.online and stim_seq_pos < set_size:
        local_signal = signal_iterator.next()
        # local signal is the same as input signal. next() as iterator just updates the amplitudes and stimulus times
        assert local_signal is input_signal, "Local signal must be equal to the input signal, this is a major problem.."

        stimulus_duration = list(itertools.chain(*local_signal.durations))[0]
        stimulus_onset = t_samp[-1] # prior to adding new step
        local_signal.time_offset(stimulus_onset)
        stimulus_offset = list(itertools.chain(*local_signal.offset_times))[0]
        t_samp.append(stimulus_offset)
        state_sample_time = stimulus_offset # new time
        simulation_time = stimulus_duration # stimulus duration..
        if local_signal.intervals[-1]:
            simulation_time += local_signal.intervals[-1]
    else:
        local_signal = None
        simulation_time = state_sample_time
        stimulus_duration = None
        stimulus_onset = 0.1 if stim_seq_pos == 0 else t_samp[stim_seq_pos - 1]

        if stim_seq_pos < len(t_samp) - 1:
            if input_signal.intervals[stim_seq_pos]:
                simulation_time += input_signal.intervals[stim_seq_pos]

    return local_signal, stimulus_duration, stimulus_onset, t_samp, state_sample_time, simulation_time


def update_spike_template(enc_layer, idx, input_signal_set, stimulus_set, local_signal, t_samp, input_signal, jitter,
                          stimulus_onset):
    """
    Read the current stimulus identity, extract the corresponding spike pattern, jitter if necessary, offset to the
    stimulus onset time and update the spike generators.

    :param enc_layer:
    :param idx:
    :param input_signal_set:
    :param stimulus_set:
    :param local_signal:
    :param t_samp:
    :param input_signal:
    :param jitter:
    :param stimulus_onset:
    :return:
    """
    assert (len(input_signal_set.spike_patterns) == stimulus_set.dims), \
        "Incorrect number of spike patterns"

    if input_signal_set.online and local_signal is not None:
        stimulus_id = [nx for nx in range(stimulus_set.dims) if t_samp[-1] in local_signal.offset_times[nx]]
    else:
        stimulus_id = [nx for nx in range(stimulus_set.dims) if t_samp[idx] in input_signal.offset_times[nx]]
    spike_pattern = input_signal_set.spike_patterns[stimulus_id[0]].copy()

    if jitter is not None:
        if jitter[1]:  # compensate for boundary effects
            spike_pattern.jitter(jitter[0])
            resize_window = spike_pattern.time_parameters()
            spikes = spike_pattern.time_slice(resize_window[0] + jitter[0], resize_window[1] - jitter[0])
            spikes.time_offset(-jitter[0])
        else:
            spikes = spike_pattern.jitter(jitter[0])
    else:
        spikes = spike_pattern

    spikes = spikes.time_offset(stimulus_onset, True)
    enc_layer.update_state(spikes)


def update_input_signals(enc_layer, local_signal, next_analog_signals_it, dt,
                         noise=False, noise_parameters=None):
    """
    Retrieves the analog signal for the next step and updates state of encoding layer generators accordingly.
    If noise is needed, it is added to the input signal before updating the encoding layer.

    :param enc_layer: [EncodingLayer]
    :param local_signal: [InputSignal]
    :param next_analog_signals_it: iterator
    :param dt: simulation resolution
    :param noise: [bool] whether noise should be added to the signal
    :param noise_parameters:

    :return:
    """
    # get analog signal for the next step
    local_signal.input_signal = next_analog_signals_it.next()
    local_signal.time_offset(dt)

    # add noise if needed and update encoding layer
    if noise:
        assert(noise_parameters is not None), "Noise parameters must be provided!"
        local_signal.input_signal = add_noise(local_signal, noise_parameters)

    enc_layer.update_state(local_signal.input_signal, ids_to_update=list(local_signal.input_signal.id_list()))


def add_noise(local_signal, noise_parameters):
    """
    Add a new noise realization to each step.

    :param local_signal:
    :param noise_parameters:
    :return:
    """
    local_noise = input_architect.InputNoise(noise_parameters, start_time=local_signal.input_signal.t_start,
                                             stop_time=local_signal.input_signal.t_stop+10)
    local_noise.generate()
    signal_array = local_signal.input_signal.as_array()
    noise_array = local_noise.noise_signal.as_array()[:, :signal_array.shape[1]]
    new_signal_array = signal_array + noise_array
    return signals.convert_array(new_signal_array, id_list=local_signal.input_signal.id_list(), dt=local_signal.dt,
                                 start=local_signal.input_signal.t_start, stop=local_signal.input_signal.t_stop)


def extract_state_vectors(net, enc_layer, sample_time):
    """

    :param net:
    :param enc_layer:
    :param sample_time:
    :return:
    """
    # Extract and store state vectors
    for ctr, n_pop in enumerate(list(itertools.chain(*[net.merged_populations,
                                                       net.populations, enc_layer.encoders]))):
        if n_pop.decoding_layer is not None:
            n_pop.decoding_layer.extract_state_vector(time_point=round(sample_time, 2), save=True) # TODO- choose
        # round precision


def flush(net, enc_layer, decoders=True):
    """
    Clears the devices attached to the network and encoding layer, optionally together with the decoders.

    :param net: [Network]
    :param enc_layer: [EncodingLayer]
    :param decoders: [bool]
    :return:
    """
    net.flush_records(decoders=decoders)
    enc_layer.flush_records(decoders=decoders)


def compile_results_multithreaded(net, enc_layers, store_start, time_correction_factor, record, store_activity,
                                  store_decoders=False, skip_first_sample=False):
    """
    Compiles results (state matrix) and extracts the activities of the populations / network.

    :param net: [Network]
    :param enc_layers: [list] list of EncodingLayer objects
    :param store_start:
    :param time_correction_factor:
    :param record: [bool]
    :param store_activity: [bool] store spiking and analog activities of the populations (continuous)
    :param store_decoders: [bool] store decoded variables (filtered spike trains, V_m) at sampled times
    :param skip_first_sample: [bool] whether to drop first element in state matrix (due to 'offset' parameter)
    :return:
    """
    extraction_time = time.time()

    def _extract_multithreaded(_pop):
        print('Extracting population {}'.format(_pop.name))
        _pop.decoding_layer.extract_activity(start=0., stop=nest.GetKernelStatus()['time'], save=True)
        for idx_state, n_state in enumerate(_pop.decoding_layer.state_variables):
            _pop.decoding_layer.state_matrix[idx_state] = _pop.decoding_layer.activity[idx_state].as_array()

            if skip_first_sample:
                _pop.decoding_layer.state_matrix[idx_state] = \
                    _pop.decoding_layer.state_matrix[idx_state][:, 1:]
        print('Finished extracting population {}'.format(_pop.name))

    if record:
        # compile state matrices
        all_populations = list(itertools.chain(*[net.merged_populations, net.populations,
                                                 list(itertools.chain(*[e.encoders for e in enc_layers]))]))
        thread_args = [pop for pop in all_populations if pop.decoding_layer]
        print('\n\n###################\n\nMULTITHREADED EXTRACTION\n\n')

        pool = ThreadPool(len(thread_args))
        pool.map(_extract_multithreaded, thread_args)
        pool.close()
        pool.join()

    if store_activity:
        # store full activity
        net.extract_population_activity(t_start=store_start,
                                        t_stop=nest.GetKernelStatus()['time'] - time_correction_factor)
        net.extract_network_activity()
        # enc_layer.extract_encoder_activity(t_start=t0,
        #                                    t_stop=nest.GetKernelStatus()['time'] - time_correction_factor)

        if not signals.empty(net.merged_populations):
            net.merge_population_activity(start=store_start, save=True,
                                          stop=nest.GetKernelStatus()['time'] - time_correction_factor)
        if store_decoders:
            for ctr, n_pop in enumerate(list(itertools.chain(*[net.merged_populations,
                                                               net.populations]))): #, enc_layer.encoders]))):
                if n_pop.decoding_layer is not None:
                    n_pop.decoding_layer.extract_activity(start=store_start, save=True,
                                                          stop=nest.GetKernelStatus()['time'] - time_correction_factor)

    print('Extraction time (multithreaded): {}'.format(time.time() - extraction_time))


def compile_results(net, enc_layers, store_start, time_correction_factor, record, store_activity,
                    store_decoders=False, skip_first_sample=False):
    """
    Compiles results (state matrix) and extracts the activities of the populations / network.

    :param net: [Network]
    :param enc_layers: [list] list of EncodingLayer objects
    :param store_start:
    :param time_correction_factor:
    :param record: [bool]
    :param store_activity: [bool] store spiking and analog activities of the populations (continuous)
    :param store_decoders: [bool] store decoded variables (filtered spike trains, V_m) at sampled times
    :param skip_first_sample: [bool] whether to drop first element in state matrix (due to 'offset' parameter)
    :return:
    """
    extraction_time = time.time()
    if record:
        # compile state matrices
        for n_pop in list(itertools.chain(*[net.merged_populations, net.populations,
                                            list(itertools.chain(*[e.encoders for e in enc_layers]))])):
            if n_pop.decoding_layer is not None:
                n_pop.decoding_layer.extract_activity(start=0., stop=nest.GetKernelStatus()['time'], save=True)
                for idx_state, n_state in enumerate(n_pop.decoding_layer.state_variables):
                    n_pop.decoding_layer.state_matrix[idx_state] = n_pop.decoding_layer.activity[idx_state].as_array()

                    # remove first sample from state matrix if necessary
                    # TODO we should also remove the very first analog signals, no? from the decoding_layer.activity
                    # or is the state matrix enough...
                    if skip_first_sample:
                        n_pop.decoding_layer.state_matrix[idx_state] = \
                            n_pop.decoding_layer.state_matrix[idx_state][:, 1:]

    # TODO this needs to be agreed upon how the store_start is actually handled... (take encoding delays into account?)
    # TODO probably the proper store_start must be set in the calling function
    if store_activity:
        # store full activity, both spiking and analog if recorded
        net.extract_population_activity(t_start=store_start,
                                        t_stop=nest.GetKernelStatus()['time'] - time_correction_factor)
        net.extract_network_activity()
        # enc_layer.extract_encoder_activity(t_start=t0,
        #                                    t_stop=nest.GetKernelStatus()['time'] - time_correction_factor)

        # store activity for merged populations if there are any
        if not signals.empty(net.merged_populations):
            net.merge_population_activity(start=store_start, save=True,
                                          stop=nest.GetKernelStatus()['time'] - time_correction_factor)
        if store_decoders:
            for ctr, n_pop in enumerate(list(itertools.chain(*[net.merged_populations,
                                                               net.populations]))): #, enc_layer.encoders]))):
                if n_pop.decoding_layer is not None:
                    n_pop.decoding_layer.extract_activity(start=store_start, save=True,
                                                          stop=nest.GetKernelStatus()['time'] - time_correction_factor)
    print('Extraction time (single-threaded): {}'.format(time.time() - extraction_time))


def time_keep(start_time, idx, set_size, t1):
    """
    Measure current elapsed time and remaining time
    :return:
    """
    t2 = time.time()
    total_time_elapsed = t2 - start_time
    cycle_count = idx + 1
    avg_cycle_time = total_time_elapsed / cycle_count
    cycles_remaining = set_size - cycle_count
    time_remaining = avg_cycle_time * cycles_remaining
    print("\nTime information: ")
    print("- Current step time: %.2f mins." % ((t2 - t1) / 60.))
    print("- Total elapsed time: %.2f mins." % (total_time_elapsed / 60.))
    print("- Estimated time remaining: %.2f mins." % (time_remaining / 60.))

    return (t2 - t1) / 60.


# TODO don't rely on certain parameter names in the parameter set, e.g., decoding_pars or encoding_pars
def process_input_sequence(parameter_set, net, enc_layers, stimulus_sets, input_signal_sets,
                               set_name, record=True, store_activity=None, sampling_offset=None):
    """

    :param parameter_set:
    :param net: [Network]
    :param enc_layers: list of encoding layers
    :param stimulus_sets: list of stimulus sets
    :param input_signal_sets: list of input signal sets
    :param set_name:
    :param record: [bool] - acquire and store state matrices (according to sampling_times and state characteristics)
    :param store_activity: [int] - record population activity for a given number of simulation epochs, starting
        from the end. NOTE! - this can lead to extreme memory consumption!
    :param sampling_offset:
    :return:
    """
    print("\n\n***** Preparing to simulate {0} set *****".format(set_name))

    # check if sampling times have been set (set_decoder_times(...))
    assert parameter_set.decoding_pars.has_key('decoder_times_set'), "Setting sampling parameters (offset) is required!"

    assert isinstance(enc_layers, list), "Must be list!"
    assert isinstance(stimulus_sets, list), "Must be list!"
    assert isinstance(input_signal_sets, list), "Must be list!"

    # determine timing compensations required
    assert  len(np.unique([e.total_delay for e in enc_layers])) == 1, "Heterogeneous encoding delays not supported!"
    encoder_delay   = enc_layers[0].total_delay
    decoder_delays  = []

    for pop in list(itertools.chain(*[net.merged_populations, net.populations,
                                        list(itertools.chain(*[e.encoders for e in enc_layers]))])):
        if pop.decoding_layer is not None:
            pop.decoding_layer.determine_total_delay()
            decoder_delays.append(pop.decoding_layer.total_delays)

    decoder_delay = max(list(itertools.chain(*decoder_delays)))
    sim_res = nest.GetKernelStatus()['resolution']

    # extract important parameters:
    sampling_times = parameter_set.decoding_pars.sampling_times

    jitter = []
    # extract jitter parameters
    for e in enc_layers:
        j = e.parameters.generator.jitter if hasattr(e.parameters.generator, "jitter") else None
        jitter.append(j)

    signal_noise_params = []
    signal_noise = []
    # extract noise parameters
    for inp_set in input_signal_sets:
        if hasattr(inp_set.parameters, "noise") and inp_set.parameters.noise.N:
            signal_noise.append(True)
            signal_noise_params.append(inp_set.parameters.noise)
        else:
            signal_noise.append(False)
            signal_noise_params.append(None)

    # init some lists here
    t_samp          = []
    labels          = []
    set_labels      = []
    set_size        = []
    input_signal    = []
    signal_it       = []
    next_analog_signals_it = []

    # determine set to use and its properties
    for idx in range(len(input_signal_sets)):
        retrieved_data = retrieve_data_set(set_name, stimulus_sets[idx], input_signal_sets[idx])
        labels.append(retrieved_data[0])
        set_labels.append(retrieved_data[1])
        set_size.append(retrieved_data[2])
        input_signal.append(retrieved_data[3])
        signal_it.append(retrieved_data[5])
        next_analog_signals_it.append(retrieved_data[6])

    # set state sampling parameters
    # TODO there's a hidden assumption here, that sampling and subsampling times are equal across signal sets
    for idx in range(len(input_signal_sets)):
        t_samp_tmp, sub_sampling_times = set_sampling_parameters(sampling_times,
                                                                 input_signal_sets[idx], input_signal[idx])
        t_samp.append(t_samp_tmp)

    t0          = nest.GetKernelStatus()['time'] + encoder_delay
    start_time  = time.time()
    timing      = dict(step_time=[], total_time=[])
    epochs      = []

    # intialize epochs dictionary
    for idx in range(len(input_signal_sets)):
        epochs.append({k: [] for k in labels[idx]})

    assert type(record) is int or type(record) is bool, "Parameter `record` must be integer or boolean!"
    store = False  # flag controlling actual activity recording, will be updated accordingly during simulation

    if record:
        print("\n!!! State matrices will be extracted and stored !!!")
    else:
        print("\n!!! No state matrices will be stored !!!")

    if store_activity:
        print("\n!!! The activity from the last {0} epochs will be stored !!!".format(record))

    ####################################################################################################################
    # one sample for each stimulus (acquired at the last time point of each stimulus)
    if sampling_times is None:
        print("\n\nSimulating {0} steps".format(str(set_size)))
        stim_seq_id = 0
        sim_time = 0.0

        # ################################ Main Loop ###################################
        while stim_seq_id < set_size[0]:  # TODO [0] here should be fixed...
            state_sample_time = t_samp[0][stim_seq_id]
            stim_start = time.time()

            print("\n\nSimulating step {0} /".format(str(stim_seq_id + 1)))
            for i in range(len(set_labels)):
                print ("\t- {0} - stimulus {1} [{2} ms]".format(set_size[i], set_labels[i][stim_seq_id], sim_time))

            # generate input for each of the input signal sets / stimulus sets
            for idx in range(len(input_signal_sets)):
                local_signal, stimulus_duration, stimulus_onset, t_samp[idx], state_sample_time, sim_time = \
                retrieve_stimulus_timing(input_signal_sets[idx], stim_seq_id, set_size[idx], signal_it[idx],
                                             t_samp[idx], state_sample_time, input_signal[idx])

                epochs[idx][set_labels[idx][stim_seq_id]].append((stimulus_onset, state_sample_time))

                # update inputs / encoders
                if all(['spike_pattern' in n or 'noise' in n for n in
                        list(signals.iterate_obj_list(enc_layers[idx].generator_names))]):
                    update_spike_template(enc_layers[idx], stim_seq_id, input_signal_sets[idx], stimulus_sets[idx],
                                          local_signal, t_samp[idx], input_signal[idx], jitter[idx], stimulus_onset)
                elif input_signal_sets is not None and local_signal is not None and input_signal_sets[idx].online:
                    update_input_signals(enc_layers[idx], local_signal, next_analog_signals_it[idx],
                                         sim_res, signal_noise[idx], signal_noise_params[idx])

            state_sample_time += encoder_delay  # correct sampling time

                # simulate main step
            net.simulate(sim_time)

            # move on to next input
            stim_seq_id += 1

            # reset encoding layer decoders
            for e in enc_layers:
                analysis.reset_decoders(net, e)

                # add sample time to decoders
                for pop in list(itertools.chain(*[net.merged_populations, net.populations, e.encoders])):
                    if pop.decoding_layer is not None and sampling_times is None:
                        pop.decoding_layer.sampled_times.append(state_sample_time)
                    elif pop.decoding_layer is not None and (isinstance(sampling_times, list) or isinstance(
                            sampling_times, np.ndarray)):
                        sample_times = set(stimulus_onset + sampling_times)
                        samp_times = set(pop.decoding_layer.sampled_times)
                        pop.decoding_layer.sampled_times = list(np.sort(list(samp_times.union(sample_times))))

            # update start of activity recording if needed
            if store_activity and set_size[0] - stim_seq_id == store_activity:
                t0 = nest.GetKernelStatus()['time']

            # flush unnecessary information
            if not store_activity or set_size[0] - stim_seq_id > store_activity:
                for e in enc_layers:
                    flush(net, e, decoders=bool(not record))

            # simulate the last delay step if we're at the end (last stimulus)
            if stim_seq_id == set_size[0]:
                net.simulate(encoder_delay + decoder_delay)
                # we also need to go one step (= simulation resolution) further to get the very last sample!
                net.simulate(sim_res)

            timing['step_time'].append(time_keep(start_time, stim_seq_id, set_size[0], stim_start))

        timing['total_time'] = (time.time() - start_time) / 60.
        time_correction_factor = encoder_delay + decoder_delay

        # correct_first_sample: if sampling at stimulus offset is intended, we need to ignore the
        # first sampled time by the multimeter (because of the way the offset parameter is implemented in NEST),
        # otherwise we do have to include the first value as well
        compile_results(net, enc_layers, t0, time_correction_factor, record,
                        store_activity=store_activity, skip_first_sample=bool(not sampling_offset))

        # compile_results_multithreaded(net, enc_layers, t0, time_correction_factor, record,
        #                               store_activity=store_activity, skip_first_sample=bool(not sampling_offset))

    return epochs, timing


# TODO allow for multiple targets per population
def process_states(net, target_matrix_map, stim_set, data_sets=None, accepted_idx=None, save_raw_readout=False,
                   evaluation_method=None, plot=False, display=True, save=False, save_paths=None, extra_label=''):
    """
    Post-processing step to set the correct timings of state samples, divide and re-organize dataset. After train and
    test data sets are given, after training the readout accordingly the performance is measured and results extracted
    and returned in a dictionary.

    :param net: Network object
    :param target_matrix_map: [dict] pop_name -> target
    :param stim_set: [list] stimulus sets
    :param data_sets: [list] list of strings with data set names, e.g. ['transient', 'train', 'test']
    :param accepted_idx:
    :param evaluation_method: [string] method to evaluate performance, e.g. `k-WTA` or `threshold`
    :param plot:
    :param display:
    :param save:
    :param save_paths:
    :param extra_label:
    :return: [dict] dictionary of performance results with three top-level keys, 'rank', 'performance'
        and 'dimensionality'.
    """
    results = dict(rank={}, performance={}, dimensionality={})

    # TODO add more checks for data_set and state matrix consistency!!!
    if data_sets is None:
        data_sets = ["transient", "unique", "train", "test"]

    start_idx = 0
    # iterate over each given data set
    for set_name in data_sets:
        if hasattr(stim_set, "{0}_set_labels".format(set_name)):
            labels = getattr(stim_set, "{0}_set_labels".format(set_name))
            if isinstance(labels[0], list):
                labels = list(itertools.chain(*getattr(stim_set, "{0}_set_labels".format(set_name))))
            set_start = start_idx
            set_end = len(labels) + set_start
            start_idx += len(labels)

            if accepted_idx is not None:
                accepted_ids = []
                for idx in accepted_idx:
                    if set_end > idx >= set_start:
                        accepted_ids.append(idx - set_start)
            else:
                accepted_ids = None

            # we now iterate over each population in the target_matrix_map and train one (for now) target for each
            for ctr, pop_name in enumerate(target_matrix_map.keys()):
                n_pop = net.get_population_by_name(pop_name)

                if n_pop.decoding_layer is None:
                    continue

                # initialize population specific entries in results dictionary
                results['rank'].update({n_pop.name: {}})
                results['performance'].update({n_pop.name: {}})
                results['dimensionality'].update({n_pop.name: {}})

                print("\nProcessing states from {0} set of population {1}".format(set_name, pop_name))

                # parse state variables
                for idx_var, var in enumerate(n_pop.decoding_layer.state_variables):
                    state_matrix = n_pop.decoding_layer.state_matrix[idx_var][:, set_start:set_end]
                    readouts = n_pop.decoding_layer.readouts[idx_var]

                    if target_matrix_map[pop_name] is not None:
                        target = target_matrix_map[pop_name][:, set_start:set_end]
                    else:
                        target = None

                    if accepted_ids is not None:
                        assert (len(accepted_ids) == target.shape[1]), "Incorrect {0} set labels or accepted " \
                                                                       "ids".format(set_name)

                    results['performance'][n_pop.name].update({var + str(idx_var): {}})
                    results['dimensionality'][n_pop.name].update({var + str(idx_var): {}})

                    print("\nPopulation {0}, variable {1}, set {2}: {3}".format(n_pop.name, var, set_name,
                                                                                str(state_matrix.shape)))
                    if set_name == 'unique':
                        results['rank'][n_pop.name].update({var + str(idx_var): analysis.get_state_rank(state_matrix)})

                    # if it's the training set, train readout and measure stability
                    elif set_name == 'train':
                        assert target is not None
                        for readout in readouts:
                            if readout.name[-1].isdigit(): # memory
                                readout.set_index()
                                print("{0}, {1}".format(readout.name, readout.index))

                            readout.train(state_matrix, np.array(target), index=readout.index,
                                          accepted=accepted_ids, display=display)

                            readout.measure_stability(display=display)
                            if plot and save:
                                readout.plot_weights(display=display, save=save_paths['figures'] + save_paths['label'])
                            elif plot:
                                readout.plot_weights(display=display, save=False)

                    # if it's the testing set, test readout performance on the set and store results in dictionary
                    elif set_name == 'test':
                        for readout in readouts:
                            assert target is not None
                            print("{0}, {1}".format(readout.name, readout.index))
                            output, tgt = readout.test(state_matrix, np.array(target), index=readout.index,
                                                       accepted=accepted_ids, display=display)
                            result_perf = results['performance'][n_pop.name][var + str(idx_var)]

                            ##############################################
                            if save and save_raw_readout:
                                perf_save = '{0}{1}_population{2}_state{3}_xlabel={4}'.format(
                                    save_paths['results'], save_paths['label'], n_pop.name, var, extra_label)
                            else:
                                perf_save = False

                            result_perf.update({
                                readout.name: readout.measure_performance(tgt, output, evaluation_method,
                                                                          display=display, save=perf_save)})
                            result_perf[readout.name].update( {'norm_wOut': readout.norm_wout})

                        results['dimensionality'][n_pop.name].update(
                            {var + str(idx_var): analysis.compute_dimensionality(state_matrix)})

                    if plot and set_name != 'transient':
                        if save:
                            analysis.analyse_state_matrix(state_matrix, labels, label=n_pop.name + var + set_name,
                                                          plot=plot, display=display,
                                                          save=save_paths['figures'] + save_paths['label'])
                        else:
                            analysis.analyse_state_matrix(state_matrix, labels, label=n_pop.name + var + set_name,
                                                          plot=plot, display=display, save=False)

                    if save and set_name != 'transient':
                        np.save(save_paths['activity'] + save_paths['label'] +
                                '_population{0}_state{1}_{2}.npy'.format(n_pop.name, var, set_name), state_matrix)
    return results
