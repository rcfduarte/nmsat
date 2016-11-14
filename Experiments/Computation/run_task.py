from __init__ import *

def run(parameter_set, plot=True, display=True, save=False, online=True, debug=False):
	"""
	:param parameter_set:
	:param plot:
	:param display:
	:param save:
	:param online:
	:param debug:
	:return:
	"""
	from Projects.EncodingDecoding.StimulusGenerator.PatternGenerator import StimulusPattern
	from Projects.EncodingDecoding.Specific_extra_scripts.auxiliary_functions import train_readouts, test_readouts

	if not isinstance(parameter_set, ParameterSet):
		if isinstance(parameter_set, basestring) or isinstance(parameter_set, dict):
			parameter_set = ParameterSet(parameter_set)
		else:
			raise TypeError("parameter_set must be ParameterSet, string with full path to parameter file or dictionary")

	###################################################################################
	# Setup extra variables and parameters
	# =================================================================================
	if plot:
		import Modules.visualization as vis
		vis.set_global_rcParams(parameter_set.kernel_pars['mpl_path'])
	paths = set_storage_locations(parameter_set, save)

	np.random.seed(parameter_set.kernel_pars['np_seed'])
	results = dict()

	##################################################################################
	# Set kernel and simulation parameters
	# ================================================================================
	print '\nRuning ParameterSet {0}'.format(parameter_set.label)
	nest.ResetKernel()
	nest.set_verbosity('M_WARNING')
	nest.SetKernelStatus(extract_nestvalid_dict(parameter_set.kernel_pars.as_dict(), param_type='kernel'))

	###################################################################################
	# Build network
	# =================================================================================
	start_build = time.time()
	net = Network(parameter_set.net_pars)
	net.merge_subpopulations([net.populations[0], net.populations[1]], name='EI')

	########################################################################################################################
	# Randomize initial variable values
	# =======================================================================================================================
	for idx, n in enumerate(list(iterate_obj_list(net.populations))):
		if hasattr(parameter_set.net_pars, "randomize_neuron_pars"):
			randomize = parameter_set.net_pars.randomize_neuron_pars[idx]
			for k, v in randomize.items():
				n.randomize_initial_states(k, randomization_function=v[0], **v[1])

	###################################################################################
	# Build Stimulus Set
	# =================================================================================
	stim_set_startbuild = time.time()

	# Create or Load StimulusPattern
	stim_pattern = StimulusPattern(parameter_set.task_pars)
	stim_pattern.generate()

	input_sequence, output_sequence = stim_pattern.as_index()

	# Convert to StimulusSet object
	stim_set = StimulusSet(unique_set=None)
	stim_set.load_data(input_sequence, type='full_set_labels')
	stim_set.discard_from_set(parameter_set.stim_pars.transient_set_length)
	stim_set.divide_set(parameter_set.stim_pars.transient_set_length, parameter_set.stim_pars.train_set_length,
	                    parameter_set.stim_pars.test_set_length)

	# Specify target and convert to StimulusSet object
	target_set = StimulusSet(unique_set=None)
	target_set.load_data(output_sequence, type='full_set_labels')
	target_set.discard_from_set(parameter_set.stim_pars.transient_set_length)
	target_set.divide_set(parameter_set.stim_pars.transient_set_length, parameter_set.stim_pars.train_set_length,
	                      parameter_set.stim_pars.test_set_length)

	print "- Elapsed Time: {0}".format(str(time.time() - stim_set_startbuild))
	stim_set_buildtime = time.time() - stim_set_startbuild
	###################################################################################
	# Build Input Signal Set
	# =================================================================================
	input_set_time = time.time()
	parameter_set.input_pars.signal.N = len(np.unique(input_sequence))
	# Create InputSignalSet
	inputs = InputSignalSet(parameter_set, stim_set, online=online)
	if not empty(stim_set.transient_set_labels):
		inputs.generate_transient_set(stim_set)
		parameter_set.kernel_pars.transient_t = inputs.transient_stimulation_time
	if not online:
		inputs.generate_full_set(stim_set)
	# inputs.generate_unique_set(stim)
	inputs.generate_train_set(stim_set)
	inputs.generate_test_set(stim_set)

	# Plot example signal
	if plot and debug and not online:
		fig_inp = pl.figure()
		ax1 = fig_inp.add_subplot(211)
		ax2 = fig_inp.add_subplot(212)
		fig_inp.suptitle('Input Stimulus / Signal')
		inp_plot = vis.InputPlots(stim_obj=stim_set, input_obj=inputs.test_set_signal, noise_obj=inputs.test_set_noise)
		inp_plot.plot_stimulus_matrix(set='test', ax=ax1, save=False, display=False)
		inp_plot.plot_input_signal(ax=ax2, save=paths['figures'] + paths['label'], display=display)
		inp_plot.plot_input_signal(save=paths['figures'] + paths['label'], display=display)
		inp_plot.plot_signal_and_noise(save=paths['figures'] + paths['label'], display=display)
	parameter_set.kernel_pars.sim_time = inputs.train_stimulation_time + inputs.test_stimulation_time

	if save:
		stim_pattern.save(paths['inputs'])
		stim_set.save(paths['inputs'])
		if debug:
			inputs.save(paths['inputs'])

	print "- Elapsed Time: {0}".format(str(time.time() - input_set_time))
	inputs_build_time = time.time() - input_set_time
	#######################################################################################
	# Encode Input
	# =====================================================================================
	encoder_start_time = time.time()
	if not online:
		input_signal = inputs.full_set_signal
	else:
		input_signal = inputs.transient_set_signal
	enc_layer = EncodingLayer(parameter_set.encoding_pars, signal=input_signal, online=online)
	enc_layer.connect(parameter_set.encoding_pars, net)

	# Attach decoders to input encoding populations
	if not empty(enc_layer.encoders) and hasattr(parameter_set.encoding_pars, "input_decoder"):
		enc_layer.connect_decoders(parameter_set.encoding_pars.input_decoder)

	if plot and debug:
		vis.extract_encoder_connectivity(enc_layer, net, display, save=paths['figures'] + paths['label'])

	#######################################################################################
	# Set-up Analysis
	# =====================================================================================
	net.connect_devices()
	net.connect_decoders(parameter_set.decoding_pars)
	encoding_time = time.time() - encoder_start_time

	######################################################################################
	# Connect Network
	# ====================================================================================
	net.connect_populations(parameter_set.connection_pars)
	build_time = time.time() - start_build
	if plot and debug:
		fig_W = pl.figure()
		topology = vis.TopologyPlots(parameter_set.connection_pars, net)
		topology.print_network(depth=3)
		ax1 = pl.subplot2grid((6, 6), (0, 0), rowspan=4, colspan=4)
		ax2 = pl.subplot2grid((6, 6), (5, 0), rowspan=1, colspan=4)
		ax3 = pl.subplot2grid((6, 6), (0, 5), rowspan=4, colspan=1)
		ax4 = pl.subplot2grid((6, 6), (5, 5), rowspan=1, colspan=1)
		topology.plot_connectivity(parameter_set.connection_pars.synapse_types,
		                           ax=[ax1, ax3, ax2, ax4], display=display, save=paths['figures'] + paths['label'])

	######################################################################################
	# Simulate (Initial Transient)
	# ====================================================================================
	if not empty(stim_set.transient_set_labels):
		if not online:
			print "\nTransient time = {0} ms".format(str(parameter_set.kernel_pars.transient_t))

		iterate_input_sequence(net, inputs.transient_set_signal, enc_layer, sampling_times=None, stim_set=stim_set,
		                       input_set=inputs, set_name='transient', store_responses=False, record=False)
		parameter_set.kernel_pars.transient_t = nest.GetKernelStatus()['time']
		inputs.transient_stimulation_time = nest.GetKernelStatus()['time']
		net.extract_population_activity(t_start=0., t_stop=parameter_set.kernel_pars.transient_t -
		                                                   parameter_set.kernel_pars.resolution)
		net.extract_network_activity()

		# sanity check
		activity = []
		for spikes in net.spiking_activity:
			activity.append(spikes.mean_rate())
		if not np.mean(activity) > 0:
			raise ValueError("No activity recorded in main network! Stopping simulation..")

		if parameter_set.kernel_pars.transient_t > 1000.:
			analysis_interval = [1000, parameter_set.kernel_pars.transient_t]
			results['population_activity'] = population_state(net, parameter_set=parameter_set,
			                                                  nPairs=500, time_bin=1.,
			                                                  start=analysis_interval[0],
			                                                  stop=analysis_interval[1] -
			                                                       parameter_set.kernel_pars.resolution,
			                                                  plot=plot, display=display,
			                                                  save=paths['figures'] + paths['label'])
			enc_layer.extract_encoder_activity()
			results.update(evaluate_encoding(enc_layer, parameter_set, analysis_interval,
			                                 inputs.transient_set_signal, plot=plot, display=display,
			                                 save=paths['figures'] + paths['label']))

		net.flush_records()
		enc_layer.flush_records()

	#######################################################################################
	# Simulate (Train period)
	# =====================================================================================
	train_start = time.time()
	if not online:
		print "\nFull time = {0} ms".format(str(inputs.full_stimulation_time))
	iterate_input_sequence(net, inputs.train_set_signal, enc_layer,
	                       sampling_times=parameter_set.decoding_pars.global_sampling_times,
	                       stim_set=stim_set, input_set=inputs, set_name='train', store_responses=False)

	if online:
		inputs.train_stimulation_time = nest.GetKernelStatus()['time'] - inputs.transient_stimulation_time

	#######################################################################################
	# Train Readouts
	# =====================================================================================
	train_readouts(parameter_set, net, stim_pattern, stim_set, target_set, encoding_layer=enc_layer, flush=True,
	               debug=debug, plot=plot, display=display, save=paths)
	train_time = time.time() - train_start

	#######################################################################################
	# Simulate (Test period)
	# =====================================================================================
	test_start = time.time()
	if not online:
		print "\nTest time = {0} ms".format(str(inputs.test_stimulation_time))
	iterate_input_sequence(net, inputs.test_set_signal, enc_layer,
	                       sampling_times=parameter_set.decoding_pars.global_sampling_times,
	                       stim_set=stim_set, input_set=inputs, set_name='test', store_responses=False)
	if online:
		inputs.test_stimulation_time = nest.GetKernelStatus()['time'] - (inputs.transient_stimulation_time +
		                                                                 inputs.train_stimulation_time)

	#######################################################################################
	# Test Readouts
	# =====================================================================================
	test_readouts(parameter_set, net, stim_pattern, stim_set, target_set, encoding_layer=enc_layer, flush=True,
	              debug=debug, plot=plot, display=display, save=paths)
	results['performance'] = analyse_performance_results(net, enc_layer, plot=plot, display=display, save=paths[
		                                                                                                      'figures'] +
	                                                                                                      paths[
		                                                                                                      'label'])
	test_time = time.time() - test_start

	#######################################################################################
	# Save Performance data
	# =====================================================================================
	if save:
		with open(paths['results'] + 'Results_' + parameter_set.label, 'w') as f:
			pickle.dump(results, f)
		parameter_set.save(paths['parameters'] + 'Parameters_' + parameter_set.label)

	#######################################################################################
	# Analyse population responses (during test phase)
	# =====================================================================================
	analysis_time = time.time()
	analysis_interval = [inputs.transient_stimulation_time + inputs.train_stimulation_time,
	                     inputs.transient_stimulation_time + inputs.train_stimulation_time
	                     + inputs.test_stimulation_time]
	results['activity_simple'] = population_state(net, parameter_set, nPairs=500, time_bin=1.,
	                                              start=analysis_interval[0],
	                                              stop=analysis_interval[1], plot=plot, display=display,
	                                              save=paths['figures'] + paths['label'])
	#######################################################################################
	# Save simple activity analysis data
	# =====================================================================================
	if save:
		with open(paths['results'] + 'Results_' + parameter_set.label, 'w') as f:
			pickle.dump(results, f)

	results['activity'] = characterize_population_activity(net, parameter_set, analysis_interval, epochs=None,
	                                                       time_bin=1., summary_only=True, time_resolved=False,
	                                                       window_len=100, color_map='Accent', plot=False,
	                                                       display=display, save=paths['figures'] + paths['label'])
	analysis_time = time.time() - analysis_time

	########################################################################################
	# Store time information
	# =====================================================================================
	results['time'] = {'build_time': build_time, 'stim_build': stim_set_buildtime,
	                   'inputs_build': inputs_build_time, 'train': train_time,
	                   'test': test_time, 'analysis': analysis_time}

	#######################################################################################
	# Save all data
	# =====================================================================================
	if save:
		with open(paths['results'] + 'Results_' + parameter_set.label, 'w') as f:
			pickle.dump(results, f)
		parameter_set.save(paths['parameters'] + 'Parameters_' + parameter_set.label)
