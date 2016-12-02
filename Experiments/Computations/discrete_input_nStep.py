from __init__ import *

def run(parameter_set, plot=False, display=False, save=False, debug=False, online=True):
	"""
	Run the RC sequence processing task
	:param parameter_set:
	:param plot:
	:param display:
	:param save:
	:param debug:
	:return:
	"""
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
	net = Network(parameter_set.net_pars)

	###################################################################################
	# Randomize initial variable values
	# =================================================================================
	for n in list(iterate_obj_list(net.populations)):
		n.randomize_initial_states('V_m', randomization_function=np.random.uniform, low=-70., high=-55.)
	# n.randomize_initial_states('V_th', randomization_function=np.random.normal, loc=-50., scale=1.)

	###################################################################################
	# Build and connect input
	# =================================================================================
	# Create StimulusSet
	stim_set_time = time.time()
	stim = StimulusSet(parameter_set, unique_set=False)
	stim.create_set(parameter_set.stim_pars.full_set_length)
	stim.discard_from_set(parameter_set.stim_pars.transient_set_length)
	stim.divide_set(parameter_set.stim_pars.transient_set_length, parameter_set.stim_pars.train_set_length,
	                parameter_set.stim_pars.test_set_length)
	print "- Elapsed Time: {0}".format(str(time.time() - stim_set_time))

	# Create InputSignalSet
	input_set_time = time.time()
	inputs = InputSignalSet(parameter_set, stim, online=online)
	if stim.transient_set_labels:
		inputs.generate_transient_set(stim)
		parameter_set.kernel_pars.transient_t = inputs.transient_stimulation_time
	if not online:
		inputs.generate_full_set(stim)
	#inputs.generate_unique_set(stim)
	inputs.generate_train_set(stim)
	inputs.generate_test_set(stim)
	print "- Elapsed Time: {0}".format(str(time.time() - input_set_time))

	# Plot example signal
	if plot and debug and not online:
		fig_inp = pl.figure()
		ax1 = fig_inp.add_subplot(211)
		ax2 = fig_inp.add_subplot(212)
		fig_inp.suptitle('Input Stimulus / Signal')
		inp_plot = vis.InputPlots(stim_obj=stim, input_obj=inputs.train_set_signal, noise_obj=inputs.train_set_noise)
		inp_plot.plot_stimulus_matrix(set='train', ax=ax1, save=False, display=False)
		inp_plot.plot_input_signal(ax=ax2, save=paths['figures'] + paths['label'], display=display)
		inp_plot.plot_input_signal(save=paths['figures'] + paths['label'], display=display)
		inp_plot.plot_signal_and_noise(save=paths['figures'] + paths['label'], display=display)
	parameter_set.kernel_pars.sim_time = inputs.train_stimulation_time + inputs.test_stimulation_time

	if save:
		stim.save(paths['inputs'])
		if debug:
			inputs.save(paths['inputs'])
	#######################################################################################
	# Encode Input
	# =====================================================================================
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

	######################################################################################
	# Connect Network
	# ====================================================================================
	net.connect_populations(parameter_set.connection_pars)

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
	if stim.transient_set_labels:
		if not online:
			print "\nTransient time = {0} ms".format(str(parameter_set.kernel_pars.transient_t))

		iterate_input_sequence(net, inputs.transient_set_signal, enc_layer, sampling_times=None, stim_set=stim,
		                       input_set=inputs, set_name='transient', store_responses=False, record=False)
		parameter_set.kernel_pars.transient_t = nest.GetKernelStatus()['time']
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
		# results.update(evaluate_encoding(enc_layer, parameter_set, analysis_interval,
		#                                  inputs.transient_set_signal, plot=plot, display=display,
		#                                  save=paths['figures']+paths['label']))

		net.flush_records()
		enc_layer.flush_records()

	#######################################################################################
	# Simulate (Train period)
	# =====================================================================================
	if not online:
		print "\nTrain time = {0} ms".format(str(inputs.train_stimulation_time))
	iterate_input_sequence(net, inputs.train_set_signal, enc_layer,
	                       sampling_times=parameter_set.decoding_pars.global_sampling_times,
	                       stim_set=stim, input_set=inputs, set_name='train', store_responses=False)

	#######################################################################################
	# Train Readouts
	# =====================================================================================
	train_all_readouts(parameter_set, net, stim, inputs.train_set_signal, encoding_layer=enc_layer, flush=True,
	                   debug=debug, plot=plot, display=display, save=paths)

	#######################################################################################
	# Simulate (Test period)
	# =====================================================================================
	if not online:
		print "\nTest time = {0} ms".format(str(inputs.test_stimulation_time))
	iterate_input_sequence(net, inputs.test_set_signal, enc_layer,
	                       sampling_times=parameter_set.decoding_pars.global_sampling_times,
	                       stim_set=stim, input_set=inputs, set_name='test',
	                       store_responses=False)

	#######################################################################################
	# Test Readouts
	# =====================================================================================
	test_all_readouts(parameter_set, net, stim, inputs.test_set_signal, encoding_layer=enc_layer, flush=False,
	                  debug=debug, plot=plot, display=display, save=paths)

	results.update(analyse_performance_results(net, enc_layer, plot=plot, display=display, save=paths['figures'] +
	                  paths['label']))

	#######################################################################################
	# Save data
	# =====================================================================================
	if save:
		with open(paths['results'] + 'Results_' + parameter_set.label, 'w') as f:
			pickle.dump(results, f)
		parameter_set.save(paths['parameters'] + 'Parameters_' + parameter_set.label)

