"""
NET Architect Module

Classes
-------------
Population                - Population object used to handle the simulated neuronal populations
Network                   - Network class is one of the central classes, containing lists of elements specifying
							the properties of each subpopulation (including the population objects)
Functions
-------------
verify_pars_consistency   - verify if all the relevant lists in a parameter set have the same dimensionality
iterate_obj_list          - build an iterator to go through the elements of a list or nested list
extract_weights_matrix    - extract synaptic weights matrix
"""
import sys
import visualization
import parameters
import analysis
import signals
import io
import itertools
import time
import copy
import numpy as np
from scipy.sparse import lil_matrix
import nest
from nest import topology as tp


def verify_pars_consistency(pars_set, not_allowed_keys, n=0):
	"""
	Verify if all the relevant list have the same dimensionality
	:param pars_set: parameter dictionary (or set) to verify
	:param not_allowed_keys: keys to remove from the dictionary (shouldn't have the same dimensions)
	:param n: target dimensionality
	:return:
	:raises ValueError is dimensionality is not consistent
	"""
	clean_dict = {k: v for k, v in pars_set.iteritems() if k not in not_allowed_keys}
	# return np.mean([len(v) for v in clean_dict.itervalues()]) == n
	for k, v in clean_dict.iteritems():
		if len(v) != n:
			raise ValueError("Dimensionality of parameters ({0}) is inconsistent!".format(str(k)))


def extract_weights_matrix(src_gids, tgets_gids, progress=True):
	"""
	Extract the synaptic weights matrix referring to connections from src_gids to tgets_gids
	:param src_gids: list or tuple of gids of source neurons
	:param tgets_gids: list or tuple of gids of target neurons
	:param progress: display progress bar
	:return: len(tgets_gids) x len(src_gids) weight matrix
	"""
	if progress:
		print("\n Extracting connectivity (weights) matrix...")
	t_start = time.time()
	w = lil_matrix((len(tgets_gids), len(src_gids)))
	a = nest.GetConnections(list(np.unique(src_gids)), list(np.unique(tgets_gids)))

	iterations = 100
	its = np.arange(0, len(a)+iterations, iterations).astype(int)
	its[-1] = len(a)
	for nnn, it in enumerate(its):
		if nnn < len(its) - 1:
			con = a[it:its[nnn+1]]
			st = nest.GetStatus(con, keys='weight')
			for idx, n in enumerate(con):
				w[n[1] - min(tgets_gids), n[0] - min(src_gids)] += st[idx]
		if progress:
			visualization.progress_bar(float(nnn+1) / float(len(its)))
	t_stop = time.time()
	if progress:
		print("Elapsed time: %s" % (str(t_stop - t_start)))
	# for consistency with pre_computedW, we transpose this matrix (should be [src X tget])
	return w.T


def extract_delays_matrix(src_gids, tgets_gids, progress=True):
	"""
	Extract the synaptic weights matrix referring to connections from src_gids to tgets_gids
	:param src_gids: list or tuple of gids of source neurons
 	:param tgets_gids: list or tuple of gids of target neurons
 	:param progress: display progress bar
 	:return: len(tgets_gids) x len(src_gids) weight matrix
 	"""
	if progress:
		print("\n Extracting connectivity (delays) matrix...")
	t_start = time.time()
	d = lil_matrix((len(tgets_gids), len(src_gids)))
	a = nest.GetConnections(list(np.unique(src_gids)), list(np.unique(tgets_gids)))

	iterations = 100
	its = np.arange(0, len(a)+iterations, iterations).astype(int)
	its[-1] = len(a)
	for nnn, it in enumerate(its):
		if nnn < len(its) - 1:
			con = a[it:its[nnn + 1]]
			st = nest.GetStatus(con, keys='delay')
			for idx, n in enumerate(con):
				d[n[1] - min(tgets_gids), n[0] - min(src_gids)] = st[idx]
		if progress:
			visualization.progress_bar(float(nnn+1) / float(len(its)))
	t_stop = time.time()
	if progress:
		print("Elapsed time: %s" % (str(t_stop - t_start)))
	return d


########################################################################################################################
class Population(object):
	"""
	Population object used to handle each of the simulated neuronal populations
	Contains the parameters of each population (name, size, topology, gids);
	allows recording devices to be connected to the population (all neurons or a
	randomly sampled subset), according to the parameters specifications (see record_spikes
	and record_analog)
	After the simulation is terminated or a short period of simulation is run, the records
	can be deleted with flush_records() or converted into a SpikeList object, which
	will remain a property of the population object (using the function activity_set())

	Input:
		- pop_set -> ParameterSet for this population
	"""

	def __init__(self, pop_set):

		self.name = pop_set.pop_names
		self.size = pop_set.n_neurons
		if pop_set.topology:
			self.layer_gid = pop_set.layer_gid
			self.topology_dict = pop_set.topology_dict
		self.topology = pop_set.topology
		self.gids = pop_set.gids
		self.spiking_activity = []
		self.analog_activity = []
		self.is_subpop = pop_set.is_subpop
		self.attached_devices = []
		self.attached_device_names = []
		self.analog_activity_names = []
		self.decoding_layer = None

		#self.decoding_pars = []
		#self.state_extractors = []
		#self.readouts = []
		#self.state_matrix = []
		#self.state_sample_times = []
		#self.response_matrix = []
		#self.state_variables = []

	def randomize_initial_states(self, var_name, randomization_function, **function_parameters):
		"""
		Randomize the initial value of a specified variable, property of the neurons in this population
		applying the function specified, with the corresponding parameters
		:param var_name: [str] - name of the variable to randomize
		:param randomization_function: [function] - randomization function to generate the values
		:param function_parameters: extra parameters of the function

		example:
		--------
		>> n.randomize_initial_states('V_m', randomization_function=np.random.uniform,
		 low=-70., high=-55.)
		"""
		assert var_name in nest.GetStatus(self.gids)[0].keys(), "Variable name not in object properties"
		print("\n- Randomizing {0} state in Population {1}".format(str(var_name), str(self.name)))
		try:
			nest.SetStatus(self.gids, var_name, randomization_function(size=len(self.gids), **function_parameters))
		except:
			for n_neuron in self.gids:
				success = False
				while not success:
					try:
						nest.SetStatus([n_neuron], var_name, randomization_function(size=1, **function_parameters))
						success = True
					except:
						pass

	def record_spikes(self, rec_pars_dict, ids=None, label=None):
		"""
		Connect a spike detector to this population
		:param rec_pars_dict: common dictionary with recording device parameters
		:param ids: neuron ids to connect to (if None, all neurons are connected)
		"""
		if label is not None:
			nest.CopyModel('spike_detector', label)
			det = nest.Create(label, 1, rec_pars_dict)
			self.attached_device_names.append(label)
		else:			
			det = nest.Create('spike_detector', 1, rec_pars_dict)
		self.attached_devices.append(det)
		if ids is None:
			nest.Connect(self.gids, det, 'all_to_all')
		else:
			nest.Connect(list(ids), det, 'all_to_all')
		return det

	def record_analog(self, rec_pars_dict, ids=None, record=['V_m'], label=None):
		"""
		Connect a multimeter to neurons with gid = ids and record the specified variables
		:param rec_pars_dict: common dictionary with recording device parameters
		:param ids: neuron ids to connect (if None all neurons will be connected)
		:param record: recordable variable(s) - as list
		"""
		st = nest.GetStatus([self.gids[np.random.randint(self.size)]])[0]
		if st.has_key('recordables'):
			assert np.mean([x in list(nest.GetStatus([self.gids[np.random.randint(self.size)]], 'recordables')[0]) for x
			                 in record]) == 1., "Incorrect setting. Record should only contain recordable instances of " \
			                                    "the current neuron model (check 'recordables')"
		assert np.mean([x in nest.GetDefaults('multimeter').keys() for x in rec_pars_dict.iterkeys()]), "Provided " \
		                                    "dictionary is inconsistent with multimeter dictionary"
		if label is not None:
			nest.CopyModel('multimeter', label)
			mm = nest.Create(label, 1, rec_pars_dict)
			self.attached_device_names.append(label)
		else:
			mm = nest.Create('multimeter', 1, rec_pars_dict)
		self.attached_devices.append(mm)
		if ids is None:
			nest.Connect(mm, self.gids)
		else:
			nest.Connect(mm, list(ids))
		return mm

	@staticmethod
	def flush_records(device):
		"""
		Delete all recorded events from extractors and clear device memory
		"""
		nest.SetStatus(device, {'n_events': 0})
		if nest.GetStatus(device)[0]['to_file']:
			io.remove_files(nest.GetStatus(device)[0]['filenames'])

	def activity_set(self, initializer, t_start=None, t_stop=None):
		"""
		Extract recorded activity from attached devices, convert it to SpikeList or AnalogList
		objects and store them appropriately
		:param initializer: can be a string, or list of strings containing the relevant filenames where the
		raw data was recorded or be a gID for the recording device, if the data is still in memory
		:param t_start:
		:param t_stop:
		:return:
		"""
		# TODO: save option!
		# if object is a string, it must be a file name; if it is a list of strings, it must be a list of filenames
		if isinstance(initializer, basestring) or isinstance(initializer, list):
			data = io.extract_data_fromfile(initializer)
			if data is not None:
				if len(data.shape) != 2:
					data = np.reshape(data, (int(len(data) / 2), 2))
				if data.shape[1] == 2:
					spk_times = data[:, 1]
					neuron_ids = data[:, 0]
					tmp = [(neuron_ids[n], spk_times[n]) for n in range(len(spk_times))]
					self.spiking_activity = signals.SpikeList(tmp, np.unique(neuron_ids).tolist())
					self.spiking_activity.complete(self.gids)
				else:
					neuron_ids = data[:, 0]
					times = data[:, 1]
					if t_start is not None and t_stop is not None:
						idx1 = np.where(times >= t_start)[0]
						idx2 = np.where(times <= t_stop)[0]
						idxx = np.intersect1d(idx1, idx2)
						times = times[idxx]
						neuron_ids = neuron_ids[idxx]
						data = data[idxx, :]
					for nn in range(data.shape[1]):
						if nn > 1:
							sigs = data[:, nn]
							tmp = [(neuron_ids[n], sigs[n]) for n in range(len(neuron_ids))]
							self.analog_activity.append(signals.AnalogSignalList(tmp, np.unique(neuron_ids).tolist(),
							                                             times=times, t_start=t_start, t_stop=t_stop))

		elif isinstance(initializer, tuple) or isinstance(initializer, int):
			if isinstance(initializer, int):
				status = nest.GetStatus([initializer])[0]['events']
			else:
				status = nest.GetStatus(initializer)[0]['events']
			if len(status) == 2:
				spk_times = status['times']
				neuron_ids = status['senders']
				tmp = [(neuron_ids[n], spk_times[n]) for n in range(len(spk_times))]
				if t_start is None and t_stop is None:
					self.spiking_activity = signals.SpikeList(tmp, np.unique(neuron_ids).tolist())
					self.spiking_activity.complete(self.gids)
				else:
					self.spiking_activity = signals.SpikeList(tmp, np.unique(neuron_ids).tolist(), t_start=t_start,
					                                  t_stop=t_stop)
					self.spiking_activity.complete(self.gids)
			elif len(status) > 2:
				times = status['times']
				neuron_ids = status['senders']
				idxs = np.argsort(times)
				times = times[idxs]
				neuron_ids = neuron_ids[idxs]
				if t_start is not None and t_stop is not None:
					idx1 = np.where(times >= t_start)[0]
					idx2 = np.where(times <= t_stop)[0]
					idxx = np.intersect1d(idx1, idx2)
					times = times[idxx]
					neuron_ids = neuron_ids[idxx]
				rem_keys = ['times', 'senders']
				new_dict = {k: v[idxs] for k, v in status.iteritems() if k not in rem_keys}
				self.analog_activity = []
				for k, v in new_dict.iteritems():
					tmp = [(neuron_ids[n], v[n]) for n in range(len(neuron_ids))]
					self.analog_activity.append(signals.AnalogSignalList(tmp, np.unique(neuron_ids).tolist(),
					                                                    times=times,
					                                             t_start=t_start, t_stop=t_stop))
					self.analog_activity_names.append(k)
		else:
			print("Incorrect initializer...")

	def connect_decoders(self, decoding_pars):
		"""
		Create and connect a DecodingLayer to this population
		"""
		if isinstance(decoding_pars, dict):
			decoding_pars = parameters.ParameterSet(decoding_pars)
		assert isinstance(decoding_pars, parameters.ParameterSet), \
			"must be initialized with ParameterSet or dictionary"

		self.decoding_layer = analysis.DecodingLayer(decoding_pars, self)


########################################################################################################################
class Network(object):
	"""
	Network class is one of the central classes, containing lists of elements specifying
	the properties of each subpopulation (including the population objects)...

	Input:
		- net_pars_set -> ParameterSet object containing all the network parameters

	Population objects are iteratively created when the Network object is initialized
	"""
	@staticmethod
	def verify_consistent_dimensions(w, src_gids, tget_gids):
		"""
		Verify whether the connection matrix complies with the (source and target) population sizes
		"""

		src_dim, tget_dim = w.shape
		if len(src_gids) == src_dim and len(tget_gids) == tget_dim:
			return True
		else:
			return False

	def __init__(self, net_pars_set):
		"""
		Initialize network object by creating all sub-populations
		:param net_pars_set: ParameterSet object containing all relevant parameters
		"""

		def create_populations(net_pars_set):

			populations = [[] for _ in range(net_pars_set.n_populations)]
			# specify the keys not to be passed to the population objects
			not_allowed_keys = ['_url', 'label', 'n_populations', 'parameters', 'names', 'description']

			print ("\nCreating populations:")
			# iterate through the population list
			for n in range(net_pars_set.n_populations):

				if isinstance(net_pars_set.pop_names[n], list):
					# Create a composite population (with distinct sub-populations)
					populations[n] = []
					for nn in range(len(net_pars_set.pop_names[n])):
						# create sub-population dictionary
						subpop_dict = {k: v[n][nn] for k, v in net_pars_set.iteritems() if k not in not_allowed_keys
						               and isinstance(v[n], list)}
						# if input is not a list, it means that the properties apply to the parent population,
						# so to all sub-populations:
						subpop_dict.update({k: v[n] for k, v in net_pars_set.iteritems() if k not in not_allowed_keys
							and (not isinstance(v[n], list))})
						neuron_dict = subpop_dict['neuron_pars']
						nest.CopyModel(subpop_dict['neuron_pars']['model'], net_pars_set.pop_names[n][nn])
						nest.SetDefaults(net_pars_set.pop_names[n][nn], parameters.extract_nestvalid_dict(neuron_dict,
																							   param_type='neuron'))
						if net_pars_set.topology[n][nn]:
							tp_dict = subpop_dict['topology_dict']
							tp_dict.update({'elements': net_pars_set.pop_names[n][nn]})
							layer = tp.CreateLayer(tp_dict)
							gids = nest.GetLeaves(layer)[0]
							subpop_dict.update({'topology_dict': tp_dict, 'layer_gid': layer,
							                    'is_subpop': True, 'gids': gids})
							populations[n].append(Population(parameters.ParameterSet(subpop_dict)))
						else:
							gids = nest.Create(net_pars_set.pop_names[n][nn], n=int(net_pars_set.n_neurons[n][nn]))
							subpop_dict.update({'is_subpop': False, 'gids': gids})
							populations[n].append(Population(parameters.ParameterSet(subpop_dict)))

						print(("- Population {0!s}, with ids [{1!s}-{2!s}]".format(net_pars_set.pop_names[n][nn],
																							 min(gids), max(gids))))

				else:
					# create a normal population
					pop_dict = {k: v[n] for k, v in net_pars_set.iteritems() if k not in not_allowed_keys}
					neuron_dict = net_pars_set.neuron_pars[n]

					# create neuron model named after the population
					nest.CopyModel(net_pars_set.neuron_pars[n]['model'], net_pars_set.pop_names[n])
					# set default parameters
					nest.SetDefaults(net_pars_set.pop_names[n], parameters.extract_nestvalid_dict(neuron_dict,
					                                                                    param_type='neuron'))

					if net_pars_set.topology[n]:
						tp_dict = pop_dict['topology_dict']
						tp_dict.update({'elements': net_pars_set.pop_names[n]})
						layer = tp.CreateLayer(tp_dict)
						gids = nest.GetLeaves(layer)[0]
						pop_dict.update({'topology_dict': tp_dict, 'layer_gid': layer,
						                 'is_subpop': False, 'gids': gids})
						populations[n] = Population(parameters.ParameterSet(pop_dict))
					else:
						# create population
						gids = nest.Create(net_pars_set.pop_names[n], n=int(net_pars_set.n_neurons[n]))
						# set up population objects
						pop_dict.update({'gids': gids, 'is_subpop': False})
						populations[n] = Population(parameters.ParameterSet(pop_dict))
					print("- Population {0!s}, with ids [{1!s}-{2!s}]".format(net_pars_set.pop_names[n],
																			  min(gids), max(gids)))

			return populations

		verify_pars_consistency(net_pars_set, ['_url', 'label', 'n_populations', 'parameters', 'names',
											   'analog_device_pars', 'description'], net_pars_set.n_populations)

		self.populations = create_populations(net_pars_set)
		self.n_populations = net_pars_set.n_populations
		self.n_neurons = net_pars_set.n_neurons
		self.population_names = net_pars_set.pop_names
		self.record_spikes = net_pars_set.record_spikes
		self.record_analogs = net_pars_set.record_analogs
		self.spike_device_pars = net_pars_set.spike_device_pars
		self.analog_device_pars = net_pars_set.analog_device_pars
		self.device_names = [[] for _ in range(self.n_populations)]
		self.device_gids = [[] for _ in range(self.n_populations)]
		self.n_devices = [[] for _ in range(self.n_populations)]
		self.device_type = [[] for _ in range(self.n_populations)]
		self.spiking_activity = [[] for _ in range(self.n_populations)]
		self.analog_activity = [[] for _ in range(self.n_populations)]
		self.connection_types = []
		self.connection_names = []
		self.synaptic_weights = {}
		self.synaptic_delays = {}
		#self.state_extractors = []
		#self.readouts = []
		self.merged_populations = []

	def merge_subpopulations(self, sub_populations, name='', merge_activity=False, store=True):
		"""
		Combine sub-populations into a main Population object
		:param sub_populations: [list] - of Population objects to merge
		:param name: [str] - name of new population
		:param merge_activity:
		:return: new Population object
		"""
		assert sub_populations, "No sub-populations to merge provided..."
		gids_list = [list(x.gids) for x in sub_populations]
		gids = list(itertools.chain.from_iterable(gids_list))

		subpop_names = [x.name for x in sub_populations]
		if signals.empty(name):
			name = ''.join(subpop_names)

		pop_dict = {'pop_names': name, 'n_neurons': len(gids), 'gids': gids, 'is_subpop': False}

		if all([x.topology for x in sub_populations]):
			positions = [list(x.topology_dict['positions']) for x in sub_populations]
			positions = list(itertools.chain.from_iterable(positions))

			elements = [x.topology_dict['elements'] for x in sub_populations]

			layer_ids = [x.layer_gid for x in sub_populations]

			tp_dict = {'elements': elements, 'positions': positions,
			           'edge_wrap': all(bool('edge_wrap' in x.topology_dict and x.topology_dict['edge_wrap'])
										for x in sub_populations)}

			pop_dict.update({'topology': True, 'layer_gid': layer_ids, 'topology_dict': tp_dict})
		else:
			pop_dict.update({'topology': False})

		new_population = Population(parameters.ParameterSet(pop_dict))

		if merge_activity:
			assert (nest.GetKernelStatus()['time']), "There is no activity to merge, run the simulation first and " \
			                                         "merge_population_activity at the network level"
			gids 				= []
			n_neurons 			= np.sum([x.size for x in sub_populations])
			spk_activity_list 	= [x.spiking_activity for x in sub_populations]
			analog_activity 	= [x.analog_activity for x in sub_populations]

			# TODO @Renato there's a weird case here when t_start = t_stop = 0 for each population, although there are
			# TODO some spikes, e.g., at 0.1. There will be an error when creating an AnalogSignal because we're
			# TODO rounding the times 2 lines below..
			if not any([sl.empty() for sl in spk_activity_list]):
				t_start = round(np.min([x.t_start for x in spk_activity_list]))
				t_stop 	= round(np.max([x.t_stop for x in spk_activity_list]))

				# TODO TEMPORARY FIX only continue when t_stop > t_start
				if t_stop > t_start:
					new_spike_list = signals.SpikeList([], [], t_start=t_start, t_stop=t_stop, dims=n_neurons)

					for n in spk_activity_list:
						if not isinstance(n, list):
							gids.append(n.id_list)
							for idd in n.id_list:
								new_spike_list.append(idd, n.spiketrains[idd])
						else:
							print("Merge specific spiking activity")   # TODO
					new_population.spiking_activity = new_spike_list

			if not signals.empty(analog_activity):
				# TODO - extend AnalogSignalList[0] with [1] ...
				for n in analog_activity:
					new_population.analog_activity.append(n)
		if store:
			self.merged_populations.append(new_population)
		else:
			return new_population

	def connect_devices(self):
		"""
		Connect recording devices to the populations, according to the parameter specifications

		NOTE: Should only be called once! Otherwise, connections are repeated..
		"""
		print ("\nConnecting Devices: ")
		for n in range(self.n_populations):

			if isinstance(self.record_spikes[n], list):
				# recorder connected to specific sub-populations
				self.n_devices[n] = 0  # number of devices in parent population
				for nn in range(len(self.populations[n])):
					if self.record_spikes[n][nn]:
						dev_dict = self.spike_device_pars[n][nn].copy()
						self.device_type[n].append(dev_dict['model'])
						dev_dict['label'] += self.population_names[n][nn] + '_' + dev_dict['model']
						self.device_names[n].append(dev_dict['label'])
						dev_gid = self.populations[n][nn].record_spikes(parameters.extract_nestvalid_dict(
							dev_dict, param_type='device'))
						self.device_gids[n].append(dev_gid)
						self.n_devices[n] += 1
						print("- Connecting %s to %s, with label %s and id %s" % (
							dev_dict['model'], self.population_names[n][nn],
							dev_dict['label'], str(dev_gid)))
					if self.record_analogs[n][nn]:
						dev_dict = self.analog_device_pars[n][nn].copy()
						self.device_type[n].append(dev_dict['model'])
						dev_dict['label'] += self.population_names[n][nn] + '_' + dev_dict['model']
						self.device_names[n].append(dev_dict['label'])
						if dev_dict['record_n'] != self.populations[n][nn].size:
							tmp = np.random.permutation(self.n_neurons[n][nn])[:dev_dict['record_n']]
							ids = []
							for i in tmp:
								ids.append(self.populations[n][nn].gids[i])
						else:
							ids = None
						dev_gid = self.populations[n][nn].record_analog(parameters.extract_nestvalid_dict(
							dev_dict, param_type='device'), ids=ids, record=dev_dict['record_from'])
						self.device_gids[n].append(dev_gid)
						self.n_devices[n] += 1
						print("- Connecting %s to %s %s, with label %s and id %s" % (
							dev_dict['model'], self.population_names[n][nn], str(ids),
							dev_dict['label'], str(dev_gid)))

			elif isinstance(self.populations[n], list):
				# recorder connected to large population, including all its subpopulations
				new_pop = self.merge_subpopulations(self.populations[n])
				self.n_devices[n] = 0
				if self.record_spikes[n]:
					dev_dict = self.spike_device_pars[n].copy()
					self.device_type[n].append(dev_dict['model'])
					dev_dict['label'] += new_pop.name + '_' + dev_dict['model']
					self.device_names[n].append(dev_dict['label'])
					dev_gid = new_pop.record_spikes(parameters.extract_nestvalid_dict(dev_dict, param_type='device'))
					self.device_gids[n].append(dev_gid)
					self.n_devices[n] += 1
					print("- Connecting %s to %s, with label %s and id %s" % (dev_dict['model'], new_pop.name,
					                                                        dev_dict['label'], str(dev_gid)))
				if self.record_analogs[n]:
					dev_dict = self.analog_device_pars[n].copy()
					self.device_type[n].append(dev_dict['model'])
					dev_dict['label'] += new_pop.name + '_' + dev_dict['model']
					self.device_names[n].append(dev_dict['label'])
					if dev_dict['record_n'] != self.populations[n].size:
						tmp = np.random.permutation(self.n_neurons[n])[:dev_dict['record_n']]
						ids = []
						for i in tmp:
							ids.append(self.populations[n].gids[i])
					else:
						ids = None
					dev_gid = new_pop.record_analog(parameters.extract_nestvalid_dict(dev_dict, param_type='device'),
					                                ids=ids,
													record=dev_dict['record_from'])
					self.device_gids[n].append(dev_gid)

					self.n_devices[n] += 1
					print("- Connecting %s to %s [%s], with label %s and id %s" %
						  (dev_dict['model'], new_pop.name, str(ids), dev_dict['label'], str(dev_gid)))
			else:
				# there are no sub-populations
				self.n_devices[n] = 0
				if self.record_spikes[n]:
					dev_dict = self.spike_device_pars[n].copy()
					self.device_type[n].append(dev_dict['model'])
					dev_dict['label'] += self.population_names[n] + '_' + dev_dict['model']
					self.device_names[n].append(dev_dict['label'])
					dev_gid = self.populations[n].record_spikes(parameters.extract_nestvalid_dict(dev_dict,
					                                                                      param_type='device'))
					self.device_gids[n].append(dev_gid)
					self.n_devices[n] += 1
					print("- Connecting %s to %s, with label %s and id %s" %
						  (dev_dict['model'], self.population_names[n], dev_dict['label'], str(dev_gid)))
				if self.record_analogs[n]:
					if isinstance(self.record_analogs[n], bool):
						dev_dict = self.analog_device_pars[n].copy()
						self.device_type[n].append(dev_dict['model'])
						dev_dict['label'] += self.population_names[n] + '_' + dev_dict['model']
						self.device_names[n].append(dev_dict['label'])
						if dev_dict['record_n'] != self.populations[n].size:
							tmp = np.random.permutation(self.n_neurons[n])[:dev_dict['record_n']]
							ids = []
							for i in tmp:
								ids.append(self.populations[n].gids[i])
						else:
							ids = None
						dev_gid = self.populations[n].record_analog(parameters.extract_nestvalid_dict(dev_dict,
						                                                                    param_type='device'),
																	ids=ids, record=dev_dict['record_from'])
						self.device_gids[n].append(dev_gid)

						self.n_devices[n] += 1
						if (ids is not None) and (len(ids) == 1):
							print("- Connecting %s to %s [%s], with label %s and id %s" % (dev_dict['model'],
							                                                               self.population_names[n],
							                                                               str(ids),
							                                                               dev_dict['label'],
							                                                               str(dev_gid)))
						elif ids is not None:
							print("- Connecting %s to %s [%s-%s], with label %s and id %s" % (dev_dict['model'],
							                                                                  self.population_names[n],
							                                                                  str(min(ids)),
							                                                                  str(max(ids)),
							                                                                  dev_dict['label'],
							                                                                  str(dev_gid)))
						else:
							print("- Connecting %s to %s [%s], with label %s and id %s" % (dev_dict['model'],
							                                                               self.population_names[n],
							                                                               str('all'),
							                                                               dev_dict['label'],
							                                                               str(dev_gid)))
					else:
						for nnn in range(int(self.record_analogs[n])):
							dev_dict = self.analog_device_pars[n][nnn].copy()
							self.device_type[n].append(dev_dict['model'])
							dev_dict['label'] += self.population_names[n] + '_' + dev_dict['model']
							self.device_names[n].append(dev_dict['label'])
							if dev_dict['record_n'] != self.populations[n].size:
								tmp = np.random.permutation(self.n_neurons[n])[:dev_dict['record_n']]
								ids = []
								for i in tmp:
									ids.append(self.populations[n].gids[i])
							else:
								ids = None
							dev_gid = self.populations[n].record_analog(parameters.extract_nestvalid_dict(dev_dict,
							                                                                    param_type='device'),
																		ids=ids, record=dev_dict['record_from'])
							self.device_gids[n].append(dev_gid)

							self.n_devices[n] += 1
							if len(ids) == 1:
								print("- Connecting %s to %s [%s], with label %s and id %s" % (dev_dict['model'],
								                                                               self.population_names[n],
								                                                               str(ids),
								                                                               dev_dict['label'],
								                                                               str(dev_gid)))
							else:
								print("- Connecting %s to %s [%s-%s], with label %s and id %s" % (dev_dict['model'],
								                                                               self.population_names[n],
								                                                               str(min(ids)),
								                                                               str(max(ids)),
								                                                               dev_dict['label'],
								                                                               str(dev_gid)))

	def connect_populations(self, connect_pars_set, progress=False):
		"""
		Connect the sub-populations according to the specifications
		:param connect_pars_set: ParameterSet for establishing connections
		:return:
		"""
		print ("\nConnecting populations: ")
		for n in range(connect_pars_set.n_synapse_types):
			print("    - %s [%s]" % (connect_pars_set.synapse_types[n], connect_pars_set.models[n]))

			# index of source and target populations in the population lists
			if connect_pars_set.synapse_types[n][1] in self.population_names:
				src_pop_idx = self.population_names.index(connect_pars_set.synapse_types[n][1])
			else:
				src_pop_idx = []
				for ii, nn in enumerate(self.population_names):
					if isinstance(nn, list):
						src_pop_idx.append(ii)
						src_pop_idx.append(nn.index(connect_pars_set.synapse_types[n][1]))
			if connect_pars_set.synapse_types[n][0] in self.population_names:
				tget_pop_idx = self.population_names.index(connect_pars_set.synapse_types[n][0])
			else:
				tget_pop_idx = []
				for ii, nn in enumerate(self.population_names):
					if isinstance(nn, list):
						tget_pop_idx.append(ii)
						tget_pop_idx.append(nn.index(connect_pars_set.synapse_types[n][0]))

			# corresponding global ids of each population to connect, or layer ids if topological connections are
			# present
			if not isinstance(src_pop_idx, int):
				if connect_pars_set.topology_dependent[n]:
					assert self.populations[src_pop_idx[0]][src_pop_idx[1]].topology, "Source layer doesn't have " \
					                                                                  "topology"
					src_gids = self.populations[src_pop_idx[0]][src_pop_idx[1]].layer_gid
				else:
					src_gids = self.populations[src_pop_idx[0]][src_pop_idx[1]].gids
			else:
				if connect_pars_set.topology_dependent[n]:
					assert self.populations[src_pop_idx].topology, "Source layer doesn't have topology"
					src_gids = self.populations[src_pop_idx].layer_gid
				else:
					src_gids = self.populations[src_pop_idx].gids
			if not isinstance(tget_pop_idx, int):
				if connect_pars_set.topology_dependent[n]:
					assert self.populations[tget_pop_idx[0]][tget_pop_idx[1]].topology, "Target layer doesn't have " \
					                                                                    "topology"
					tget_gids = self.populations[tget_pop_idx[0]][tget_pop_idx[1]].layer_gid
				else:
					tget_gids = self.populations[tget_pop_idx[0]][tget_pop_idx[1]].gids
			else:
				if connect_pars_set.topology_dependent[n]:
					assert self.populations[tget_pop_idx].topology, "Target layer doesn't have topology"
					tget_gids = self.populations[tget_pop_idx].layer_gid
				else:
					tget_gids = self.populations[tget_pop_idx].gids

			# copy and modify synapse model
			if hasattr(connect_pars_set, "synapse_names"):
				synapse_name = connect_pars_set.synapse_names[n]
			else:
				synapse_name = connect_pars_set.synapse_types[n][1] + '_' + connect_pars_set.synapse_types[n][0]

			nest.CopyModel(connect_pars_set.models[n], synapse_name)
			nest.SetDefaults(synapse_name, connect_pars_set.model_pars[n])
			self.connection_names.append(synapse_name)
			self.connection_types.append(connect_pars_set.synapse_types[n])

			#**** set up connections ****
			if synapse_name.find('copy') > 0:  # re-connect the same neurons...
				start = time.time()
				print("    - Connecting {0} (*)".format(synapse_name))

				device_models = ['spike_detector', 'spike_generator', 'multimeter']
				target_synapse_name = synapse_name[:synapse_name.find('copy')-1]
				conns = nest.GetConnections(synapse_model=target_synapse_name)

				iterate_steps = 100
				its = np.arange(0, len(conns)+1, iterate_steps).astype(int)

				for nnn, it in enumerate(its):
					if nnn < len(its)-1:
						con = conns[it:its[nnn+1]]
						st = nest.GetStatus(con)
						source_gids = [x['source'] for x in st if
						               nest.GetStatus([x['target']])[0]['model'] not in device_models]
						target_gids = [x['target'] for x in st if
						               nest.GetStatus([x['target']])[0]['model'] not in device_models]
						weights = [x['weight'] for x in st if nest.GetStatus([x['target']])[0]['model'] not in device_models]
						delays = [x['delay'] for x in st if nest.GetStatus([x['target']])[0]['model'] not in device_models]
						models = [x['synapse_model'] for x in st if
						          nest.GetStatus([x['target']])[0]['model'] not in device_models]
						receptors = [x['receptor'] for x in st if
						             nest.GetStatus([x['target']])[0]['model'] not in device_models]
						syn_dict = parameters.copy_dict(connect_pars_set.syn_specs[n], {'model': synapse_name, 'weight':
							connect_pars_set.weight_dist[n], 'delay': connect_pars_set.delay_dist[n]})
						conn_dict = connect_pars_set.conn_specs[n]

						syn_dicts = [{'synapsemodel': list(np.repeat(synapse_name, len(source_gids)))[iddx],
						              'source': source_gids[iddx],
						              'target': target_gids[iddx],
						              'weight': syn_dict['weight'],		              # TODO distributions??
						              'delay': syn_dict['delay'],
						              'receptor_type': syn_dict['receptor_type']} for iddx in range(len(target_gids))]
						nest.DataConnect(syn_dicts)
					if progress:
						visualization.progress_bar(float(nnn) / float(len(its)))
				print("\tElapsed time: {0} s".format(str(time.time()-start)))
			else:
				# 1) if pre-computed weights matrices are given
				if (connect_pars_set.pre_computedW[n] is not None) and (not connect_pars_set.topology_dependent[n]):

					if isinstance(connect_pars_set.pre_computedW[n], str):
						w = np.load(connect_pars_set.pre_computedW[n])
					else:
						w = connect_pars_set.pre_computedW[n]

					if self.verify_consistent_dimensions(w, src_gids, tget_gids):
						for preSyn_matidx, preSyn_gid in enumerate(src_gids):
							postSyn_matidx = w[preSyn_matidx, :].nonzero()[0]
							postSyn_gid = list(postSyn_matidx + min(tget_gids))
							weights = [w[preSyn_matidx, x] for x in postSyn_matidx]
							if len(connect_pars_set.delay_dist[n]) > 1 and not isinstance(
									connect_pars_set.delay_dist[n], dict):
								delays = connect_pars_set.delay_dist[n]
							elif isinstance(connect_pars_set.delay_dist[n], dict):
								delays = [parameters.copy_dict(connect_pars_set.delay_dist[n]) for _ in range(len(
									weights))]
							else:
								delays = np.repeat(connect_pars_set.delay_dist[n], len(weights))
							for idx, tget in enumerate(postSyn_gid):
								syn_dict = parameters.copy_dict(connect_pars_set.syn_specs[n], {'model': synapse_name,
								                                                         'weight':
									weights[idx], 'delay': delays[idx]})
								nest.Connect([preSyn_gid], [tget], syn_spec=syn_dict)
							if progress:
								visualization.progress_bar(float(preSyn_matidx)/float(len(src_gids)))
					else:
						raise Exception("Dimensions of W are inconsistent with population sizes")

				# 2) if no pre-computed weights, and no topology in pre/post-synaptic populations, use dictionaries
				elif (connect_pars_set.pre_computedW[n] is None) and (not connect_pars_set.topology_dependent[n]):

					syn_dict = parameters.copy_dict(connect_pars_set.syn_specs[n], {'model': synapse_name, 'weight':
						connect_pars_set.weight_dist[n], 'delay': connect_pars_set.delay_dist[n]})
					conn_dict = connect_pars_set.conn_specs[n]

					nest.Connect(src_gids, tget_gids, conn_spec=conn_dict, syn_spec=syn_dict)

				# 3) if no precomputed weights, but topology in populations and topological connections
				elif connect_pars_set.topology_dependent[n]:
					assert (nest.is_sequence_of_gids(src_gids) and len(src_gids) == 1), "Source ids are not topology layer"
					assert (nest.is_sequence_of_gids(tget_gids) and len(tget_gids) == 1), "Target ids are not topology " \
				                                                                      "layer"
					tp_dict = parameters.copy_dict(connect_pars_set.conn_specs[n], {'synapse_model': synapse_name})
					tp.ConnectLayers(src_gids, tget_gids, tp_dict)

	def flush_records(self, decoders=False):
		"""
		Delete all data from all devices connected to the network
		:return:
		"""
		if not signals.empty(self.device_names):
			print("\nClearing device data: ")
		devices = list(itertools.chain.from_iterable(self.device_names))

		for idx, n in enumerate(list(itertools.chain.from_iterable(self.device_gids))):
			print(" - {0} {1}".format(devices[idx], str(n)))
			nest.SetStatus(n, {'n_events': 0})
			if nest.GetStatus(n)[0]['to_file']:
				io.remove_files(nest.GetStatus(n)[0]['filenames'])

		if decoders:
			for ctr, n_pop in enumerate(list(itertools.chain(*[self.merged_populations, self.populations]))):
				if n_pop.decoding_layer is not None:
					n_pop.decoding_layer.flush_records()

	def copy(self):
		"""
		Returns a copy of the entire network object. Doesn't create new NEST objects.
		:return:
		"""
		return copy.deepcopy(self)

	# TODO @comment more, maybe 1 example why this would be needed? also, what stays the same?
	def clone(self, original_parameter_set, devices=True, decoders=True):
		"""
		Creates a new network object
		:param original_parameter_set:
		:param devices:
		:param decoders:
		:return:
		"""
		param_set = copy.deepcopy(original_parameter_set)

		def create_clone(net):
			"""
			Create Network object
			:return:
			"""
			neuron_pars = [0 for n in range(len(list(signals.iterate_obj_list(net.populations))))]
			topology = [0 for n in range(len(list(signals.iterate_obj_list(net.populations))))]
			topology_dict = [0 for n in range(len(list(signals.iterate_obj_list(net.populations))))]
			pop_names = [0 for n in range(len(list(signals.iterate_obj_list(net.populations))))]
			spike_device_pars = [0 for n in range(len(list(signals.iterate_obj_list(net.populations))))]
			analog_dev_pars = [0 for n in range(len(list(signals.iterate_obj_list(net.populations))))]
			status_elements = ['archiver_length', 'element_type', 'frozen', 'global_id',
			                   'has_connections', 'local', 'recordables', 't_spike',
			                   'thread', 'thread_local_id', 'vp', 'n_synapses',
			                   'local_id', 'model', 'parent']
			for idx_pop, pop_obj in enumerate(list(signals.iterate_obj_list(net.populations))):
				# get generic neuron_pars (update individually later):
				neuron = param_set.extract_nestvalid_dict(nest.GetStatus([pop_obj.gids[0]])[0], param_type='neuron')
				d_tmp = {k: v for k, v in neuron.items() if k not in status_elements}
				neuron_pars[idx_pop] = param_set.copy_dict(d_tmp, {'model': nest.GetStatus([pop_obj.gids[0]])[0][
					'model']})
				if isinstance(pop_obj.topology, dict):
					topology[idx_pop] = True
					topology_dict[idx_pop] = pop_obj.topology
				else:
					topology[idx_pop] = False
					topology_dict[idx_pop] = pop_obj.topology
				pop_names[idx_pop] = net.population_names[idx_pop] + '_clone'
				if net.record_spikes[idx_pop]:
					spike_device_pars[idx_pop] = param_set.copy_dict(net.spike_device_pars[idx_pop],
						{'label': net.spike_device_pars[idx_pop]['label']+'_clone'})
				else:
					spike_device_pars[idx_pop] = None
				if net.record_analogs[idx_pop]:
					analog_dev_pars[idx_pop] = param_set.copy_dict(net.analog_device_pars[idx_pop],
						{'label': net.analog_device_pars[idx_pop]['label']+'_clone'})
				else:
					analog_dev_pars[idx_pop] = None

			network_parameters = {'n_populations': net.n_populations,
			                      'pop_names': pop_names,
			                      'n_neurons': net.n_neurons,
			                      'neuron_pars': neuron_pars,
			                      'topology': topology,
			                      'topology_dict': topology_dict,
			                      'record_spikes': net.record_spikes,
			                      'spike_device_pars': spike_device_pars,
			                      'record_analogs': net.record_analogs,
			                      'analog_device_pars': analog_dev_pars}
			clone_net = Network(param_set.ParameterSet(network_parameters, label='clone'))
			#clone_net.connect_devices()

			for pop_idx, pop_obj in enumerate(clone_net.populations):
				for n_neuron in range(net.n_neurons[pop_idx]):
					src_gid = net.populations[pop_idx].gids[n_neuron]
					tget_gid = pop_obj.gids[n_neuron]
					status_dict = nest.GetStatus([src_gid])[0]
					st = {k: v for k, v in status_dict.items() if k not in status_elements}
					nest.SetStatus([tget_gid], st)
			return clone_net

		def connect_clone(network, clone, progress=True):
			"""
			Connect the populations in the clone network
			(requires iteration through all the connections in the mother network...)
			:param network:
			:param clone:
			:return:
			"""
			devices = ['stimulator', 'structure']
			base_idx = min(list(itertools.chain(*[n.gids for n in clone.populations])))-1
			print("\n Replicating connectivity in clone network (*)")
			for syn_idx, synapse in enumerate(network.connection_names):
				start = time.time()
				copy_synapse_name = synapse + '_clone'
				nest.CopyModel(synapse, copy_synapse_name)
				print("\t- {0}".format(str(network.connection_types[syn_idx])))
				conns = nest.GetConnections(synapse_model=synapse)
				# ##
				iterate_steps = 100
				its = np.arange(0, len(conns)+iterate_steps, iterate_steps).astype(int)
				its[-1] = len(conns)
				# ##
				clone.connection_names.append(copy_synapse_name)
				conn_types = network.connection_types[syn_idx]
				connection_type = (conn_types[0]+'_clone', conn_types[1]+'_clone')
				clone.connection_types.append(connection_type)

				# src_idx = clone.population_names.index(connection_type[1])
				# tget_idx = clone.population_names.index(connection_type[0])
				# src_gids = clone.populations[src_idx].gids
				# tget_gids = clone.populations[tget_idx].gids
				# base_idx = min(list(itertools.chain(*[src_gids, tget_gids]))) - 1

				for nnn, it in enumerate(its):
					if nnn < len(its) - 1:
						con = conns[it:its[nnn+1]]
						st = nest.GetStatus(con)
						source_gids = [x['source']+base_idx for x in st if nest.GetDefaults(nest.GetStatus([x[
							                                      'target']])[0]['model'])['element_type'] not in
						               devices]
						target_gids = [x['target']+base_idx for x in st if nest.GetDefaults(nest.GetStatus([x[
							                                        'target']])[0]['model'])['element_type'] not in
						               devices]
						weights = [x['weight'] for x in st if nest.GetDefaults(nest.GetStatus([x['target']])[0][
							'model'])['element_type'] not in devices]
						delays = [x['delay'] for x in st if nest.GetDefaults(nest.GetStatus([x['target']])[0][
							'model'])['element_type'] not in devices]

						receptors = [x['receptor'] for x in st if nest.GetDefaults(nest.GetStatus([x['target']])[0][
							'model']).has_key('element_type') and nest.GetDefaults(nest.GetStatus([x['target']])[0][
							                                            'model'])['element_type'] not in devices]
						# modify target receptors in cases where they are used:
						# for iddx, n_rec in enumerate(receptors):
						# 	# check if neuron accepts different receptors
						# 	if nest.GetStatus([target_gids[iddx]])[0].has_key('rec_type'):
						# 		rec_types = list(nest.GetStatus([target_gids[iddx]])[0]['rec_type'])
						# 		receptors[iddx] = int(nest.GetStatus([target_gids[iddx]])[0]['rec_type'][n_rec])
						syn_dicts = [{'synapse_model': list(np.repeat(copy_synapse_name, len(source_gids)))[iddx],
						              'source': source_gids[iddx],
						              'target': target_gids[iddx],
						              'weight': weights[iddx],
						              'delay': delays[iddx],
						              'receptor_type': receptors[iddx]} for iddx in range(len(target_gids))]
						nest.DataConnect(syn_dicts)
						if progress:
							visualization.progress_bar(float(nnn)/float(len(its)))
				print("\tElapsed time: {0} s".format(str(time.time() - start)))

		def connect_decoders(network, parameters):
			"""
			:param parameters:
			:return:
			"""
			target_populations = parameters.decoding_pars.state_extractor.source_population
			copy_targets = [n+'_clone' for n in target_populations]
			parameters.decoding_pars.state_extractor.source_population = copy_targets
			network.connect_decoders(parameters.decoding_pars)

		clone_network = create_clone(self)
		connect_clone(self, clone_network)
		if devices:
			clone_network.connect_devices()
		if decoders:
			connect_decoders(clone_network, param_set)

		return clone_network

	def mirror(self, copy_net=None, from_main=True, to_main=False):
		"""
		Returns a network object equal to self and either connected
		to main network or receiving connections from main network
		:return:
		"""
		if copy_net is None:
			copy_net = self.copy()
			cn = None
		else:
			assert isinstance(copy_net, Network), "copy_net must be Network object"
			cn = 1.

		if to_main:
			print("Connecting CopyNetwork: ")
			device_models = ['spike_detector', 'spike_generator', 'multimeter']
			for pop_idx, pop_obj in enumerate(copy_net.populations):
				original_gid_range = [min(self.populations[pop_idx].gids),
				                      max(self.populations[pop_idx].gids)]
				copy_gid_range = [min(pop_obj.gids), max(pop_obj.gids)]

				start = time.time()
				print("    - {0}, {1}".format(copy_net.population_names[pop_idx], self.population_names[pop_idx]))

				for n_neuron in range(self.n_neurons[pop_idx]):
					src_gid = self.populations[pop_idx].gids[n_neuron]
					tget_gid = pop_obj.gids[n_neuron]

					conns = nest.GetConnections(source=[src_gid])

					# for memory conservation, iterate:
					iterate_steps = 100
					its = np.arange(0, len(conns), iterate_steps).astype(int)

					for nnn, it in enumerate(its):
						if nnn < len(its)-1:
							conn = conns[it:its[nnn+1]]
							st = nest.GetStatus(conn)

						target_gids = [x['target'] for x in st if
						               nest.GetStatus([x['target']])[0]['model'] not in device_models]
						weights = [x['weight'] for x in st if
						           nest.GetStatus([x['target']])[0]['model'] not in device_models]
						delays = [x['delay'] for x in st if nest.GetStatus([x['target']])[0]['model'] not in device_models]
						models = [x['synapse_model'] for x in st if
						          nest.GetStatus([x['target']])[0]['model'] not in device_models]
						receptors = [x['receptor'] for x in st if
						             nest.GetStatus([x['target']])[0]['model'] not in device_models]

						tgets = [x + (copy_gid_range[0] - original_gid_range[0]) for x in target_gids]
						syn_dicts = [{'synapsemodel': models[iddx], 'source': tget_gid,
						              'target': tgets[iddx], 'weight': weights[iddx],
						              'delay': delays[iddx], 'receptor_type': receptors[iddx]} for iddx in range(len(
							target_gids))]
						nest.DataConnect(syn_dicts)
				print("Elapsed Time: {0}".format(str(time.time()-start)))

		elif from_main:
			print("\nConnecting CopyNetwork: ")
			device_models = ['spike_detector', 'spike_generator', 'multimeter']
			for pop_idx, pop_obj in enumerate(copy_net.populations):
				original_gid_range = [min(self.populations[pop_idx].gids),
				                      max(self.populations[pop_idx].gids)]
				copy_gid_range = [min(pop_obj.gids), max(pop_obj.gids)]

				start = time.time()
				print("\t    - {0}, {1}".format(copy_net.population_names[pop_idx], self.population_names[pop_idx]))

				for n_neuron in range(self.n_neurons[pop_idx]):
					src_gid = self.populations[pop_idx].gids[n_neuron]
					tget_gid = pop_obj.gids[n_neuron]

					conns = nest.GetConnections(target=[src_gid])

					# for memory conservation, iterate:
					iterate_steps = 100
					its = np.arange(0, len(conns), iterate_steps).astype(int)

					for nnn, it in enumerate(its):
						if nnn < len(its)-1:
							conn = conns[it:its[nnn+1]]
							st = nest.GetStatus(conn)
							source_gids = [x['source'] for x in st if
							               nest.GetStatus([x['source']])[0]['model'] not in device_models]
							weights = [x['weight'] for x in st if
							           nest.GetStatus([x['target']])[0]['model'] not in device_models]
							delays = [x['delay'] for x in st if nest.GetStatus([x['target']])[0]['model'] not in device_models]
							models = [x['synapse_model'] for x in st if
							          nest.GetStatus([x['target']])[0]['model'] not in device_models]
							receptors = [x['receptor'] for x in st if
							             nest.GetStatus([x['target']])[0]['model'] not in device_models]

							sources = [x + (copy_gid_range[0] - original_gid_range[0]) for x in source_gids]
							syn_dicts = [{'synapsemodel': models[iddx], 'source': sources[iddx],
							              'target': tget_gid, 'weight': weights[iddx],
							              'delay': delays[iddx], 'receptor_type': receptors[iddx]} for iddx in range(len(
								source_gids))]
							nest.DataConnect(syn_dicts)
				print("Elapsed Time: {0}".format(str(time.time() - start)))
		if cn is None:
			return copy_net

	def simulate(self, t=10, reset_devices=False, reset_network=False):
		"""
		Simulate network
		:param t: total time
		:return:
		"""
		nest.Simulate(t)
		if reset_devices:
			self.flush_records()
		if reset_network:
			nest.ResetNetwork()

	def extract_population_activity(self, t_start=None, t_stop=None):
		"""
		Iterate through the populations in the network, verify which recording devices have been connected to them
		and extract this data. The activity is then converted in SpikeList or AnalogList objects and attached to
		the properties of the corresponding population.
		To merge the data from multiple populations see extract_network_activity()
		"""
		if not signals.empty(self.device_names):
			print("\nExtracting and storing recorded activity from devices:")

		for idx, n in enumerate(self.populations):
			if isinstance(n, list):
				for idxx, nn in enumerate(n):
					if nn.attached_devices:
						print("- Population {0}".format(nn.name))
						for nnn in nn.attached_devices:
							if nest.GetStatus(nnn)[0]['to_memory']:
								# initialize activity_set with device gid
								nn.activity_set(nnn, t_start=t_start, t_stop=t_stop)
							elif nest.GetStatus(nnn)[0]['to_file']:
								# initialize activity_set with file paths
								nn.activity_set(list(nest.GetStatus(nnn)[0]['filenames']), t_start=t_start, t_stop=t_stop)
			else:
				if n.attached_devices:
					print("- Population {0}".format(n.name))
					for nn in n.attached_devices:
						if nest.GetStatus(nn)[0]['to_memory']:
							# initialize activity_set with device gid
							n.activity_set(nn, t_start=t_start, t_stop=t_stop)
						elif nest.GetStatus(nn)[0]['to_file']:
							n.activity_set(list(nest.GetStatus(nn)[0]['filenames']), t_start=t_start, t_stop=t_stop)

	def extract_network_activity(self):
		"""
		Combine the activity lists attached to each population into a list of SpikeList or AnalogList objects
		corresponding to each sub-population
		NOTE: This function will copy the activity lists contained in the lower-level Population objects into the
		properties of the Network object (it may be redundant to maintain both activity sets, and it's only useful in
		certain situations)
		"""
		for n in range(self.n_populations):
			if isinstance(self.populations[n], list) and self.n_devices[n]:
				for nn in range(len(self.populations[n])):
					self.spiking_activity[n].append(self.populations[n][nn].spiking_activity)
					self.analog_activity[n].append(self.populations[n][nn].analog_activity)
			elif self.n_devices[n]:
				self.spiking_activity[n] = self.populations[n].spiking_activity
				self.analog_activity[n] = self.populations[n].analog_activity

	def merge_population_activity(self, start=0., stop=1000., save=False):
		"""
		Merge spike and analog data from the different populations
		:return:
		"""
		analog_activity = []
		spiking_activity = signals.SpikeList([], [], start, stop, np.sum(list(signals.iterate_obj_list(self.n_neurons))))
		gids = []
		for n in list(signals.iterate_obj_list(self.spiking_activity)):
			gids.append(n.id_list)
			for idd in n.id_list:
				spiking_activity.append(idd, n.spiketrains[idd])
		for n in list(signals.iterate_obj_list(self.analog_activity)):
			analog_activity.append(n)

		if save and not signals.empty(self.merged_populations):
			# verify if the merged population corresponds to the network's populations...
			self.merged_populations[0].spiking_activity = spiking_activity
			self.merged_populations[0].analog_activity = analog_activity

		return spiking_activity, analog_activity

	def extract_synaptic_weights(self, src_gids=None, tget_gids=None):
		"""
		Read and store the weights for all the connected populations, or just for
		the provided sources and targets
		:param src_gids: list of gids of source neurons
		:param tget_gids: list of gids of target neurons
		"""

		if src_gids and tget_gids:
			syn_name = str(nest.GetStatus(nest.GetConnections([src_gids[0]], [tget_gids[0]]))[0]['synapse_model'])
			self.synaptic_weights.update({syn_name: extract_weights_matrix(src_gids, tget_gids)})
		else:
			for nn in self.connection_types:
				src_idx = self.population_names.index(nn[1])
				tget_idx = self.population_names.index(nn[0])
				src_gids = self.populations[src_idx].gids
				tget_gids = self.populations[tget_idx].gids
				self.synaptic_weights.update({nn: extract_weights_matrix(src_gids, tget_gids)})

	def extract_synaptic_delays(self, src_gids=None, tget_gids=None):
		"""

		:param src_gids:
		:param tget_gids:
		:return:
		"""
		if src_gids and tget_gids:
			syn_name = str(nest.GetStatus(nest.GetConnections([src_gids[0]], [tget_gids[0]]))[0]['synapse_model'])
			self.synaptic_weights.update({syn_name: extract_weights_matrix(src_gids, tget_gids)})
		else:
			for nn in self.connection_types:
				src_idx = self.population_names.index(nn[1])
				tget_idx = self.population_names.index(nn[0])
				src_gids = self.populations[src_idx].gids
				tget_gids = self.populations[tget_idx].gids
				self.synaptic_delays.update({nn: extract_delays_matrix(src_gids, tget_gids)})

	def connect_decoders(self, decoding_pars):
		"""
		Connect a Decoding Layer to the current network object
		:return:
		"""
		if isinstance(decoding_pars, dict):
			decoding_pars = parameters.ParameterSet(decoding_pars)
		assert isinstance(decoding_pars, parameters.ParameterSet), "DecodingLayer must be initialized with " \
		                                                           "ParameterSet or dictionary"

		population_names = list(signals.iterate_obj_list(self.population_names))
		population_objs = list(signals.iterate_obj_list(self.populations))
		merged_population_names = [x.name for x in self.merged_populations]
		merged_population_objs = [x for x in self.merged_populations]
		decoder_params = {}

		# initialize state extractors:
		if hasattr(decoding_pars, "state_extractor"):
			pars_st = decoding_pars.state_extractor
			print("\nConnecting Decoders: ")

			# group state_extractors by source population
			sources = []
			source_populations = []
			for ext_idx, n_src in enumerate(pars_st.source_population):
				if isinstance(n_src, list):
					pop_label = ''.join(n_src)
					if pop_label not in merged_population_names:
						sub_population_idx = [population_names.index(x) for x in n_src]
						sub_populations = [population_objs[x] for x in sub_population_idx]
						source_population = self.merge_subpopulations(sub_populations=sub_populations, name=pop_label,
						                            merge_activity=True)
					else:
						source_population = merged_population_objs[merged_population_names.index(pop_label)]
				else:
					pop_label = n_src
					pop_index = population_names.index(n_src)
					source_population = population_objs[pop_index]
				sources.append(pop_label)
				source_populations.append(source_population)
			unique_sources = set(sources)

			extractor_indices = {}
			for n_ext in list(unique_sources):
				extractor_indices.update({n_ext: np.where(np.array(sources) == n_ext)[0]})

			# create decoder parameters dictionary for each source population
			keys = ['state_specs', 'state_variable', 'reset_states', 'average_states', 'standardize']
			for k, v in extractor_indices.items():
				decoder_params.update({k: {}})
				for k1 in keys:
					decoder_params[k].update({k1: [pars_st[k1][x] for x in v]})

			if hasattr(decoding_pars, "readout"):
				pars_rd = decoding_pars.readout
				for k, v in extractor_indices.items():
					decoder_params[k].update({'readout': [pars_rd[x] for x in v]})
		else:
			raise IOError("DecodingLayer requires the specification of state extractors")

		pops = [source_populations[sources.index(x)] for x in unique_sources]
		for (population_name, population) in zip(unique_sources, pops):
			population.connect_decoders(parameters.ParameterSet(decoder_params[population_name]))
