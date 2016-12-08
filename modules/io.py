__author__ = 'duarte'
"""
=================================================================================
Input/Output Module
=================================================================================
(adapted from NeuroTools.io)

Handle all the inputs and outputs

Classes:
--------------
FileHandler        - abstract class which should be overriden, managing how a file will load/write
                     its data
StandardTextFile   - object used to manipulate text representation of NeuroTools objects (spikes or
                     analog signals)
StandardPickleFile - object used to manipulate pickle representation of NeuroTools objects (spikes or
                     analog signals)
NestFile           - object used to manipulate raw NEST file that would not have been saved by pyNN
                     (without headers)
DataHandler        - object to establish the interface between NeuroTools.signals and NeuroTools.io

Functions:
---------------
extract_data_fromfile     - extract raw_data from a text file (typically written by a nest device)
is_not_empty_file         - simple function to verify if the file is empty (sub-optimal)

"""
import os
import cPickle
import numpy as np
from modules.parameters import *
from modules import check_dependency


def extract_data_fromfile(fname):
	"""
	Extract raw data from a nest.device recording stored in file
	:param fname: filename or list of filenames
	"""
	from parameters import isiterable
	if isiterable(fname):
		data = None
		for f in fname:
			print "Reading data from file {0}".format(f)
			if os.path.isfile(f) and os.path.getsize(f) > 0:
				with open(f, 'r') as fp:
					if fp.readline(4) == '....':
						info = fp.readlines()[1:]
					else:
						info = []
				if len(info):
					with open(f, 'w') as fp:
						fp.writelines(info)

				if data is None:
					# data = np.loadtxt(f)
					data = get_data(f)
				else:
					# data = np.concatenate((data, np.loadtxt(f)))
					data = np.concatenate((data, get_data(f)))
	else:
		with open(fname, 'r') as fp:
			if fp.readline(4) == '....':  # some bug in the recording...
				info = fp.readlines()[1:]
			else:
				info = []
		if len(info):
			with open(fname, 'w') as fp:
				fp.writelines(info)

		# data = np.loadtxt(fname)
		data = get_data(fname)
	return data


def get_data(filename, sepchar="\t", skipchar="#"):
	"""
	Load data from a text file and returns a list of data
	"""
	with open(filename, "r") as myfile:
		contents = myfile.readlines()
	data = []
	header = True
	idx    = 0
	while header:
		if contents[idx][0] != skipchar:
			header = False
			break
		idx += 1
	for i in xrange(idx, len(contents)):
		line = contents[i].strip().split(sepchar)
		if i == 0:
			line_len = len(line)
		if len(line) == line_len:
			id = [float(line[0])]
			id += map(float, line[1:])
			data.append(np.array(id))
	return np.array(data)


def remove_files(fname):
	"""
	Remove all files in list
	:param fname:
	:return:
	"""
	from parameters import isiterable
	if isiterable(fname):
		for ff in fname:
			if os.path.isfile(ff) and os.path.getsize(ff) > 0:
				os.remove(ff)
	else:
		if os.path.isfile(fname) and os.path.getsize(fname) > 0:
			os.remove(fname)


def process_template(template_file, field_dictionary, save_to=None):
	"""
	Read a template file and replace the provided fields
	:param template_file:
	:param field_dictionary:
	:return:
	"""
	res = []

	with open(template_file, 'r') as fp:
		tf = fp.readlines()

	for line in tf:
		new_line = line
		for k, v in field_dictionary.items():
			new_line = new_line.replace(k, v)
		res.append(new_line)

	if save_to:
		with open(save_to, 'w') as fp:
			fp.writelines(res)

	return res


def set_storage_locations(parameter_set, save=True):
	"""
	Define paths to store data
	:param parameter_set: ParameterSet object
	:param save: [bool] is False, no paths are created
	:return save_paths: dictionary containing all relevant storage locations
	"""
	if save:
		print "\nSetting storage paths..."
		main_folder = parameter_set.kernel_pars.data_path + parameter_set.kernel_pars.data_prefix + '/'
		if not os.path.exists(main_folder):
			os.mkdir(main_folder)
		save_label = parameter_set.label
		figures = main_folder + 'Figures/'
		if not os.path.exists(figures):
			os.mkdir(figures)
		inputs = main_folder + 'Inputs/'
		if not os.path.exists(inputs):
			os.mkdir(inputs)
		parameters = main_folder + 'Parameters/'
		if not os.path.exists(parameters):
			os.mkdir(parameters)
		results = main_folder + 'Results/'
		if not os.path.exists(results):
			os.mkdir(results)
		activity = main_folder + 'Activity/'
		if not os.path.exists(activity):
			os.mkdir(activity)
		others = main_folder + 'Other/'
		if not os.path.exists(others):
			os.mkdir(others)
		return {'main': main_folder, 'figures': figures, 'inputs': inputs, 'parameters': parameters, 'results':
			results, 'activity': activity, 'other': others, 'label': save_label}
	else:
		print "No data will be saved!"
		return {'label': False, 'figures': False, 'activity': False}


# ########################################################################################
class FileHandler(object):
	"""
    Class to handle all the file read/write methods for the key objects of the
    signals class, i.e SpikeList and AnalogSignalList.

    This is an abstract class that will be implemented for each format (txt, pickle, hdf5)
    The key methods of the class are:
        write(object)              - Write an object to a file
        read_spikes(params)        - Read a SpikeList file with some params
        read_analogs(type, params) - Read an AnalogSignalList of type `type` with some params

    Inputs:
        filename - the file name for reading/writing data

    If you want to implement your own file format, you just have to create an object that will
    inherit from this FileHandler class and implement the previous functions.
	"""

	def __init__(self, filename):
		self.filename = filename

	def __str__(self):
		return "%s (%s)" % (self.__class__.__name__, self.filename)

	def write(self, object):
		"""
        Write the object to the file.

        Examples:
            >> handler.write(SpikeListObject)
            >> handler.write(VmListObject)
		"""
		return _abstract_method(self)

	def read_spikes(self, params):
		"""
		Read a SpikeList object from a file and return the SpikeList object, created from the File and
		from the additional params that may have been provided
		Examples:
			>> params = {'id_list' : range(100), 't_stop' : 1000}
			>> handler.read_spikes(params)
				SpikeList Object (with params taken into account)
		"""
		return _abstract_method(self)

	def read_analogs(self, type, params):
		"""
		Read an AnalogSignalList object from a file and return the AnalogSignalList object of type
		`type`, created from the File and from the additional params that may have been provided

		`type` can be in ["vm", "current", "conductance"]

		Examples:
            >> params = {'id_list' : range(100), 't_stop' : 1000}
            >> handler.read_analogs("vm", params)
                VmList Object (with params taken into account)
            >> handler.read_analogs("current", params)
				CurrentList Object (with params taken into account)
		"""
		if not type in ["vm", "current", "conductance"]:
			raise Exception("The type %s is not available for the Analogs Signals" % type)
		return _abstract_method(self)


# ################################################################################################
class DataHandler(object):
	"""
	Class to establish the interface for loading/saving objects

	Inputs:
		filename - the user file for reading/writing data. By default, if this is
				   string, a StandardTextFile is created
		object   - the object to be saved. Could be a SpikeList or an AnalogSignalList

	Examples:
		>> txtfile = StandardTextFile("results.dat")
		>> DataHandler(txtfile)
		>> picklefile = StandardPickleFile("results.dat")
		>> DataHandler(picklefile)

	"""

	def __init__(self, user_file, object=None):
		if type(user_file) == str:
			user_file = StandardTextFile(user_file)
		elif not isinstance(user_file, FileHandler):
			raise Exception("The user_file object should be a string or herits from FileHandler !")
		self.user_file = user_file
		self.object = object

	def load_spikes(self, **params):
		"""
		Function to load a SpikeList object from a file. The data type is automatically
		inferred. Return a SpikeList object

		Inputs:
			params - a dictionary with all the parameters used by the SpikeList constructor

		Examples:
			>> params = {'id_list' : range(100), 't_stop' : 1000}
			>> handler.load_spikes(params)
				SpikeList object

		See also
			SpikeList, load_analogs
		"""

		### Here we should have an automatic selection of the correct manager
		### according to the file format.
		### For the moment, we try the pickle format, and if not working
		### we assume this is a text file
		#print("Loading spikes from %s, with parameters %s" % (self.user_file, params))
		return self.user_file.read_spikes(params)

	def load_analogs(self, type, **params):
		"""
		Read an AnalogSignalList object from a file and return the AnalogSignalList object of type
		`type`, created from the File and from the additional params that may have been provided

		`type` can be in ["vm", "current", "conductance"]

		Examples:
			>> params = {'id_list' : range(100), 't_stop' : 1000}
			>> handler.load_analogs("vm", params)
				VmList Object (with params taken into account)
			>> handler.load_analogs("current", params)
				CurrentList Object (with params taken into account)

		See also
			AnalogSignalList, load_spikes
		"""
		### Here we should have an automatic selection of the correct manager
		### according to the file format.
		### For the moment, we try the pickle format, and if not working
		### we assume this is a text file
		#print("Loading analog signal of type '%s' from %s, with parameters %s" % (type, self.user_file, params))
		return self.user_file.read_analogs(type, params)

	def save(self):
		"""
		Save the object defined in self.object with the method os self.user_file

		Note that you can add your own format for I/O of such NeuroTools objects
		"""
		### Here, you could add your own format if you have created the appropriate
		### manager.
		### The methods of the manager are quite simple: should just inherits from the FileHandler
		### class and have read() / write() methods
		if self.object is None:
			raise Exception("No object has been defined to be saved !")
		else:
			self.user_file.write(self.object)


#################################################################################################
class StandardPickleFile(FileHandler):
	# There's something kinda wrong with this right now...
	def __init__(self, filename):
		FileHandler.__init__(self, filename)
		self.metadata = {}

	def __fill_metadata(self, object):
		"""
		Fill the metadata from those of a NeuroTools object before writing the object
		"""
		self.metadata['dimensions'] = str(object.dimensions)
		self.metadata['first_id'] = np.min(object.id_list)
		self.metadata['last_id'] = np.max(object.id_list)
		if hasattr(object, 'dt'):
			self.metadata['dt'] = object.dt

	def __reformat(self, params, object):
		self.__fill_metadata(object)
		if 'id_list' in params and params['id_list'] is not None:
			id_list = params['id_list']
			if isinstance(id_list, int):  # allows to just specify the number of neurons
				params['id_list'] = range(id_list)
		do_slice = False
		t_start = object.t_start
		t_stop = object.t_stop
		if 't_start' in params and params['t_start'] is not None and params['t_start'] != object.t_start:
			t_start = params['t_start']
			do_slice = True
		if 't_stop' in params and params['t_stop'] is not None and params['t_stop'] != object.t_stop:
			t_stop = params['t_stop']
			do_slice = True
		if do_slice:
			object = object.time_slice(t_start, t_stop)
		return object

	def write(self, object):
		fileobj = file(self.filename, "w")
		return cPickle.dump(object, fileobj)

	def read_spikes(self, params):
		fileobj = file(self.filename, "r")
		object = cPickle.load(fileobj)
		object = self.__reformat(params, object)
		return object

	def read_analogs(self, type, params):
		return self.read_spikes(params)


#################################################################################################
class StandardTextFile(FileHandler):
	def __init__(self, filename):
		FileHandler.__init__(self, filename)
		self.metadata = {}

	def __read_metadata(self):
		"""
		Read the informations that may be contained in the header of
		the NeuroTools object, if saved in a text file
		"""
		cmd = ''
		variable = None
		label = None
		f = open(self.filename, 'r')
		for line in f.readlines():
			if line[0] == '#':
				if line[1:].strip().find('variable') != -1:
					variable = line[1:].strip().split(" = ")
				elif line[1:].strip().find('label') != -1:
					label = line[1:].strip().split(" = ")
				else:
					cmd += line[1:].strip() + ';'
			else:
				break
		f.close()
		exec cmd in None, self.metadata
		if not variable is None:
			self.metadata[variable[0]] = variable[1]
		if not variable is None:
			self.metadata[label[0]] = label[1]

	def __fill_metadata(self, object):
		"""
		Fill the metadata from those of a NeuroTools object before writing the object
		"""
		self.metadata['dimensions'] = str(object.dimensions)
		if len(object.id_list) > 0:
			self.metadata['first_id'] = np.min(object.id_list)
			self.metadata['last_id'] = np.max(object.id_list)
		if hasattr(object, "dt"):
			self.metadata['dt'] = object.dt

	def __check_params(self, params):
		"""
		Establish a control/completion/correction of the params to create an object by
		using comparison and data extracted from the metadata.
		"""
		if 'dt' in params:
			if params['dt'] is None and 'dt' in self.metadata:
				#print("dt is infered from the file header")
				params['dt'] = self.metadata['dt']
		if not ('id_list' in params) or (params['id_list'] is None):
			if ('first_id' in self.metadata) and ('last_id' in self.metadata):
				params['id_list'] = range(int(self.metadata['first_id']), int(self.metadata['last_id']) + 1)
				#print("id_list (%d...%d) is infered from the file header" % (
				#int(self.metadata['first_id']), int(self.metadata['last_id']) + 1))
			else:
				raise Exception("id_list can not be infered while reading %s" % self.filename)
		elif isinstance(params['id_list'], int):  # allows to just specify the number of neurons
			params['id_list'] = range(params['id_list'])
		elif not isinstance(params['id_list'], list):
			raise Exception("id_list should be an int or a list !")
		if not ('dims' in params) or (params['dims'] is None):
			if 'dimensions' in self.metadata:
				params['dims'] = self.metadata['dimensions']
			else:
				params['dims'] = len(params['id_list'])
		return params

	def get_data(self, sepchar="\t", skipchar="#"):
		"""
		Load data from a text file and returns an array of the data
		"""
		with open(self.filename, "r") as myfile:
			contents = myfile.readlines()
		data = []
		header = True
		idx = 0
		while header and idx < len(contents):
			if contents[idx][0] != skipchar:
				header = False
				break
			idx += 1
		for i in xrange(idx, len(contents)):
			line = contents[i].strip().split(sepchar)
			id = [float(line[-1])]
			id += map(float, line[0:-1])
			data.append(id)
		#print("Loaded %d lines of data from %s" % (len(data), self))
		data = np.array(data, np.float32)
		return data

	def write(self, object):
		# can we write to the file more than once? In this case, should use seek, tell
		# to always put the header information at the top?
		# write header
		self.__fill_metadata(object)
		fileobj = open(self.filename, 'w')
		header_lines = ["# %s = %s" % item for item in self.metadata.items()]
		fileobj.write("\n".join(header_lines) + '\n')
		np.savetxt(fileobj, object.raw_data(), fmt='%g', delimiter='\t')
		fileobj.close()

	def read_spikes(self, params):
		self.__read_metadata()
		p = self.__check_params(params)
		import signals

		data = self.get_data()
		result = signals.SpikeList(data, p['id_list'], p['t_start'], p['t_stop'], p['dims'])
		del data
		return result

	def read_analogs(self, type, params):
		self.__read_metadata()
		p = self.__check_params(params)
		import signals

		if type == "vm":
			return signals.VmList(self.get_data(), p['id_list'], p['dt'], p['t_start'], p['t_stop'], p['dims'])
		elif type == "current":
			return signals.CurrentList(self.get_data(), p['id_list'], p['dt'], p['t_start'], p['t_stop'], p['dims'])
		elif type == "conductance":
			data = np.array(self.get_data())
			if len(data[0, :]) > 2:
				g_exc = signals.ConductanceList(data[:, [0, 1]], p['id_list'], p['dt'], p['t_start'], p['t_stop'],
				                                p['dims'])
				g_inh = signals.ConductanceList(data[:, [0, 2]], p['id_list'], p['dt'], p['t_start'], p['t_stop'],
				                                p['dims'])
				return [g_exc, g_inh]
			else:
				return signals.ConductanceList(data, p['id_list'], p['dt'], p['t_start'], p['t_stop'], p['dims'])


########################################################################################################################
class Standardh5File(object):
	"""
	Handle h5py dictionaries
	"""

	def __init__(self, filename):
		assert(check_dependency('h5py')), "h5py not found"
		import h5py
		import os
		self.filename = filename
		assert(os.path.isfile(filename)), 'File %s not found' % filename
		assert(os.access(filename, os.R_OK)), 'File %s not readable, change permissions' % filename
		assert(os.access(filename, os.W_OK)), 'File %s not writable, change permissions' % filename

	def __str__(self):
		return "%s" % (self.filename)

	def load(self):
		"""
		Loads h5-file and extracts the dictionary within it.

		Outputs:
		  dict - dictionary, one or several pairs of string and any type of variable,
		         e.g dict = {'name1': var1,'name2': var2}
		"""
		assert(check_dependency('h5py')), "h5py not found"
		import h5py
		print("Loading %s" % self.filename)
		with h5py.File(self.filename, 'r') as f:
			dict = {}
			for k, v in zip(f.keys(), f.values()):
				dict[k] = np.array(v[:])
		return dict

	def save(self, dictionary):
		"""
		Stores a dictionary (dict), in a file (filename).
		Inputs:
		  filename - a string, name of file to store the dictionary
		  dict     - a dictionary, one or several pairs of string and any type of variable,
		             e.g. dict = {'name1': var1,'name2': var2}
		"""
		assert(check_dependency('h5py')), "h5py not found"
		import h5py
		f = h5py.File(self.filename, 'w')
		for k in dictionary:
			f[k] = dictionary[k]
		f.close()