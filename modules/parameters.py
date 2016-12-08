from dill.source import indent

__author__ = 'duarte'
"""
====================================================================================
Parameters Module
====================================================================================
(adapted and modified from NeuroTools.parameters)

Module for dealing with model parameters

Classes
-------
Parameter - simple, single parameter class
ParameterRange - specify list or array of possible values for a given parameter
ParameterSet - represent/manage hierarchical parameter sets
ParameterDerived - specify a parameter derived from the value of another

Functions
---------
set_params_dict - import multiple parameter dictionaries from a python script
				and gathers them all in	a single dictionary, which can then
				be used to create a ParameterSet object
extract_nestvalid_dict - verify whether the parameters dictionary are in the correct format,
				with adequate keys, in agreement with the nest parameters dictionaries
				 so that they can later be passed as direct input to nest
import_mod_file - import a module given the path to its file
isiterable - check whether input is iterable (list, dictionary, array...)
nesteddict_walk - walk a nested dictionary structure and return an iterator
contains_instance - check whether instances are part of object
nesteddict_flatten - flatten a nested dictionary
load_parameters - easiest way to create a ParameterSet from a dictionary file
string_table - convert a table written as multi-line string into a nested dictionary
copy_dict - copies a dictionary and updates it with extra key-value pairs
"""

import numpy as np
np.set_printoptions(threshold=np.nan)
#from os import path
import os
import sys
import types
import time
import itertools
try:
	import ntpath
except ImportError as e:
	raise ImportError("Import dependency not met: %s" % e)
from modules.io import *
import scipy.stats as st
import inspect
import pprint


##########################################################################################
def set_params_dict(source_file):
	"""
	Import multiple parameter dictionaries from a python script and gathers them all in
	a single dictionary, which can then be used to create a ParameterSet object
	:param source_file: [string] path+filename of source script or, if source script is in current directory,
	it can be given as just the filename without extension
	:return: Full dictionary
	"""
	d = dict()

	if os.path.isfile(source_file):
		module_name = import_mod_file(source_file)
	else:
		__import__(source_file)
		module_name = source_file

	for attr, val in eval(module_name).__dict__.iteritems():
		if isinstance(val, dict) and attr != '__builtins__':
			d[str(attr)] = val

	return d


def extract_nestvalid_dict(d, param_type='neuron'):
	"""
	Verify whether the parameters dictionary are in the correct format, with adequate keys, in agreement with the nest
	parameters dictionaries so that they can later be passed as direct input to nest
	:param d: parameter dictionary
	:param param_type: type of parameters - kernel, neuron, population, network, connections, topology,
	:return: valid dictionary
	"""
	# QUESTION why import nest here? is it not already imported? and if not, it should be!
	import nest
	if param_type == 'neuron' or param_type == 'synapse' or param_type == 'device':
		assert d['model'] in nest.Models(), "Model %s not currently implemented in %s" % (d['model'], nest.version())

		accepted_keys = nest.GetDefaults(d['model']).keys()
		accepted_keys.remove('model')
	elif param_type == 'kernel':
		accepted_keys = nest.GetKernelStatus().keys()
		# nest_dict = {k: v for k, v in d.iteritems() if k in accepted_keys}
	else:
		# TODO
		print("{!s} not implemented yet".format(param_type))
		assert False

	# DONE added warning for unknown kernel dictionary entries
	nest_dict = {}
	for k, v in d.iteritems():
		if k in accepted_keys:
			nest_dict[k] = v
		else:
			pass
			# print("WARNING: unknown kernel parameter `{}`!".format(k))
	return nest_dict


def isiterable(x):
	"""
	Verify if input is iterable (list, dictionary, array...)
	:param x: input
	:return: boolean
	"""
	return hasattr(x, '__iter__') and not isinstance(x, basestring)


def nesteddict_walk(d, separator='.'):
	"""
	Walk a nested dict structure, using a generator

	Nested dictionary entries will be created by joining to the parent dict entry
	separated by separator

	:param d: dictionary
	:param separator:
	:return: generator object (iterator)
	"""

	for key1, value1 in d.items():
		if isinstance(value1, dict):
			for key2, value2 in nesteddict_walk(value1, separator):  # recurse into subdict
				yield "%s%s%s" % (key1, separator, key2), value2
		else:
			yield key1, value1


def import_mod_file(full_path_to_module):
	"""
	import a module from a path
	:param full_path_to_module:
	:return: imports module
	"""
	module_dir, module_file = os.path.split(full_path_to_module)
	module_name, module_ext = os.path.splitext(module_file)
	sys.path.append(module_dir)
	try:
		module_obj = __import__(module_name)
		module_obj.__file__ = full_path_to_module
		globals()[module_name] = module_obj
		return module_name
	except Exception as er:
		raise ImportError("Unable to load module {0}, check if the name is repeated with other scripts in "
		                  "path. Error is {1}".format(str(module_name), str(er)))


def contains_instance(search_instances, cls):
	"""
	Check whether any of the search instances is in cls
	"""
	return any(isinstance(o, cls) for o in search_instances)


def nesteddict_flatten(d, separator='.'):
	"""
	Return a flattened version of a nested dict structure.

	Composite keys are created by joining each key to the key of the parent dict using `separator`.
	:param d:
	:param separator:
	:return: flattened dictionary (no nesting)
	"""
	flatd = {}
	for k, v in nesteddict_walk(d, separator):
		flatd[k] = v

	return flatd


def load_parameters(parameters_url, **modifications):
	"""
	Load a ParameterSet from a url to a text file with parameter dictionaries
	:param parameters_url: [str] full path to file
	:param modifications: [path=value], where path is a . (dot) delimited path to
	a parameter in the  parameter tree rooted in this ParameterSet instance
	:return: ParameterSet
	"""
	parameters = ParameterSet(parameters_url)
	parameters.replace_values(**modifications)

	return parameters


def string_table(tablestring):
	"""
	Convert a table written as a multi-line string into a dict of dicts.
	:param tablestring:
	:return:
	"""
	tabledict = {}
	rows = tablestring.strip().split('\n')
	column_headers = rows[0].split()
	for row in rows[1:]:
		row = row.split()
		row_header = row[0]
		tabledict[row_header] = {}
		for col_header, item in zip(column_headers[1:], row[1:]):
			tabledict[row_header][col_header] = float(item)
	return tabledict


def copy_dict(source_dict, diffs={}):
	"""
	Returns a copy of source dict, updated with the new key-value pairs provided
	:param source_dict: dictionary to be copied and updated
	:param diffs: new key-value pairs to add
	:return: copied and updated dictionary
	"""
	assert isinstance(source_dict, dict), "Input to this function must be a dictionary"
	result = source_dict.copy()
	result.update(diffs)
	return result


def compare_dict(dict1, dict2):
	"""
	Compares 2 dictionaries
	:param dict1:
	:param dict2:
	:return: bool
	"""
	return dict1 == dict2


def remove_all_labels(parameter_set):
	"""
	Removes all the 'label' entries from the parameter set
	:param parameter_set:
	:return:
	"""
	new_pars = {k: v for k, v in parameter_set.items() if k != 'label'}
	for k, v in parameter_set.items():
		if k != 'label':
			if isinstance(v, dict) and 'label' in v.keys():
				# for k1, v1 in v.items():
					# if k1 != 'label':
				new_pars[k] = {k1: v1 for k1, v1 in v.items() if k1 != 'label'}
			new_pars[k] = v
	return ParameterSet(new_pars)


def delete_keys_from_dict(dict_del, the_keys):
	"""
    Delete the keys present in the lst_keys from the dictionary.
    Loops recursively over nested dictionaries.
    """
	# make sure the_keys is a set to get O(1) lookups
	if type(the_keys) is not set:
		the_keys = set(the_keys)
	for k,v in dict_del.items():
		if k in the_keys:
			del dict_del[k]
		if isinstance(v, dict):
			delete_keys_from_dict(v, the_keys)
	return dict_del


######################################################################################
class Parameter(object):
	"""
	Simpler class specifying single parameters
	"""

	def __init__(self, name, value, units=None):
		self.name = name
		self.value = value
		self.units = units
		self.type = type(value)

	def __repr__(self):
		s = "%s = %s" % (self.name, self.value)
		if self.units is not None:
			s += " %s" % self.units
		return s


#########################################################################################
class ParameterSet(dict):
	"""
	Class to manage hierarchical parameter sets

	Usage example:

		> sim_params = ParameterSet({'dt': 0.1, 'tstop': 1000.0})
		> exc_cell_params = ParameterSet({'tau_m': 20.0, 'cm': 0.5})
		> inh_cell_params = ParameterSet({'tau_m': 15.0, 'cm': 0.5})
		> network_params = ParameterSet({'excitatory_cells': exc_cell_params, 'inhibitory_cells': inh_cell_params})
		> P = ParameterSet({'sim': sim_params, 'network': network_params})
		> P.sim.dt
		0.1
		> P.network.inhibitory_cells.tau_m
		15.0
	"""

	@staticmethod
	def read_from_file(pth, filename):
		"""
		Import parameter dictionary stored as a text file. The
		file must be in standard format, i.e. {'key1': value1, ...}
		:param pth: path
		:param filename:
		:return:
		"""

		assert os.path.exists(pth), "Incorrect path..."
		#assert os.path.isfile(filename), "Incorrect filename..."
		
		## !!! TODO modify !!
		if pth[-1] != '/':
			pth += '/'
		with open(pth + filename, 'r') as fi:
			contents = fi.read()
		D = eval(contents)

		return D

	def __init__(self, initializer, label='global'):
		"""

		:param initializer: parameters dictionary, or string locating the full path of text file with
		parameters dictionary written in standard format
		:param label: label the parameters set
		:return: ParameterSet
		"""

		def convert_dict(d, label):
			"""
			Iterate through the dictionary d, replacing all items by ParameterSet objects
			:param d:
			:param label:
			:return:
			"""
			for k, v in d.items():
				if isinstance(v, ParameterSet):
					d[k] = v
				elif isinstance(v, dict):
					d[k] = convert_dict(d, k)
				else:
					d[k] = v
			return ParameterSet(d, label)

		# self._url = None
		if isinstance(initializer, basestring):  # url or str
			if os.path.exists(initializer):
				with open(initializer, 'r') as f:
					pstr = f.read()
				# self._url = initializer

				pth = ntpath.dirname(initializer)
				filename = ntpath.basename(initializer)
				initializer = ParameterSet.read_from_file(pth, filename)

		# initializer is now a dictionary (if it was a path before), and we can iterate it and replace all items by
		# ParameterSets
		if isinstance(initializer, dict):
			for k, v in initializer.items():
				#print k
				if isinstance(v, ParameterSet):
					self[k] = v

				elif isinstance(v, dict):
					self[k] = ParameterSet(v)

				else:
					self[k] = v
		else:
			raise TypeError("initializer must be either a string specifying "
			                "the full path of the parameters file, "
			                "or a parameters dictionary")

		# set label
		if isinstance(initializer, dict):
			if 'label' in initializer:
				self.label = initializer['label']
			else:
				self.label = label
		elif hasattr(initializer, 'label'):
			self.label = label or initializer.label
		else:
			self.label = label

		# Define some aliases, allowing, e.g. for name, value in P.parameters() or for name in P.names()...
		# self.names = self.keys
		# self.parameters = self.items

	def flat(self):
		__doc__ = nesteddict_walk.__doc__
		return nesteddict_walk(self)

	def flatten(self):
		__doc__ = nesteddict_flatten.__doc__
		return nesteddict_flatten(self)


	def __eq__(self, other):
		"""
		Simple function for equality check of ParameterSet objects. Check is done implicitly by converting the objects
		to dictionaries.
		:param other:
		:return:
		"""
		# compare instances
		if not isinstance(self, other.__class__):
			return False
		# return 	self.as_dict() == other.as_dict()
		# ########################################
		# DEBUGGING
		ad, od = self.as_dict(), other.as_dict()
		for key in ad:
			if ad[key] != od[key]:
				# pp = pprint.PrettyPrinter(indent=4)
				# print "KEY: ", key
				# pp.pprint(ad[key])
				# pp.pprint(od[key])
				return False
		return True

	def __getattr__(self, name):
		"""
		Allow accessing parameters using dot notation.
		"""
		try:
			return self[name]
		except KeyError:
			return self.__getattribute__(name)

	def __setattr__(self, name, value):
		"""
		Allow setting parameters using dot notation.
		"""
		self[name] = value

	def __getitem__(self, name):
		"""
		Modified get that detects dots '.' in the names and goes down the
		nested tree to find it
		"""
		split = name.split('.', 1)
		if len(split) == 1:
			return dict.__getitem__(self, name)
		# nested get
		return dict.__getitem__(self, split[0])[split[1]]

	def flat_add(self, name, value):
		"""
		Like `__setitem__`, but it will add `ParameterSet({})` objects
		into the namespace tree if needed.
		"""

		split = name.split('.', 1)
		if len(split) == 1:
			dict.__setitem__(self, name, value)
		else:
			# nested set
			try:
				ps = dict.__getitem__(self, split[0])
			except KeyError:
				# setting nested name without parent existing
				# create parent
				ps = ParameterSet({})
				dict.__setitem__(self, split[0], ps)
			# and try again
			ps.flat_add(split[1], value)

	def __setitem__(self, name, value):
		"""
		Modified set that detects dots '.' in the names and goes down the
		nested tree to set it
		"""
		if isinstance(name, str):
			split = name.split('.', 1)
			if len(split) == 1:
				dict.__setitem__(self, name, value)
			else:
				# nested set
				dict.__getitem__(self, split[0])[split[1]] = value

	def update(self, E, **F):
		"""
		Update ParameterSet with dictionary entries
		"""
		if hasattr(E, "has_key"):
			for k in E:
				self[k] = E[k]
		else:
			for (k, v) in E:
				self[k] = v
		for k in F:
			self[k] = F[k]

	def __getstate__(self):
		"""
		For pickling.
		"""
		return self

	def save(self, url=None):
		"""
		Write the ParameterSet to a text file
		The text format should be valid python code...
		:param url: locator (full path+ filename, str) - if None, data will be saved to self._url
		"""

		if not url:
			print "Please provide url"
			# url = self._url
		assert url != ''
		# if not self._url:
		# 	self._url = url

		with open(url, 'w') as fp:
			fp.write(self.pretty())

	def pretty(self, indent='   '):
		"""
		Return a unicode string representing the structure of the `ParameterSet`.
		evaluating the string should recreate the object.
		:param indent: indentation type
		:return: string
		"""

		def walk(d, indent, ind_incr):
			s = []
			for k, v in d.items():
				if isinstance(v, list):
					if k == 'randomize_neuron_pars' and np.mean([isinstance(x, dict) for x in v]) == 1.:
						s.append('%s"%s": [' % (indent, k))
						for x in v:
							s.append('{')
							s.append(walk(x, indent + ind_incr, ind_incr))
							s.append('},\n')
						s.append('],\n')
						#
						# 	for k1, v1 in x.items():
						# 		s.append('%s"%s": {' % (indent, k1))
						# 		s.append(walk(v1, indent + ind_incr, ind_incr))
						# 		s.append('%s},' % indent)
						# 	s.append('},')
						# s.append('],')
						v_arr = np.array([])
						continue
					elif np.mean([isinstance(x, list) for x in v]):
						v_arr = np.array(list(itertools.chain(*v)))
					else:
						v_arr = np.array(v)
					mark = np.mean([isinstance(x, tuple) for x in v])
					if (v_arr.dtype == object) and mark:
						if len(v) == 1:
							if v and isinstance(v[0], tuple) and isinstance(v[0][0], types.BuiltinFunctionType):
								if isinstance(v_arr.any(), types.BuiltinFunctionType):
									if v_arr.any().__name__ in np.random.__all__:
										s.append('%s"%s": [(%s, %s)],' % (indent, k, 'np.random.{0}'.format(v_arr.any(
										).__name__), str(v_arr.all())))
								elif isinstance(v_arr.any(), types.MethodType) and v_arr.any().__str__().find('scipy'):
									s.append('%s"%s": [(%s, %s)],' % (indent, k, 'st.{0}.rvs'.format(str(v_arr.any().im_self.name)),
									                                  str(v_arr.all())))
						else:
							if v and isinstance(v_arr.any(), tuple):
								if isinstance(v_arr.any()[0], types.BuiltinFunctionType):
									list_idx = np.where(v_arr.any() in v)[0][0]
									string = '%s"%s": [' % (indent, k)
									for idx, nnn in enumerate(v):
										if idx == list_idx:
											tmp = np.array(nnn)
											string += '(%s, %s), ' % ('np.random.{0}'.format(tmp.any().__name__), str(tmp.all()))
										elif nnn is not None and isinstance(nnn[0], types.BuiltinFunctionType):
											tmp = np.array(nnn)
											string += '(%s, %s), ' % ('np.random.{0}'.format(tmp.any().__name__), str(tmp.all()))
										# elif nnn is not None and isinstance(nnn[0], types.MethodType):
										# 	tmp = np.array(nnn)
										# 	string += '(%s, %s), ' % ('st.{0}.rvs'.format(str(tmp.any(
										# 													).im_self.name)), str(tmp.all()))
										else:
											string += '%s, ' % (str(nnn))
									string += '], '
									s.append(string)
							elif v and isinstance(v_arr.any(), types.BuiltinFunctionType) and isinstance(v_arr.all(),
							                                                                             dict):
								string = '%s"%s": [' % (indent, k)
								for idx, nnn in enumerate(v):
									tmp = np.array(nnn)
									string += '(%s, %s), ' % (
									'np.random.{0}'.format(tmp.any().__name__), str(tmp.all()))
								string += '], '
								s.append(string)
							# elif v and isinstance(v_arr.any(), types.MethodType) and isinstance(v_arr.all(), dict):
							# 	string = '%s"%s": [' % (indent, k)
							# 	for idx, nnn in enumerate(v):
							# 		tmp = np.array(nnn)
							# 		string += '(%s, %s), ' % (
							# 		'st.{0}.rvs'.format(tmp.any().im_self.name), str(tmp.all()))
							# 	string += '], '
							# 	s.append(string)
					else:
						if np.mean([isinstance(x, np.ndarray) for x in v]):
							string = '%s"%s": [' % (indent, k)
							for idx, n in enumerate(v):
								if isinstance(n, np.ndarray):
									string += 'np.' + repr(n) + ', '
								else:
									string += str(n) + ', '
							string += '], '
							s.append(string)
						elif hasattr(v, 'items'):
							# if hasattr(v, '_url') and v._url:
							# s.append('%s"%s": url("%s"),' % (indent, k, v._url))
							# else:
							s.append('%s"%s": {' % (indent, k))
							s.append(walk(v, indent + ind_incr, ind_incr))
							s.append('%s},' % indent)
						elif isinstance(v, basestring):
							s.append('%s"%s": "%s",' % (indent, k, v))
						else:  # what if we have a dict or ParameterSet inside a list? currently they are not expanded. Should they be?
							s.append('%s"%s": %s,' % (indent, k, v))

				elif isinstance(v, types.BuiltinFunctionType):
					if v.__name__ in np.random.__all__:
						s.append('%s"%s"' % (indent, 'np.random.{0}'.format(v.__name__)))
					# else:
					# 	continue
				else:
					if hasattr(v, 'items'):
						# if hasattr(v, '_url') and v._url:
						# 	s.append('%s"%s": url("%s"),' % (indent, k, v._url))
						# else:
						s.append('%s"%s": {' % (indent, k))
						s.append(walk(v, indent + ind_incr, ind_incr))
						s.append('%s},' % indent)
					elif isinstance(v, basestring):
						s.append('%s"%s": "%s",' % (indent, k, v))
					elif isinstance(v, np.ndarray):
						s.append('%s"%s": %s,' % (indent, k, repr(v)[6:-1]))
					elif isinstance(v, tuple) and (isinstance(v[0], types.MethodType) or isinstance(v[0], types.BuiltinFunctionType)):
						v_arr = np.array(v)
						if isinstance(v[0], types.MethodType):
							s.append('%s"%s": (%s, %s),' % (indent, k, 'st.{0}.rvs'.format(str(v_arr.any().im_self.name)),
						                                  str(v_arr.all())))
						elif isinstance(v[0], types.BuiltinFunctionType):
							s.append('%s"%s": (%s, %s),' % (indent, k, 'np.random.{0}'.format(v_arr.any(
										).__name__), str(v_arr.all())))
					else:  # what if we have a dict or ParameterSet inside a list? currently they are not expanded. Should they be?
						s.append('%s"%s": %s,' % (indent, k, v))
			return '\n'.join(s)

		return '{\n' + walk(self, indent, indent) + '\n}'

	def tree_copy(self):
		"""
		Return a copy of the `ParameterSet` tree structure.
		Nodes are not copied, but re-referenced.
		"""

		tmp = ParameterSet({})
		for key in self:
			value = self[key]
			if isinstance(value, ParameterSet):
				tmp[key] = value.tree_copy()
			# elif isinstance(value, ParameterReference):
			# 	tmp[key] = value.copy()
			else:
				tmp[key] = value
		# if tmp._is_space():
		# 	tmp = ParameterSpace(tmp)
		return tmp

	def as_dict(self):
		"""
		Return a copy of the `ParameterSet` tree structure
		as a nested dictionary
		"""

		tmp = {}

		for key in self:
			if not isinstance(self[key], types.BuiltinFunctionType):
				value = self[key]
				if isinstance(value, ParameterSet):
					tmp[key] = value.as_dict()
				else:
					tmp[key] = value
		return tmp

	def replace_values(self, **args):
		"""
		This expects its arguments to be in the form path=value, where path is a
		. (dot) delimited path to a parameter in the  parameter tree rooted in
		this ParameterSet instance.

		This function replaces the values of each parameter in the args with the
		corresponding values supplied in the arguments.
		"""
		for k in args.keys():
			self[k] = args[k]

	def clean(self, termination='pars'):
		"""
		Remove fields from ParameterSet that do not contain the termination
		This is mostly useful if, in the specification of the parameters
		additional auxiliary variables were set, and have no relevance for
		the experiment at hand...
		"""
		accepted_keys = [x for x in self.iterkeys() if x[-len(termination):] == termination]
		new_dict = {k: v for k, v in self.iteritems() if k in accepted_keys}

		return ParameterSet(new_dict)


#########################################################################################
class ParameterSpace:
	"""
	A collection of `ParameterSets`, representing multiple points in
	parameter space. Created by putting `ParameterRange` objects within 
	a `ParameterSet`.
	"""

	def __init__(self, initializer, emoo=False, keep_all=False):
		"""
		Generate ParameterSpace containing a list of all ParameterSets
		:param initializer: file url
		:return:
		"""
		assert isinstance(initializer, str), "Filename must be provided"

		def validate_parameters_file(module):
			"""
			Checks for any errors / incompatibilities in the structure of the parameter file. Function
			`build_parameters` is required, with or without arguments.
			:param module: imported parameter module object
			:return:
			"""
			# TODO anything else?
			# build_parameters function must be defined!
			if not hasattr(module, "build_parameters"):
				raise ValueError("`build_parameters` function is missing!")

			# check parameter range and function arguments
			range_arguments = inspect.getargspec(module.build_parameters)[0]
			if not hasattr(module, "parameter_range"):
				if len(range_arguments) == 0:
					return
				raise ValueError("`parameter_range` and arguments of `build_parameters` do not match!")

			for arg in range_arguments:
				if arg not in module.parameter_range:
					raise ValueError('ParameterRange variable `%s` is not in `parameter_range` dictionary!' % arg)
				if not isiterable(module.parameter_range[arg]):
					raise ValueError('ParameterRange variable `%s` is not iterable! Should be list!' % arg)


		def validate_parameter_sets(param_sets):
			# TODO @Q: how / what should we handle and check here?
			"""

			:param param_sets:
			:return:
			"""
			required_dicts = ["kernel_pars", "encoding_pars", "net_pars"]
			for p_set in param_sets:
				for d in required_dicts:
					if d not in p_set:
						raise ValueError("Required parameter (dictionary) `%s` not found!" % d)

				# `data_prefix` required
				if "data_prefix" not in p_set["kernel_pars"]:
					raise ValueError("`data_prefix` missing from `kernel_pars`!")


		def parse_parameters_file(url):
			"""

			:param url:
			:return:
			"""
			module_name = import_mod_file(url)
			module_obj 	= globals()[module_name]
			try:
				validate_parameters_file(module_obj)
			except ValueError as error:
				print ("Invalid parameter file! Error: %s" % error)
				exit(-1)

			range_args	= inspect.getargspec(module_obj.build_parameters)[0]  # arg names in build_parameters function
			n_ranges 	= len(range_args)  # nr range args TODO @continue
			n_runs 	 	= int(np.prod([len(module_obj.parameter_range[arg]) for arg in range_args]))  # nr combinations
			param_axes 	= dict()
			param_sets 	= []

			# build cross product of parameter ranges, sorting corresponds to argument ordering in build_parameters
			# empty if no range parameters are present: [()]
			range_combinations = list(itertools.product(*[module_obj.parameter_range[arg] for arg in range_args]))
			# call build_parameters for each range combination, and pack returned values into a list (set)
			# contains a single dictionary if no range parameters
			param_ranges = [module_obj.build_parameters( *elem ) for elem in range_combinations]
			print param_ranges
			global_label = param_ranges[0]['kernel_pars']['data_prefix']

			if n_ranges <= 3 and not emoo:
				# verify parameter integrity / completeness
				try:
					validate_parameter_sets(param_ranges)
				except ValueError as error:
					print ("Invalid parameter file! Error: %s" % error)

				# build parameter axes
				axe_prefixes = ['x', 'y', 'z']
				for range_index in range(n_runs):
					params_label = global_label
					for axe_index, axe in enumerate(axe_prefixes[:n_ranges]):  # limit axes to n_ranges
						param_axes[axe + 'label'] 		= range_args[axe_index]
						param_axes[axe + 'ticks'] 		= module_obj.parameter_range[param_axes[axe + 'label']]
						param_axes[axe + 'ticklabels'] 	= [str(xx) for xx in param_axes[axe + 'ticks']]
						params_label += '_' + range_args[axe_index]
						params_label += '=' + str(range_combinations[range_index][axe_index])


					p_set = ParameterSet(param_ranges[range_index], label=params_label)
					if not keep_all:
						p_set = p_set.clean(termination='pars')
					p_set.update({'label': params_label})
					param_sets.append(p_set)

			elif (n_ranges > 3) or (emoo is True):
				# TODO @Q: what should exactly happen here?
				print ("Preparing ParameterSpace for EMOO...")
				params_label = global_label

				for nn_rang in range(n_ranges):
					param_axes.update({range_args[nn_rang]: module_obj.parameter_range[range_args[nn_rang]]})
					params_label += '_' + range_args[nn_rang] + '=' + str(module_obj.parameter_range[range_args[nn_rang]][0])

				p_set = ParameterSet(param_ranges[0], label=params_label)
				if not keep_all:
					p_set = p_set.clean(termination='pars')
				p_set.update({'label': params_label})
				param_sets.append(p_set)

			return param_sets, param_axes, global_label, n_ranges

		def parse_parameters_dict(url):
			"""
			Simple parser for dictionary (text) parameter scripts, not .py!
			:param url:
			:return:
			"""
			# TODO is set label always global?
			param_set 	= ParameterSet(url, label='global')
			try:
				validate_parameter_sets([param_set])
			except ValueError as error:
				print ("Invalid parameter file! Error: %s" % error)

			param_axes	= {}
			label 		= param_set.kernel_pars["data_prefix"]
			dim			= 1
			return param_set, param_axes, label, dim


		if initializer.endswith(".py"):
			self.parameter_sets, self.parameter_axes, self.label, self.dimensions = parse_parameters_file(initializer)
		else:
			self.parameter_sets, self.parameter_axes, self.label, self.dimensions = parse_parameters_dict(initializer)

		if emoo:
			self.additional_parameters = dict()
			from modules.analysis import Emoo


	def iter_sets(self):
		"""
		An iterator which yields the ParameterSets in ParameterSpace
		"""
		for val in self.parameter_sets:
			yield val


	def __eq__(self, other):
		"""
		@barni Added this function for testing purposes.
		:param other:
		:return:
		"""
		if not isinstance(other, self.__class__):
			return False

		for key in self.parameter_axes:
			if key not in other.parameter_axes:
				return False
			else:
				if isinstance(self.parameter_axes[key], np.ndarray):
					if not np.array_equal(self.parameter_axes[key], other.parameter_axes[key]):
						return False
				elif self.parameter_axes[key] != other.parameter_axes[key]:
					return False

		if self.label != other.label or self.dimensions != other.dimensions:
			return False

		return self.parameter_sets == other.parameter_sets


	def __getitem__(self, item):
		return self.parameter_sets[item]

	def __len__(self):
		return len(self.parameter_sets)

	def save(self, target_full_path):
		"""
		Save the full ParameterSpace by re-writing the parameter file
		:return:
		"""
		with open(target_full_path, 'w') as fp:
			fp.writelines(self.parameter_file)

	def compare_sets(self, parameter):
		"""
		Determine whether a given parameter is common to all parameter sets
		:param parameter: parameter to compare
		"""
		common = dict(pair for pair in self.parameter_sets[0].items() if all((pair in d.items() for d in
		                                                                      self.parameter_sets[1:])))
		result = False
		if parameter in common.keys():
			result = True
		else:
			for k, v in common.items():
				if isinstance(v, dict):
					if parameter in v.keys():
						result = True
		return result

	def run(self, computation_function, **parameters):
		"""
		Run a computation on all the parameters
		:param computation_function: function to execute
		:param parameters: kwarg arguments for the function
		"""
		system = self.parameter_sets[0].kernel_pars.system

		if system['local']:
			print "\nRunning {0} serially on {1} Parameter Sets".format(str(computation_function), str(len(self)))

			for par_set in self.parameter_sets:
				print "\n- Parameters: {0}".format(str(par_set.label))
				results = computation_function(par_set, **parameters)
			return results
		else:
			print "\nPreparing job description files..."
			export_folder = system['remote_directory']
			import os
			if not os.path.exists(export_folder + '{0}'.format(self.label)):
				os.mkdir(export_folder + '{0}'.format(self.label))
			main_experiment_folder = export_folder + '{0}/'.format(self.label)
			remote_run_folder = export_folder + self.label + '/'
			if system['system_label'] == 'bwUniCluster':
				py_file_common_header = "import sys\nsys.path.append('../')\nfrom Modules.parameters import *\nfrom " \
				                        "Modules.analysis import *\nfrom Experiments.Computations import *\n\n"
			else:
				py_file_common_header = "import sys\nsys.path.append('../')\nimport matplotlib\nmatplotlib.use(" \
				                        "'Agg')\nfrom Modules.parameters import *\nfrom " \
				                        "Modules.analysis import *\nfrom Experiments.Computations import *\n\n"

			write_to_submit = []
			submit_jobs_file = main_experiment_folder + 'submit_jobs.py'
			job_list = main_experiment_folder + 'job_list.txt'
			cancel_jobs_file = main_experiment_folder + 'cancel_jobs.py'

			for par_set in self.parameter_sets:
				system2 = par_set.kernel_pars.system
				template_file = system2['jdf_template']
				exec_file_name = remote_run_folder + par_set.label + '.sh'
				local_exec_file = main_experiment_folder + par_set.label + '.sh'
				computation_file = main_experiment_folder + par_set.label + '.py'
				remote_computation_file = remote_run_folder + par_set.label + '.py'

				par_set.save(main_experiment_folder + par_set.label+'.txt')
				parameters_file = main_experiment_folder + par_set.label+'.txt'
				with open(computation_file, 'w') as fp:
					fp.write(py_file_common_header)
					fp.write(computation_function.__name__ + "({0}, **{1})".format("'./" + par_set.label+".txt'",
					                                                               str(parameters)))

				system2['jdf_fields'].update({'{{ computation_script }}': remote_computation_file,
				                             '{{ script_folder }}': remote_run_folder})

				process_template(template_file, system2['jdf_fields'].as_dict(), save_to=local_exec_file)
				#write_to_submit.append("\t\tos.system('msub {0}')\n".format(exec_file_name))
				write_to_submit.append("{0}\n".format(exec_file_name))

			with open(job_list, 'w') as fp:
				fp.writelines(write_to_submit)

			with open(submit_jobs_file, 'w') as fp:
				fp.write("import os\n")
				fp.write("import sys\n\n")
				fp.write("def submit_jobs(start_idx=0, stop_idx=None):\n")
				#fp.write("")
				fp.write("\twith open('{0}') as fp:\n\t\tfor idx, line in enumerate(fp):\n".format('./job_list.txt'))
				fp.write("\t\t\tif stop_idx is not None:\n")
				fp.write("\t\t\t\tif (idx>=start_idx) and (idx<=stop_idx):\n")
				if system['system_label'] == 'bwUniCluster':
					fp.write("\t\t\t\t\tos.system('msub {0}'.format(line))\n")
					fp.write("\t\t\telse:\n")
					fp.write("\t\t\t\tos.system('msub {0}'.format(line))")
				else:
					fp.write("\t\t\t\t\tos.system('sbatch {0}'.format(line))\n")
					fp.write("\t\t\telse:\n")
					fp.write("\t\t\t\tos.system('sbatch {0}'.format(line))")
				#fp.writelines(write_to_submit)
				fp.write("\n\n")
				fp.write("if __name__=='__main__':\n\tif len(sys.argv)>1:\n\t\tsubmit_jobs(int(sys.argv[1]), "
				         "int(sys.argv[2]))")

			with open(cancel_jobs_file, 'w') as fp:
				if system['system_label'] == 'bwUniCluster':
					fp.write("import os\nimport numpy as np\nimport sys\ndef cancel_range(init, end):\n\trang = np.arange("
					         "init, end)\n\tfor n in rang:\n\t\tos.system('canceljob '+ str(n))\n\nif "
					         "__name__=='__main__':\n\tcancel_range(int(sys.argv[1]), int(sys.argv[2]))")
				else:
					fp.write(
						"import os\nimport numpy as np\nimport sys\ndef cancel_range(init, end):\n\trang = np.arange("
						"init, end)\n\tfor n in rang:\n\t\tos.system('scancel '+ str(n))\n\nif "
						"__name__=='__main__':\n\tcancel_range(int(sys.argv[1]), int(sys.argv[2]))")

	def reset_parameters(self, variables):
		"""
		Prepare a new ParameterSet with the new combination of parameters (for emoo)
		"""
		var = variables
		par_ranges = []
		par_ranges_index = []
		for idx, nn in enumerate(self.parameter_file):
			if nn.find('ParameterRange') != -1:
				par_ranges.append(nn)
				par_ranges_index.append(idx)
		new_data = list(np.copy(self.parameter_file))
		n_ranges = len(par_ranges)

		par_names1 = [n_par[n_par.find('name=') + 5:] for n_par in par_ranges]
		par_names = [n_par[1:n_par.find(',') - 1] for n_par in par_names1]
		idxs = [var.keys().index(n_par) for n_par in par_names]

		params_label = self.label.split('_')[0]
		for nn_rang in range(n_ranges):
			new_data[par_ranges_index[nn_rang]] = par_names[nn_rang] + ' = ' + str(var.values()[idxs[nn_rang]]) + '\n'
			params_label += '_' + par_names[nn_rang] + '=' + str(var.values()[idxs[nn_rang]])

		time_stamp = time.time() # to avoid redundant filenames (were causing problems)
		params_file = "tmp_{0}.py".format(str(int(time_stamp)))
		with open(params_file, 'w') as fp:
			fp.writelines(new_data)
		par_set = ParameterSet(set_params_dict(params_file), label=params_label)
		par_set = par_set.clean(termination='pars')
		par_set.update({'label': params_label})
		os.remove(params_file)
		if os.path.isfile(params_file + 'c'):
			os.remove(params_file + 'c')

		return par_set

	def optimization_run(self, variables, save_to=None):
		"""
		Function to optimize
		:param variables:
		:return:
		"""
		print variables
		parameter_set = self.reset_parameters(variables)
		computation_function = self.additional_parameters['computation_function']
		# TODO try:
		results = computation_function(parameter_set, **self.additional_parameters['parameters'])
		objective_error = dict()

		# Extract the relevant data from the results dictionaries (experiment-specific,
		# specified by results_subfields and operation...)
		if self.additional_parameters['results_subfields'] is None:
			operation = self.additional_parameters['operation']
			if save_to is not None:
				save_to.append({})
			for n in self.additional_parameters['objectives']:
				targets = n.values()[0]

				if operation is not None:
					obtained = operation(results[str(n.keys()[0])])
				else:
					obtained = results[str(n.keys()[0])]
				if len(targets) > 1:
					target = np.mean(targets)
					precision = np.std(targets)
				else:
					target = targets
					precision = 1
				objective_error.update({str(n.keys()[0]): (target - obtained) ** 2})
				if save_to is not None:
					save_to[-1].update({n.keys()[0]: obtained})
		else:
			r = "results"
			for n in self.additional_parameters['results_subfields']:
				r += "['{0}']".format(n)
			results_to_mine = eval(r)

			targets = []
			if save_to is not None:
				save_to.append({})
			#print self.additional_parameters['objectives'].items()
			for nn in self.additional_parameters['objectives'].items():
				operation = self.additional_parameters['operation']
				targets = nn[1]

				if operation is not None:
					obtained = operation(results_to_mine[str(nn[0])])
				else:
					obtained = results_to_mine[str(nn[0])]

				if isinstance(targets, list):
					target = np.mean(targets)
					precision = np.std(targets)
				else:
					target = targets
					precision = 1
				objective_error.update({str(nn[0]): (target - obtained) ** 2})
				if save_to is not None:
					save_to[-1].update({nn[0]: obtained})

		return objective_error

	def checkpopulation(self, population, columns, gen, save_to=None):
		"""
		Check the fitness of the population
		:param population:
		:param columns:
		:param gen:
		:return:
		"""
		objectives = self.additional_parameters['objectives']
		errors = np.zeros_like(population[:, 0])
		for n in objectives.items():
			errors += (population[:, columns[str(n[0])]] ** 2)
		i = np.argmin(errors)

		print "***************************************"
		print "Generation %d: " % gen,
		for n in objectives.items():
			print "Smallest Error for {0} = {1}".format(str(n[0]), str(population[i, columns[str(n[0])]]))
			if save_to is not None:
				save_to['smallest_errors']['Gen{0}'.format(str(gen))].update({str(n[0]): population[i, columns[str(n[0])]]})
		return i

	def run_emoo(self, computation_function, objectives, optimization_options, results_subfields=None,
	             operation=None, **parameters):
		"""
		Run a evolutionary multi-objective parameter optimization using the provided ranges of the free parameters and
		the objectives
		:param computation_function: function to execute
		:param objectives: dictionary with {target_variable_name: target_value} (these have to be keys or
		sub-dictionaries in the
		results dictionary provided by the computation function)
		:param parameters: kwarg arguments for the function
		"""
		self.additional_parameters.update({'objectives': objectives, 'results_subfields': results_subfields,
		                                  'operation': operation, 'parameters': parameters,
		                                  'computation_function': computation_function})

		# ParametersRanges are used to set the parameters to search...
		# Define the variables and their lower and upper search bounds
		params_to_optimize = {k: [min(v), max(v)] for k, v in self.parameter_axes.items()}
		variables = [[k, v[0], v[1]] for k, v in params_to_optimize.items()]

		# Store the results...
		results = {'parameters_evolution': {},
		           'objectives_evolution': {},
		           'error_evolution': {},
		           'smallest_errors': {}}

		# Initiate the Evlutionary Multiobjective Optimization
		from modules.analysis import Emoo
		emoo = Emoo(N=optimization_options['n_individuals'], C=optimization_options['pop_capacity'],
		            variables=variables, objectives=objectives)
		emoo.setup(eta_m_0=optimization_options['eta_m_0'], eta_c_0=optimization_options['eta_c_0'],
		           p_m=optimization_options['p_m'])
		emoo.master_mode = True
		emoo.get_objectives_error = self.optimization_run
		emoo.checkpopulation = self.checkpopulation
		emoo.evolution(optimization_options['n_generations'], save_to=results, save_path=self[0].kernel_pars.data_path +
		                                                                                 self[0].kernel_pars.data_prefix
		                                                                                 + '_EMOO_Results.pck')

		return emoo, results

	def harvest(self, data_path, result=None, operation=None):
		"""
		Gather stored results data and populate the space spanned by the Parameters with the corresponding results
		:param data_path: full path to global data folder
		:param result: specific result to extract from each results dictionary
		:param operation: function to apply to result, if any
		:return:
		"""
		if self.parameter_axes:
			l = ['x', 'y', 'z']
			domain_lens = [len(self.parameter_axes[l[idx]+'ticks']) for idx in range(self.dimensions)]
			domain_values = [self.parameter_axes[l[idx]+'ticks'] for idx in range(self.dimensions)]
			var_names = [self.parameter_axes[l[idx]+'label'] for idx in range(self.dimensions)]
			dom_len = np.prod(domain_lens)

			results_array = np.empty(tuple(domain_lens), dtype=object)
			parameters_array = np.empty(tuple(domain_lens), dtype=object)

			assert len(self) == dom_len, "Domain length inconsistent"
			idx_x = 0; idx_y = 0; idx_z = 0
			pars_labels = [self.parameter_sets[n].label for n in range(len(self))]
			for n in range(int(dom_len)):
				params_label = self.label
				index = []
				if self.dimensions >= 1:
					idx_x = n % domain_lens[0]
					params_label += '_' + var_names[0] + '=' + str(domain_values[0][idx_x])
					index.append(idx_x)
				if self.dimensions >= 2:
					idx_y = (n // domain_lens[0]) % domain_lens[1]
					params_label += '_' + var_names[1] + '=' + str(domain_values[1][idx_y])
					index.append(idx_y)
				if self.dimensions == 3:
					idx_z = ((n // domain_lens[0]) // domain_lens[1]) % domain_lens[2]
					params_label += '_' + var_names[2] + '=' + str(domain_values[2][idx_z])
					index.append(idx_z)

				parameters_array[tuple(index)] = pars_labels[pars_labels.index(params_label)]
				try:
					with open(data_path+'Results_'+params_label, 'r') as fp:
						results = cPickle.load(fp)
					print "Loading ParameterSet {0}".format(self.label)
				except:
					print "Dataset {0} Not Found, skipping".format(params_label)
					continue
				if result:
					assert isinstance(results, dict), "Results must be dictionary"
					for k, v in results.items():
						if k == result:
							if operation:
								results_array[tuple(index)] = operation(v)
							else:
								results_array[tuple(index)] = v
						elif isinstance(v, dict):
							for k1, v1 in v.items():
								if k1 == result:
									if operation:
										results_array[tuple(index)] = operation(v1)
									else:
										results_array[tuple(index)] = v1
								elif isinstance(v1, dict):
									for k2, v2 in v1.items():
										if k2 == result:
											if operation:
												results_array[tuple(index)] = operation(v2)
											else:
												results_array[tuple(index)] = v2
						else:
							raise IOError("Provided result not in results dictionary, or at higher depth")
				else:
					results_array[tuple(index)] = results
		else:
			parameters_array = self.label
			results_array = []
			try:
				with open(data_path+'Results_'+self.label, 'r') as fp:
					results = cPickle.load(fp)
				print "Loading Dataset {0}".format(self.label)
				if result:
					assert isinstance(results, dict), "Results must be dictionary"
					for k, v in results:
						if k == result:
							if operation:
								results_array = operation(v)
							else:
								results_array = v
						elif isinstance(v, dict):
							for k1, v1 in v.items():
								if k1 == result:
									if operation:
										results_array = operation(v1)
									else:
										results_array = v1
								elif isinstance(v1, dict):
									for k2, v2 in v1.items():
										if k2 == result:
											if operation:
												results_array = operation(v2)
											else:
												results_array = v
				else:
					results_array = results
			except:
				print "Dataset {0} Not Found, skipping".format(self.label)

		return parameters_array, results_array

	def harvest_results_from_key(self, data_path, key1=None, key2=None, key3=None, key4=None, operation=None):
		"""
		Gather stored results data, reading and storing only specific keys present in the results dictionary (for memory
		conservation), up to 3 nesting levels, and populate the space spanned by the Parameters with the corresponding
		results
		:param data_path: full path to global data folder
		:param key1: First dictionary key...
		:param key2: Second dictionary key (if result is nested)
		:param operation: some operation to perform on the resulting data
		:return:
		"""
		if self.parameter_axes:
			l = ['x', 'y', 'z']
			domain_lens = [len(self.parameter_axes[l[idx] + 'ticks']) for idx in range(self.dimensions)]
			domain_values = [self.parameter_axes[l[idx] + 'ticks'] for idx in range(self.dimensions)]
			var_names = [self.parameter_axes[l[idx] + 'label'] for idx in range(self.dimensions)]
			dom_len = np.prod(domain_lens)

			results_array = np.empty(tuple(domain_lens), dtype=object)

			assert len(self) == dom_len, "Domain length inconsistent"
			for n in range(int(dom_len)):
				params_label = self.label
				index = []
				if self.dimensions >= 1:
					idx_x = n % domain_lens[0]
					params_label += '_' + var_names[0] + '=' + str(domain_values[0][idx_x])
					index.append(idx_x)
				if self.dimensions >= 2:
					idx_y = (n // domain_lens[0]) % domain_lens[1]
					params_label += '_' + var_names[1] + '=' + str(domain_values[1][idx_y])
					index.append(idx_y)
				if self.dimensions == 3:
					idx_z = ((n // domain_lens[0]) // domain_lens[1]) % domain_lens[2]
					params_label += '_' + var_names[2] + '=' + str(domain_values[2][idx_z])
					index.append(idx_z)

				try:
					with open(data_path + 'Results_' + params_label, 'r') as fp:
						results = cPickle.load(fp)
					print "Loading ParameterSet {0}".format(params_label)
				except:
					print "Dataset {0} Not Found, skipping".format(params_label)
					continue
				if key1 is None and key2 is None and key3 is None and key4 is None:
					assert isinstance(results, dict), "Results data must be a dictionary"
					results_array[tuple(index)] = results

				elif key1 is not None and key2 is None and key3 is None and key4 is None:
					assert isinstance(results, dict), "Results data must be a dictionary"
					assert key1 in results, "{0} not in results dictionary".format(key1)

					if operation is None:
						results_array[tuple(index)] = results[key1]
					else:
						results_array[tuple(index)] = operation(results[key1])

				elif key1 is not None and key2 is not None and key3 is None and key4 is None:
					assert isinstance(results, dict), "Results data must be a dictionary"
					assert key1 in results, "{0} not in results dictionary".format(key1)
					assert key2 in results[key1], "{0} not in subfield {1}".format(key2, key1)

					if operation is None:
						results_array[tuple(index)] = results[key1][key2]
					else:
						results_array[tuple(index)] = operation(results[key1][key2])

				elif key1 is not None and key2 is not None and key3 is not None and key4 is None:
					assert isinstance(results, dict), "Results data must be a dictionary"
					assert key1 in results, "{0} not in results dictionary".format(key1)
					assert key2 in results[key1], "{0} not in subfield {1}".format(key2, key1)
					assert key3 in results[key1][key2], "{0} not in subfield {1}".format(key3, key2)

					if operation is None:
						results_array[tuple(index)] = results[key1][key2][key3]
					else:
						results_array[tuple(index)] = operation(results[key1][key2][key3])
				elif key1 is not None and key2 is not None and key3 is not None and key4 is not None:
					assert isinstance(results, dict), "Results data must be a dictionary"
					assert key1 in results, "{0} not in results dictionary".format(key1)
					assert key2 in results[key1], "{0} not in subfield {1}".format(key2, key1)
					assert key3 in results[key1][key2], "{0} not in subfield {1}".format(key3, key2)
					assert key4 in results[key1][key2][key3], "{0} not in subfield {1}".format(key4, key3)
					if operation is None:
						results_array[tuple(index)] = results[key1][key2][key3][key4]
					else:
						results_array[tuple(index)] = operation(results[key1][key2][key3][key4])
		else:
			results_array = []
			try:
				with open(data_path+'Results_'+self.label, 'r') as fp:
					results = cPickle.load(fp)
				print "Loading Dataset {0}".format(self.label)
			except:
				raise IOError("Dataset {0} Not Found".format(self.label))

			if key1 is not None and key2 is None and key3 is None:
				assert isinstance(results, dict), "Results data must be a dictionary"
				assert key1 in results, "{0} not in results dictionary".format(key1)

				if operation is None:
					results_array = results[key1]
				else:
					results_array = operation(results[key1])

			elif key1 is not None and key2 is not None and key3 is None:
				assert isinstance(results, dict), "Results data must be a dictionary"
				assert key1 in results, "{0} not in results dictionary".format(key1)
				assert key2 in results[key1], "{0} not in subfield {1}".format(key2, key1)

				if operation is None:
					results_array = results[key1][key2]
				else:
					results_array = operation(results[key1][key2])

			elif key1 is not None and key2 is not None and key3 is not None:
				assert isinstance(results, dict), "Results data must be a dictionary"
				assert key1 in results, "{0} not in results dictionary".format(key1)
				assert key2 in results[key1], "{0} not in subfield {1}".format(key2, key1)
				assert key3 in results[key1][key2], "{0} not in subfield {1}".format(key3, key2)

				if operation is None:
					results_array = results[key1][key2][key3]
				else:
					results_array = operation(results[key1][key2][key3])
			elif key1 is not None and key2 is not None and key3 is not None and key4 is not None:
				assert isinstance(results, dict), "Results data must be a dictionary"
				assert key1 in results, "{0} not in results dictionary".format(key1)
				assert key2 in results[key1], "{0} not in subfield {1}".format(key2, key1)
				assert key3 in results[key1][key2], "{0} not in subfield {1}".format(key3, key2)
				assert key4 in results[key1][key2][key3], "{0} not in subfield {1}".format(key4, key3)
				if operation is None:
					results_array = results[key1][key2][key3][key4]
				else:
					results_array = operation(results[key1][key2][key3][key4])

		return results_array #.astype(float)

	def harvest_results_from_keylists(self, data_path, key1=[], key2=None, key3=None, operation=None):
		"""

		:return:
		"""
		if self.parameter_axes:
			l = ['x', 'y', 'z']
			domain_lens = [len(self.parameter_axes[l[idx] + 'ticks']) for idx in range(self.dimensions)]
			domain_values = [self.parameter_axes[l[idx] + 'ticks'] for idx in range(self.dimensions)]
			var_names = [self.parameter_axes[l[idx] + 'label'] for idx in range(self.dimensions)]
			dom_len = np.prod(domain_lens)

			if isinstance(key1, list):
				for n_key in key1:
					globals()['results_{0}'.format(n_key)] = np.empty(tuple(domain_lens))
			elif isinstance(key2, list):
				for n_key in key1:
					globals()['results_{0}'.format(n_key)] = np.empty(tuple(domain_lens))
			elif isinstance(key3, list):
				for n_key in key3:
					globals()['results_{0}'.format(n_key)] = np.empty(tuple(domain_lens))

			#results_array = np.empty(tuple(domain_lens))#, dtype=object)

			assert len(self) == dom_len, "Domain length inconsistent"
			for n in range(int(dom_len)):
				params_label = self.label
				index = []
				if self.dimensions >= 1:
					idx_x = n % domain_lens[0]
					params_label += '_' + var_names[0] + '=' + str(domain_values[0][idx_x])
					index.append(idx_x)
				if self.dimensions >= 2:
					idx_y = (n // domain_lens[0]) % domain_lens[1]
					params_label += '_' + var_names[1] + '=' + str(domain_values[1][idx_y])
					index.append(idx_y)
				if self.dimensions == 3:
					idx_z = ((n // domain_lens[0]) // domain_lens[1]) % domain_lens[2]
					params_label += '_' + var_names[2] + '=' + str(domain_values[2][idx_z])
					index.append(idx_z)

				try:
					with open(data_path + 'Results_' + params_label, 'r') as fp:
						results = cPickle.load(fp)
					print "Loading ParameterSet {0}".format(params_label)
				except:
					print "Dataset {0} Not Found, skipping".format(params_label)
					continue

				if key1 is not None and key2 is None and key3 is None:

					try:
						if isinstance(results, dict) and isinstance(key1, list):
							#assert isinstance(results, dict), "Results data must be a dictionary"
							#assert isinstance(key1, list), "Please provide a list of keys"

							list_len = len(key1)
							results_out = [[] for _ in range(list_len)]

							for iid, n_value in enumerate(key1):
								assert n_value in results, "{0} not in results dictionary".format(n_value)

								if operation is None:
									results_array = results[n_value]
								else:
									results_array = operation(results[n_value])

								globals()['results_{0}'.format(n_value)][tuple(index)] = results_array

								results_out[iid] = (globals()['results_{0}'.format(n_value)]).astype(float)
					except:
						print "Dataset {0} Not complying, skipping".format(params_label)

				elif key1 is not None and key2 is not None and key3 is None:
					#assert isinstance(results, dict), "Results data must be a dictionary"
					#assert key1 in results, "{0} not in results dictionary".format(key1)
					#assert isinstance(key2, list), "Please provide a list of keys"

					try:
						if isinstance(results, dict) and (key1 in results) and isinstance(key2, list):
							list_len = len(key2)
							results_out = [[] for _ in range(list_len)]

							for iid, n_value in enumerate(key2):
								assert n_value in results[key1], "{0} not in subfield {1}".format(n_value, key1)

								if operation is None:
									results_array = results[key1][n_value]
								else:
									results_array = operation(results[key1][n_value])

								globals()['results_{0}'.format(n_value)][tuple(index)] = results_array

								results_out[iid] = globals()['results_{0}'.format(n_value)] # (globals()['results_{
							# 0}'.format(n_value)]).astype(float)
					except:
						print "Dataset {0} Not complying, skipping".format(params_label)

				elif key1 is not None and key2 is not None and key3 is not None:
					#assert isinstance(results, dict), "Results data must be a dictionary"
					#assert key1 in results, "{0} not in results dictionary".format(key1)
					#assert key2 in results[key1], "{0} not in subfield {1}".format(key2, key1)
					#assert isinstance(key3, list), "Please provide a list of keys"

					try:
						if isinstance(results, dict) and (key1 in results) and (key2 in results[key1]) and \
								isinstance(key3, list):
							list_len = len(key3)
							results_out = [[] for _ in range(list_len)]

							for iid, n_value in enumerate(key3):
								assert n_value in results[key1][key2], "{0} not in subfield {1}".format(n_value, key2)

								if operation is None:
									results_array = results[key1][key2][n_value]
								else:
									results_array = operation(results[key1][key2][n_value])

								globals()['results_{0}'.format(n_value)][tuple(index)] = results_array

								results_out[iid] = (globals()['results_{0}'.format(n_value)]).astype(float)

								#results_out[iid] = results_array.astype(float)
					except:
						print "Dataset {0} Not complying, skipping".format(params_label)
		else:
			results_array = []
			try:
				with open(data_path + 'Results_' + self.label, 'r') as fp:
					results = cPickle.load(fp)
				print "Loading Dataset {0}".format(self.label)
			except:
				raise IOError("Dataset {0} Not Found".format(self.label))

			if key1 is not None and key2 is None and key3 is None:
				assert isinstance(results, dict), "Results data must be a dictionary"
				assert isinstance(key1, list), "Please provide a list of keys"

				list_len = len(key1)
				results_out = [[] for _ in range(list_len)]

				for iid, n_value in enumerate(key1):
					assert n_value in results, "{0} not in results dictionary".format(n_value)

					if operation is None:
						results_array = results[n_value]
					else:
						results_array = operation(results[n_value])
					results_out[iid] = results_array.astype(float)

			elif key1 is not None and key2 is not None and key3 is None:
				assert isinstance(results, dict), "Results data must be a dictionary"
				assert key1 in results, "{0} not in results dictionary".format(key1)

				assert isinstance(key2, list), "Please provide a list of keys"

				list_len = len(key2)
				results_out = [[] for _ in range(list_len)]

				for iid, n_value in enumerate(key2):
					assert n_value in results[key1], "{0} not in subfield {1}".format(n_value, key1)

					if operation is None:
						results_array = results[key1][n_value]
					else:
						results_array = operation(results[key1][n_value])
					results_out[iid] = results_array.astype(float)

			elif key1 is not None and key2 is not None and key3 is not None:
				assert isinstance(results, dict), "Results data must be a dictionary"
				assert key1 in results, "{0} not in results dictionary".format(key1)
				assert key2 in results[key1], "{0} not in subfield {1}".format(key2, key1)

				assert isinstance(key3, list), "Please provide a list of keys"

				list_len = len(key3)
				results_out = [[] for _ in range(list_len)]

				for iid, n_value in enumerate(key3):
					assert n_value in results[key1][key2], "{0} not in subfield {1}".format(n_value, key2)

					if operation is None:
						results_array = results[key1][key2][n_value]
					else:
						results_array = operation(results[key1][key2][n_value])

					results_out[iid] = results_array.astype(float)
		#print results_out
		return results_out

	def compile_parameters_table(self):
		"""
		Use the first parameter set to generate the standard table...
		:return:
		"""
		template_file = self[0].report_pars.report_templates_path + 'StandardTable.tex'
		out_file = self[0].report_pars.report_path + self[0].report_pars.report_filename
		fields = self[0].report_pars.table_fields.as_dict()
		tmp = process_template(template_file, fields, save_to=out_file)

		return tmp, out_file

	@staticmethod
	def extract_result_from_array(results_array, field, operation=None):
		"""
		returns an array with a single numerical result...
		:param results_array:
		:return:s
		"""
		result = np.zeros_like(results_array)
		for index, d in np.ndenumerate(results_array):
			if d is None:
				result[index] = np.nan
			else:
				if operation is None:
					result[index] = d[field]
				else:
					result[index] = operation(d[field])
		return result
