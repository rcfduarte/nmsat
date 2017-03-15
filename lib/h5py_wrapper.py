# -*- coding: utf-8 -*-
"""
Core functionality
"""

import ast
import collections
import h5py
import numpy as np
import os
import re
from subprocess import call
import warnings

from . import h5py_lib

# deprecation warnings are printed to sys.stdout
warnings.simplefilter('default', category=DeprecationWarning)

# check whether quantities is available
try:
    import quantities as pq
    quantities_found = True
except ImportError:
    quantities_found = False

# make sure correct h5py version is available
if int(re.sub('\.', '', h5py.version.version)) < 231:
    raise ImportError("Using h5py version {version}. Version must "
                      "be >= 2.3.1".format(version=h5py.version.version))


def save(filename, d, write_mode='a', overwrite_dataset=False,
         resize=False, dict_label='', compression=None):
    """
    Save a dictionary to an hdf5 file.

    Parameters
    ----------
    filename : string
        The file name of the hdf5 file.
    d : dict
        The dictionary to be stored.
    write_mode : {'a', 'w'}, optional
        Analog to normal file handling in python. Defaults to 'a'.
    overwrite_dataset : bool, optional
        Whether datasets should be overwritten if already existing.
        Defaults to False.
    resize : bool, optional
        If True, the hdf5 file is resized after writing all data,
        may reduce file size. Uses h5repack (see
        https://www.hdfgroup.org/HDF5/doc/RM/Tools.html#Tools-Repack).
        Caution: slows down writing. Defaults to False.
    dict_label : string, optional
        If not empty, the dictionary is stored under the given path in the hdf5
        file, with levels separated by '/'.
        For instance, dict_label='test/trial/spiketrains'. Defaults to ''.
    compression : {'gzip', 'szip','lzf', 0,...,10}, optional
       Compression strategy to reduce file size. An integer >0, <=10 leads to
       usage of gzip,indicating the level of compression. 'gzip' is recommended.
       See http://docs.h5py.org/en/latest/high/dataset.html for details.
       Caution: This slows down writing and loading of data.
       Attention: Will be ignored for scalar data.

    Returns
    -------
    None

    Examples
    --------
    >>> d = {}
    >>> d['a'] = {'a1': [1, 2, 3], 'a2': 4., 'a3': {'a31': 'Test'}}
    >>> d['b'] = 'string'
    >>> import h5py_wrapper as h5w
    >>> h5w.save('example.h5', d)
    """
    try:
        f = h5py.File(filename, write_mode)
    except IOError:
        raise IOError("unable to create {filename} (File "
                      "accessability: Unable to open "
                      "file)".format(filename=filename))
    else:
        try:
            if dict_label:
                base = f.require_group(dict_label)
                _dict_to_h5(f, d, overwrite_dataset, parent_group=base,
                            compression=compression)
            else:
                _dict_to_h5(f, d, overwrite_dataset, compression=compression)
        finally:  # make sure file is closed even if an exception is raised
            fname = f.filename
            f.close()
            if overwrite_dataset is True and resize is True:
                call(['h5repack', '-i', fname, '-o', fname + '_repack'])
                call(['mv', fname + '_repack', fname])


def load(filename, path='', lazy=False):
    """
    Loads a dictionary from an hdf5 file.

    Parameters
    ----------
    filename : string
        The file name of the hdf5 file.
    path : string, optional
        If not empty, specifies a path to access deeper levels in the hdf5 file.
    lazy : boolean, optional
        If True, only keys from all levels of the dictionary are loaded
        with values. Defaults to False.

    Returns
    -------
    dictionary : dict
        Dictionary from the hdf5 file.

    Examples
    --------
    >>> d = {}
    >>> d['a'] = {'a1': [1, 2, 3], 'a2': 4., 'a3': {'a31': 'Test'}}
    >>> d['b'] = 'string'
    >>> import h5py_wrapper as h5w
    >>> h5w.save('example_load.h5', d, overwrite_dataset=True)
    >>> h5w.load('example_load.h5')
    {u'a': {u'a1': array([1, 2, 3]), u'a3': {u'a31': 'Test'}, u'a2': 4.0}, u'b': 'string'}

    """
    try:
        f = h5py.File(filename, 'r')
    except IOError:
        raise IOError("unable to open {filename} (File accessability: "
                      "Unable to open file)".format(filename=filename))
    else:
        try:
            if not path:
                _, d = _dict_from_h5(f, lazy=lazy)
            else:
                try:
                    _, d = _dict_from_h5(f[path], lazy=lazy)
                except KeyError:
                    raise KeyError("unable to open {filename}/{path} "
                                   "(Key accessability: Unable to access "
                                   "key)".format(filename=filename, path=path))
        finally:
            f.close()
    return d

# ______________________________________________________________________________
# Auxiliary functions


def _dict_to_h5(f, d, overwrite_dataset, compression=None, parent_group=None):
    """
    Recursively adds the dictionary to the hdf5 file f.
    """
    if parent_group is None:
        parent_group = f.parent
    for key, value in d.items():
        if isinstance(value, collections.MutableMapping):
            group_name = os.path.join(parent_group.name, str(key))
            group = f.require_group(group_name)
            _dict_to_h5(f, value, overwrite_dataset, parent_group=group,
                        compression=compression)

            # explicitly store type of key
            group.attrs['_key_type'] = type(key).__name__
        else:
            if str(key) not in parent_group.keys():
                _create_dataset(parent_group, key, value,
                                compression=compression)
            else:
                if overwrite_dataset is True:
                    del parent_group[str(key)]
                    _create_dataset(parent_group, key, value,
                                    compression=compression)
                else:
                    raise KeyError("Dataset {key} already "
                                   "exists.".format(key=os.path.join(
                                       parent_group.name, key)))


def _create_dataset(parent_group, key, value, compression=None):
    """
    Creates the dataset in parent_group.
    """
    if value is None:  # h5py cannot store NoneType.
        dataset = parent_group.create_dataset(
            str(key), data='None', compression=compression)
    elif isinstance(value, (list, np.ndarray, tuple)):
        if np.array(value).dtype.name == 'object':
            # We store 2d arrays with unequal dimensions by reducing
            # it to a 1d array and additionally storing the original shape.
            # This does not work for more than two dimensions.
            if len(np.shape(value)) > 1:
                raise ValueError("Dataset {key} has an unsupported "
                                 "format.".format(key=os.path.join(
                                     parent_group.name, key)))
            else:
                oldshape = np.array([len(x) for x in value])
                value_types = [type(x).__name__ for x in value]
                data_reshaped = np.hstack(value)
                dataset = parent_group.create_dataset(
                    str(key), data=data_reshaped, compression=compression)
                dataset.attrs['oldshape'] = oldshape
                dataset.attrs['custom_shape'] = True
                dataset.attrs['custom_value_types'] = value_types
        elif quantities_found and isinstance(value, pq.Quantity):
            dataset = parent_group.create_dataset(str(key), data=value)
            dataset.attrs['_unit'] = value.dimensionality.string
        # To handle unicode strings
        # Assumes that all entries of value are of the same data type
        elif len(value) > 0 and isinstance(value[0], unicode):
            value = [ii.encode('utf-8') for ii in value]
            dataset = parent_group.create_dataset(
                str(key), data=value, compression=compression)
        else:
            dataset = parent_group.create_dataset(
                str(key), data=value, compression=compression)
    # ignore compression argument for scalar datasets
    elif not isinstance(value, collections.Iterable):
        dataset = parent_group.create_dataset(str(key), data=value)
    elif isinstance(value, unicode):
        dataset = parent_group.create_dataset(
            str(key), data=value.encode('utf-8'), compression=compression)
    else:
        dataset = parent_group.create_dataset(
            str(key), data=value, compression=compression)

    # explicitly store type of key and value
    dataset.attrs['_key_type'] = type(key).__name__
    dataset.attrs['_value_type'] = type(value).__name__


def _dict_from_h5(f, lazy=False):
    """
    Recursively loads the dictionary from the hdf5 file f.
    Converts all datasets to numpy types.
    """
    name = _evaluate_key(f)
    if h5py.h5i.get_type(f.id) == 5:  # check if f is a dataset
        return name, _load_dataset(f, lazy)
    else:
        d = {}
        for obj in f.values():
            sub_name, sub_d = _dict_from_h5(obj, lazy=lazy)
            d[sub_name] = sub_d
        return name, d


def _load_dataset(f, lazy=False):
    """
    Loads the dataset of group f and returns its name and value.
    If lazy is True, it returns None as value.
    """
    if lazy:
        return None
    else:
        try:
            value_type = f.attrs['_value_type']
        except KeyError:
            raise KeyError("No value type stored. This file has "
                           "probably been created with a previous release version. "
                           "Please use the conversion script to convert your "
                           "file.")
        if value_type == 'NoneType':
            return None
        else:
            if (len(f.attrs.keys()) > 0 and
                    'custom_shape' in f.attrs.keys()):
                return _load_custom_shape(f)
            elif '_unit' in f.attrs.keys():
                return _cast_value_type(f.value, value_type,
                                        unit=f.attrs['_unit'])
            else:
                return _cast_value_type(f.value, value_type)


def _evaluate_key(f):
    """
    Evaluate the key of f and handle non-string data types.
    """
    name = os.path.basename(f.name)  # to return only name of this level
    if ('_key_type' in f.attrs.keys() and
            f.attrs['_key_type'] not in ['str', 'unicode', 'string_']):
        name = ast.literal_eval(name)
    return name


def _load_custom_shape(f):
    """
    Reshape array with unequal dimensions into original shape.
    """
    data_reshaped = []
    value = f.value
    for (j, i), value_type in zip(h5py_lib.accumulate(f.attrs['oldshape']),
								  f.attrs['custom_value_types']):
        cast_value = _cast_value_type(value[j:j + i],
                                      value_type)
        data_reshaped.append(cast_value)
    return eval(valuetype_dict[value_type])(data_reshaped)


def _cast_value_type(value, value_type, unit=None):
    """
    Casts value into the correct type defined in attrs.
    """
    if value_type in valuetype_dict:
        if unit:
            if quantities_found:
                value = eval(valuetype_dict[value_type])(value, unit)
            else:
                raise ImportError("Could not find quantities package, "
                                  "please install the package and "
                                  "reload the wrapper.")
        else:
            if value_type in ['list', 'tuple']:
                # ensures that all dimensions of the array are converted to the correct value type
                value = _array_to_type(value, value_type)
            else:
                value = eval(valuetype_dict[value_type])(value)
        return value
    else:
        raise NotImplementedError("Unsupported data type: "
                                  "{value_type}.".format(value_type=value_type))


def _array_to_type(value, value_type):
    """
    Casts members of arrays to the specified type recursively.
    """
    if len(value) > 0 and isinstance(value[0], np.ndarray):
        return eval(valuetype_dict[value_type])(_array_to_type(i, value_type) for i in value)
    else:
        return eval(valuetype_dict[value_type])(value)

# Look-up table with supported datatypes
valuetype_dict = {'tuple': 'tuple',
                  'ndarray': 'np.array',
                  'list': 'list',
                  'float': 'float',
                  'int': 'int',
                  'str': 'str',
                  'bool': 'bool',
                  'int64': 'np.int64',
                  'Quantity': 'pq.Quantity'}


# Deprecated names for load and save routine
def add_to_h5(filename, d, write_mode='a', overwrite_dataset=False,
              resize=False, dict_label='', compression=None):
    warnings.warn("Deprecated function name. This function "
                  "will be removed in the next release. Please use save() instead.",
                  DeprecationWarning)
    save(filename, d, write_mode=write_mode, overwrite_dataset=overwrite_dataset,
         resize=resize, dict_label=dict_label, compression=compression)


def load_h5(filename, path='', lazy=False):
    warnings.warn("Deprecated function name. This function "
                  "will be removed in the next release. Please use load() instead.",
                  DeprecationWarning)
    return load(filename, path=path, lazy=lazy)
