# encoding: utf8
"""
Collection of convenience functions.

"""
import numpy as np
import os
import requests
import tarfile


def get_previous_version(version):
    """
    Retrieves the given version of the wrapper from github as a tar
    archive and extracts its contents to the current directory.

    """
    base_url = "https://github.com/INM-6/h5py_wrapper/archive/v"
    r = requests.get(''.join((base_url, version, ".tar.gz")))
    try:
        r.raise_for_status()
        fn = ''.join((os.path.join(os.getcwd(), version), '.tar.gz'))
        with open(fn, 'wb') as f:
            f.write(r.content)
        with tarfile.open(fn) as f:
            f.extract(''.join(('h5py_wrapper-', version, '/wrapper.py')))
            f.extract(''.join(('h5py_wrapper-', version, '/__init__.py')))
        os.rename('-'.join(('h5py_wrapper', version)),
                  '_'.join(('h5py_wrapper', version.replace('.', ''))))
        os.remove(fn)
    except requests.exceptions.HTTPError:
        raise ImportError("Requested release version does not exist.")


def accumulate(iterator):
    """
    Creates a generator to iterate over the accumulated
    values of the given iterator.
    """
    total = 0
    for item in iterator:
        yield total, item
        total += item


def convert_numpy_types_in_dict(d):
    """
    Convert all numpy datatypes to default datatypes in a dictionary (in place).
    """
    for key, value in d.items():
        if isinstance(value, dict):
            convert_numpy_types_in_dict(value)
        elif isinstance(value, (np.int)):
            d[key] = int(value)
        elif isinstance(value, (np.float)):
            d[key] = float(value)
        elif isinstance(value, (np.bool_)):
            d[key] = bool(value)
