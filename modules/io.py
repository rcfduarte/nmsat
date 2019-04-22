"""
========================================================================================================================
Input/Output Module
========================================================================================================================
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

========================================================================================================================
Copyright (C) 2018  Renato Duarte, Barna Zajzon

Uses parts from NeuroTools for which Copyright (C) 2008  Daniel Bruederle, Andrew Davison, Jens Kremkow
Laurent Perrinet, Michael Schmuker, Eilif Muller, Eric Mueller, Pierre Yger

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
# other imports
import os
import numpy as np
from collections import MutableMapping
import sys
import cPickle as pickle
import logging
import StringIO
# import psutil
import os
import resource
from time import time

# NMSAT imports
import parameters
from modules import check_dependency

has_h5 = check_dependency('h5py')
if has_h5:
    import h5py

# assume we run on a local machine, will be changed accordingly if otherwise (for logging purposes)
run_local = True


class Timer:
    """

    """
    def __init__(self):
        self.timers = {}

    def start(self, name):
        """
        Starts a new timer at the current timepoint. If the timer already exists, the start time will be overwritten,
        otherwise a new timer entry is created.
        :param name:
        :return:
        """
        self.timers[name] = {
            'start': time(),
            'stop': 0.,
            'duration': 0.
        }

    def restart(self, name):
        assert name in self.timers
        self.timers[name] = {
            'start': time(),
            'stop': 0.,
            'duration': 0.
        }

    def stop(self, name):
        if name in self.timers:
            self.timers[name]['stop'] = time()
            self.timers[name]['duration'] = time() - self.timers[name]['start']

    def accumulate(self, name):
        if name in self.timers:
            self.timers[name]['stop'] = time()
            self.timers[name]['duration'] += time() - self.timers[name]['start']

    def get_all_timers(self):
        return self.timers


def log_stats(cmd=None, flush_file=None, flush_path=None):
    """
    Dump some statistics (system, timers, etc) at the end of the run.

    :param cmd:
    :param flush_file:
    :param flush_path:
    :return:
    """
    logger.info('************************************************************')
    logger.info('************************ JOB STATS *************************')
    logger.info('************************************************************')

    if cmd:
        logger.info('Calling command: {}'.format(cmd))

    # system
    logger.info('')
    logger.info('System & Resources:')
    # logger.info('\t\tMemory usage (MB, psutil): {}'.format(memory_usage_psutil()))
    logger.info('\t\tPeak total memory usage (MB): {}'.format(memory_usage_resource()))

    # timing
    logger.info('')
    logger.info('Timers:')
    global log_timer
    for name, timer in log_timer.get_all_timers().iteritems():
        logger.info('\t\t{}: {} s'.format(name.capitalize(), timer['duration']))


# flush to a main file
    if flush_file and flush_path:
        global main_log
        with open(os.path.join(flush_path, flush_file), 'w') as f:
            f.write(main_log.getvalue())


def get_logger(name):
    """
    Initialize a new logger called `name`.
    :param name: Logger name
    :return: logging.Logger object
    """
    logging.basicConfig(format='[%(filename)s - %(levelname)s] %(message)s'.format(name), level=logging.INFO)
    logger_ = logging.getLogger(name)

    if name == 'main.py':
        file_handler = logging.FileHandler('main.log')
        logger_.addHandler(file_handler)

    return logger_


def update_log_handles(main=True, job_name=None, path=None):
    """
    Update log streams / files (including paths) according to how the program is run. If running from main.py,
    we buffer all output to `main_log` and will flush to file at the end. If this is a job (on a cluster), write
    directly to a file.

    :param main: if running from main.py, log to buffer and flush later
    :param job_name:
    :param path:
    :return:
    """
    if main:
        global main_log
        # logging.basicConfig(stream=main_log, level=logging.INFO)
        handler = logging.StreamHandler(main_log)
    else:
        handler = logging.FileHandler('{}/job_{}.log'.format(path, job_name))

    handler_format = logging.Formatter('[%(name)s - %(levelname)s] %(message)s')
    handler.setFormatter(handler_format)
    handler.setLevel(logging.INFO)

    for name, logger_ in logging.Logger.manager.loggerDict.iteritems():
        # whether to log into the main file (always, unless it's a submitted job on a cluster
        if main:
            if isinstance(logger_, logging.Logger):
                logger_.addHandler(handler)
        else:
            logger_.addHandler(handler)
            logger_.propagate = False
            global run_local
            run_local = False


def extract_data_fromfile(fname):
    """
    Extract raw data from a nest.device recording stored in file
    :param fname: filename or list of filenames
    """
    if parameters.isiterable(fname):
        data = None
        for f in fname:
            print("Reading data from file {0}".format(f))
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
                    data = get_data(f)
                else:
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
    if parameters.isiterable(fname):
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
        print("\nSetting storage paths...")
        save_label = parameter_set.label

        main_folder = parameter_set.kernel_pars.data_path + parameter_set.kernel_pars.data_prefix + '/'

        figures = main_folder + 'Figures/'
        inputs = main_folder + 'Inputs/'
        parameters = main_folder + 'Parameters/'
        results = main_folder + 'Results/'
        activity = main_folder + 'Activity/'
        others = main_folder + 'Other/'
        logs = main_folder + 'Logs/'

        dirs = {'main': main_folder, 'figures': figures, 'inputs': inputs, 'parameters': parameters, 'results':
            results, 'activity': activity, 'other': others, 'logs': logs}

        for d in dirs.values():
            try:
                os.makedirs(d)
            except OSError:
                pass

        dirs['label'] = save_label

        return dirs
    else:
        print("No data will be saved!")
        return {'label': False, 'figures': False, 'activity': False}


def set_project_paths(project):
    """
    Adds the project folders to the python path
    :param project: name given to the main project folder
    :return:
    """
    NMSAT_HOME = os.environ.get("NMSAT_HOME")
    if NMSAT_HOME is None:
        print("Please set the project root directory environment variable! (source configure.sh)\nExiting.")
        exit(0)
    project_home = NMSAT_HOME + "/projects/%s/" % project
    for add_path in os.walk(project_home):
        sys.path.append(add_path[0])


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

    def save(self):
        """
        Save the object defined in self.object with the method os self.user_file

        Note that you can add your own format for I/O of such NeuroTools objects
        """
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
        return pickle.dump(object, fileobj)


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


########################################################################################################################
class Standardh5File(object):
    """
    Handle h5py dictionaries
    """

    def __init__(self, filename):
        self.filename = filename
        assert(has_h5), "h5py required!"
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
        f = h5py.File(self.filename, 'w')
        for k in dictionary:
            f[k] = dictionary[k]
        f.close()


########################################################################################################################
class NestedDict(MutableMapping):
    """

    """
    def __init__(self, initial_value=None, root=True):
        super(self.__class__, self).__init__()
        self._val = {}
        if initial_value is not None:
            self._val.update(initial_value)
            self._found = False
        self._root = root

    def __getitem__(self, item):
        self._found = False

        def _look_deeper():
            result = tuple()
            for k, v in self._val.items():
                if isinstance(v, dict):
                    n = NestedDict(self[k], root=False)
                    if n[item]:
                        result += (n[item],)
                    self._found = self._found or n._found
            if self._root:
                if self._found:
                    self._found = False
                else:
                    raise KeyError(item)
            result = result[0] if len(result) == 1 else result
            return result

        def _process_list():
            if len(item) == 1:
                return self[item[0]]
            trunk, branches = item[0], item[1:]
            nd = NestedDict(self[trunk], root=False)
            return nd[branches] if len(branches) > 1 else nd[branches[0]]

        if isinstance(item, list):
            return _process_list()
        elif self.__isstring_containing_char(item, '/'):
            item = item.split('/')
            return _process_list()
        elif item in self._val:
            self._found = True
            return self._val.__getitem__(item)
        else:
            return _look_deeper()

    def __setitem__(self, branch_key, value):
        self._found = False

        def _process_list():
            branches, tip = branch_key[1:], branch_key[0]
            if self[tip]:
                if isinstance(self[tip], tuple):
                    if isinstance(self[branches], tuple):
                        raise KeyError('ambiguous keys={!r}'.format(branch_key))
                    else:
                        self[branches][tip] = value
                else:
                    self[tip] = value
            else:
                raise KeyError('no key found={!r}'.format(tip))

        def _look_deeper():
            nd = NestedDict(root=False)
            for k, v in self._val.items():
                if v and (isinstance(v, dict) or isinstance(v, NestedDict)):
                    nd._val = self._val[k]
                    nd[branch_key] = value
                    self._found = self._found or nd._found
            if self._root:
                if self._found:
                    self._found = False
                else:
                    self._val.__setitem__(branch_key, value)

        if isinstance(branch_key, list) or isinstance(branch_key, tuple):
            _process_list()
        elif self.__isstring_containing_char(branch_key, '/'):
            branch_key = branch_key.split('/')
            _process_list()
        elif branch_key in self._val:
            self._found = True
            self._val.__setitem__(branch_key, value)
        else:
            _look_deeper()

    def __delitem__(self, key):
        self._val.__delitem__(key)

    def __iter__(self):
        return self._val.__iter__()

    def __len__(self):
        return self._val.__len__()

    def __repr__(self):
        return __name__ + str(self._val)

    def __call__(self):
        return self._val

    def __contains__(self, item):
        return self._val.__contains__(item)

    def anchor(self, trunk, branch, value=None):
        nd = NestedDict(root=False)
        for k, v in self._val.items():
            if v and isinstance(v, dict):
                nd._val = self._val[k]
                nd.anchor(trunk, branch, value)
                self._found = self._found or nd._found
            if k == trunk:
                self._found = True
                if not isinstance(self._val[trunk], dict):
                    if self._val[trunk]:
                        raise ValueError('value of this key is not a logical False')
                    else:
                        self._val[trunk] = {}  # replace None, [], 0 and False to {}
                self._val[trunk][branch] = value
        if self._root:
            if self._found:
                self._found = False
            else:
                raise KeyError

    def setdefault(self, key, default=None):
        if isinstance(key, list) or isinstance(key, tuple):
            trunk, branches = key[0], key[1:]
            self._val.setdefault(trunk, {})
            if self._val[trunk]:
                pass
            else:
                self._val[trunk] = default

            nd = NestedDict(self[trunk], root=False)
            if len(branches) > 1:
                nd[branches] = default
            elif len(branches) == 1:
                nd._val[branches[0]] = default
            else:
                raise KeyError
        else:
            self._val.setdefault(key, default)


    @staticmethod
    def __isstring_containing_char(obj, char):
        if isinstance(obj, str):
            if char in obj:
                return True
        return False


def asciify(d, is_root=True, al=list, lvl=0):
    """

    :param d:
    :param is_root:
    :param al:
    :param lvl:
    :return:
    """
    if is_root:
        al = []
        lvl = 0
    if d:
        for k, v in d.items():
            il = []
            for i in range(lvl):
                if i == lvl - 1:
                    il.append('`-- ')
                else:
                    il.append('    ')
            il.append(k)
            al.append(il)
            if d[k]:
                lvl += 1
                al = asciify(d[k], is_root=False, al=al, lvl=lvl)
                lvl -= 1
    if is_root:
        empt_fill = '    '
        vert_fill = '|   '
        end__fill = '`-- '
        plus_fill = '+-- '
        replacement = set()
        for each_line in reversed(al):
            to_remove = []
            for index in replacement:
                if each_line[index] == empt_fill:
                    each_line[index] = vert_fill
                elif each_line[index] == end__fill:
                    each_line[index] = plus_fill
                else:
                    to_remove.append(index)
            while to_remove:
                replacement.discard(to_remove.pop())
            for i, e in enumerate(each_line):
                if e == end__fill:
                    replacement.add(i)
        sl = [''.join(x) for x in al]
        return sl
    else:
        return al


# def memory_usage_psutil():
#     process = psutil.Process(os.getpid())
#     mem = process.memory_info()[0] / float(2 ** 20)
#     return mem


def memory_usage_resource():
    self = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.
    children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024.
    return self + children

# some variables need to be initialized
log_timer = Timer()
logger = get_logger(__name__)
main_log = StringIO.StringIO()