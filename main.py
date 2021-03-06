#!/usr/bin/python
import os
import sys
import copy

cmdl_parse = copy.copy(sys.argv)

import matplotlib
matplotlib.use('Agg')
from argparse import ArgumentParser
from os import path, environ
import importlib

try:
    from modules.parameters import ParameterSpace
    from modules import io
except ImportError as e:
    NMSAT_HOME = environ.get("NMSAT_HOME")

    # exit if environment variable is not set
    if NMSAT_HOME is None:
        raise EnvironmentError("Environment variable with the project root directory not found! Please run "
                               "`source configure.sh`.")

    raise ImportError("Import dependency not met: {}. ".format(e))

logger = io.get_logger(__name__)
version = "0.2"


def run_experiment(params_file_full_path, computation_function="noise_driven_dynamics", cluster=None, **parameters):
    """
    Entry point, parses parameters and runs experiments locally or creates scripts to be run on a cluster.

    :param params_file_full_path: full path to parameter file
    :param computation_function: which experiment to run
    :param cluster: name of cluster template, e.g., Blaustein. Corresponding entry must be in defaults.paths!
    :param parameters: other CLI input parameters
    :return:
    """

    io.update_log_handles(main=True)

    try:
        # determine the project folder and add it to sys.path
        project_dir, _ = path.split(path.split(params_file_full_path)[0])
        sys.path.append(project_dir)
        experiment = importlib.import_module("computations." + computation_function)
    except Exception as err:
        logger.error("Could not find experiment `{0}`. Is it in the project's ./computations/ directory? "
                     "\nError: {1}".format(computation_function, str(err)))
        exit(-1)

    io.log_timer.start('parameters')
    if 'keep_all' in parameters.keys():
        pars = ParameterSpace(params_file_full_path, keep_all=parameters['keep_all'])
        parameters.pop('keep_all')
    else:
        pars = ParameterSpace(params_file_full_path)

    pars.update_run_parameters(cluster)
    pars.save(pars[0].kernel_pars.data_path+pars[0].kernel_pars.data_prefix+'_ParameterSpace.py')
    io.log_timer.stop('parameters')

    if pars[0].kernel_pars.system['local']:
        pars.run(experiment.run, **parameters)
    else:
        pars.run(experiment.run, project_dir, **parameters)

    io.log_stats(cmd=str.join(' ', cmdl_parse), flush_file='main.log',
                 flush_path=os.path.join(pars[0].kernel_pars.data_path+pars[0].kernel_pars.data_prefix, 'Logs'))

def print_version():
    print("NMSAT version {0}".format(version))
    print("Copyright (C) Renato Duarte and Barna Zajzon, 2019")


def print_welcome_message():
    print("""
   *** Neural Microcircuit Simulation and Analysis Toolkit ***

                          Version {0}

          This program is provided AS IS and comes with
          NO WARRANTY. See the file LICENSE for details.
""".format(version))


def create_parser():
    """
    Create command line parser with options.

    :return: ArgumentParser
    """
    parser_ = ArgumentParser(prog="main.py")
    parser_.add_argument("--version", action="version", version=print_version(),
                         help="print current version")
    parser_.add_argument("-f", dest="p_file", nargs=1, default=None, metavar="parameter_file",
                         help="absolute path to parameter file", required=True)
    parser_.add_argument("-c", dest="c_function", nargs=1, default=None, metavar="computation_function",
                         help="computation function to execute", required=True)
    parser_.add_argument("--cluster", dest="cluster", nargs=1, default=None, metavar="Blaustein",
                         help="name of cluster entry in default paths dictionary")
    parser_.add_argument("--extra", dest="extra", nargs='*', default=[], metavar="extra arguments",
                         help="extra arguments for the computation function")
    return parser_


if __name__ == "__main__":
    print_welcome_message()
    args = create_parser().parse_args(cmdl_parse[1:])  # avoids PyNEST sys.argv pollution

    d = dict([arg.split('=', 1) for arg in args.extra])
    for k, v in d.iteritems():
        if v == 'False':
            d.update({k: False})
        elif v == 'True':
            d.update({k: True})
        elif v == 'None':
            d.update({k: None})

    cluster_name = args.cluster[0] if args.cluster is not None else None
    run_experiment(args.p_file[0], computation_function=args.c_function[0], cluster=cluster_name, **d)