# Neural Microcircuit Simulation and Analysis Toolkit (NMSAT) 

NMSAT is a python package that provides a set of tools to build, simulate and analyse neuronal
microcircuit models with any degree of complexity, as well as to probe the circuits with arbitrarily
complex input stimuli / signals and to analyse the relevant functional aspects of single neuron,
population and network dynamics. It provides a high-level wrapper for PyNEST (which is used as
the core simulation engine). As such, the complexity of the microcircuits analysed and their building
blocks (neuron and synapse models, circuit topology and connectivity, etc.), are determined by the
models available in NEST. The use of NEST allows efficient and highly scalable simulations of very
large and complex circuits, constrained only by the computational resources available to the user.


The modular design allows the user to specify numerical experiments with varying degrees of
complexity depending on concrete research objectives. The generality of some of these experiments
allows the same types of measurements to be performed on a variety of different circuits, which can
be useful for benchmarking and comparison purposes. Additionally, the code was designed to allow
an effortless migration across computing systems, i.e. the same simulations can be executed in a
local machine, in a computer cluster or a supercomputer, with straightforward resource allocation
(see kernel parameters). 

The code is licensed under GPLv3 and available on [GitHub](https://github.com/rcfduarte/nmsat/).


### Disclaimer

The code was developed primarily for personal use, as part of a PhD thesis due to the need to perform
similar experiments and analyses on very diverse systems. The goal was to use the same code to
run diverse numerical experiments, covering a broad range of complexity, in a fully transparent and
reproducible manner, while making efficient use of computing resources. Due to the inherent time
pressures of a PhD project and the very broad scope, the code is imperfect and under active use /
development. Despite our best efforts, it is prone to errors and often difficult to understand, particularly
due to the strict specificities on the structure of the parameters dictionaries and how they are used.

### Getting started

For a detailed description of the framework and to make the most out of it, please read the 
[documentation](https://zbarni.github.io/nmsat/). 

#### Dependencies

* **Python** 2.7
* [**NEST**](http://www.nest-simulator.org/) version 2.8.0 or higher
* **numpy** version 1.7.0 or higher 
* **scipy** version 0.12.0 or higher
* **scikit-learn** version 0.18.0 or higher
* **matplotlib** version 1.2.0 or higher

Optional (for additional functionality):
* **PySpike** version 0.5.1
* **h5py** version 2.2.1 or higher
* **mayavi** 
* **networkx**


#### Installing

The code is currently available only in this GitHub repository. To use it, simply download the source or fork and clone the 
repository. To configure your system, source the configuration file

```bash
source /{path}/nmsat/configure.sh
```

This last step requires the user to manually specify all the paths for his system, by editing the paths dictionary in 
`/defaults/paths.py`, as:

```python
paths = {
  'system_label': {
  'data_path':            NMSAT_HOME + '/data/',
  'jdf_template':         NMSAT_HOME + '/defaults/cluster_templates/Blaustein_jdf.sh',
  'matplotlib_rc':        NMSAT_HOME + '/defaults/matplotlib_rc',
  'remote_directory':     NMSAT_HOME + '/export/',
  'queueing_system':      'slurm'}
}
```

The `system_label` specifies the name of the system. If running simulations on a local machine, the name must be set as 'local' (which is the default), otherwise, it can be any arbitrary name, as long as it is used consistently throughout (see examples). The remaining entries in this dictionary refer to:


* `data_path` - specify where to store the output data generated by an experiment
* `jdf_template` - path to a system-specific job description file (see example in /defaults/cluster_templates); if running locally set to None
* `matplotlibrc` - in case the user wants to customize matplotlib\footnote{\url{http://matplotlib.org/users/customizing.html}}
* `remote_directory` - folder where the job submission files will be written (only applicable if not running locally, but must be specified anyway)
* `queueing_system` - type of job schedulling system used (current options include 'slurm' and 'sge')..

### Running an experiment
A numerical experiment in this framework consists of 2 or 3 main files (see examples):


* **parameters_file** - specifying all the complex parameter sets and dictionaries required to set up
the experiment.
* **experiment_script** - mostly used during development, for testing and debugging purposes.
These scripts parse the parameters_file and run the complete experiment
* **computation_file** - after the development and testing phase is completed, the experiment can
be copied to a computation_file, which can be used from the main... (*)


These files should be stored within a `project` folder, and in the `parameters`, `scripts` and `computations` 
folders, respectively.

You can run experiments both **locally** and on a **remote cluster**. To run an experiment on your local computer,  
just go to the main nmsat directory and execute the experiment as:

```python
python main.py -f {parameters_file} -c {computation_function} --extra {extra_parameters}
```

where `parameters_file` refers to the (full or relative) path to the parameters file for the experiment,
`computation_function` is the name of the computation function to be executed on that parameter
set (must match the name of a file in the project’s ’computations’ folder) and `extra_parameters` are
parameter=value pairs for different, extra parameters (specific to each computation).

To run experiments on a cluster, please check out the [documentation](https://zbarni.github.io/network_simulation_testbed/standard-use-case/#cluster).

<!--A numerical simulation/experiment is specified as a global parameter file, a complete script parsing the parameters and executing the experiment (primarily usefull for development/debugging) and a similar function that executes the same series of commands but can be executed directly from main as:-->
<!--```-->
<!--python main.py -f {parameter_file} -c {function_name} --extra {additional_parameters}-->
<!--```-->
<!--All the details and specificities of the experiment are determined in complex parameters files. The main computation function then parses the contents of these files and, following the specifications, assembles and runs the simulation, using the framework's modules to build, simulate, analyse and plot. -->

### Simulation output

After a simulation is completed, all the relevant output data is stored in the pre-specified data_path,
within a folder named after the project label. The output data structure is organized as follows:

```
data
├── experiment_label
│   ├── Figures
│   ├── Results
│   ├── Parameters
│   ├── Activity
│   ├── Input
│   ├── Output
```

### Analysing and plotting
Analysis and plotting can be (and usually is) done within the main computation, so as to extract and
store only the information that is relevant for the specific experiment. Multiple, standard analysis and
plotting routines are implemented for various complex experiments, with specific objectives. Naturally,
this is highly mutable as new experiments always require specific analyses methods.


Alternatively, as all the relevant data is stored in the results dictionaries, you can read it and
process it offline, applying the same or novel visualization routines.


### Harvesting stored results
The Results folder stores all the simulation results for the given experiment, as pickle dictionaries.
Within each project, as mentioned earlier, a read_data folder should be included, which contains files
to parse and extract the stored results (see examples).


```python
project      = 'project_name'
data_path    = '/path/label/'
data_label   = 'example1'
results_path = data_path + data_label + '/Results/'

# set defaults and paths
set_project_paths(project)
set_global_rcParams(paths['local']['matplotlib_rc'])

# re-create ParameterSpace
pars_file = data_path + data_label + '_ParameterSpace.py'
pars = ParameterSpace(pars_file)

# print the full nested structure of the results dictionaries
pars.print_stored_keys(results_path)

# harvest a specific result based on the results structure
data_array = pars.harvest(results_path, key_set='dict_key1/dict_key1.1/dict_key1.1.1')
```

### Examples

Complete examples can be found in the `nmsat/projects/examples` folder. Currently there are 4 examples that you 
can try out:

* Single neuron fI curve
* Single neuron with patterned synaptic input
* Balanced random network 
* Stimulus processing

For more details about each example read the **Examples** section in the [documentation](/).

### Authors and contributors

* **[Renato Duarte](https://github.com/rcfduarte)**
* **[Barna Zajzon](https://github.com/zbarni)**


### Help and support
[r.duarte@fz-juelich.de](r.duarte@fz-juelich.de)

### Citing us
If you find NMSAT useful and use it in your research, please cite it as [zenodo]

### License 

Copyright (C) 2017  Renato Duarte  

Copyright (C) 2008  Daniel Bruederle, Andrew Davison, Jens Kremkow
Laurent Perrinet, Michael Schmuker, Eilif Muller, Eric Mueller, Pierre Yger

Neural Mircocircuit Testbed is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

The GNU General Public License does not permit this software to be
redistributed in proprietary programs.

See the [LICENSE](LICENSE) file for details.


### Acknowledgments

* funding...
