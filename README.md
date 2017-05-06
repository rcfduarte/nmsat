# Neural Microcircuit Simulation and Analysis Toolkit (NMSAT) 

NMSAT is a python package that provides a set of tools to build, simulate and
 analyse neuronal microcircuit models with any degree of complexity, as well as to probe the circuits with 
 arbitrarily complex input stimuli / signals and to analyse the relevant functional aspects of single neuron and 
 population dynamics. It provides a high-level wrapper for PyNEST 
 (which is used as the core simulation engine). As such, the complexity of the microcircuits analysed and their 
 building blocks (neuron and synapse models, circuit topology and connectivity, etc), are determined by the models 
 available in NEST. The use of NEST allows efficient and highly scalable simulations of very large and 
 complex circuits, constrained only by the computational resources available.
The modular design allows the user to specify numerical experiments with varying degrees of complexity depending on concrete research objectives. The generality allows the same types of measurements and complex experiments to be performed on a variety of different circuits, which can be useful for benchmarking and comparison purposes. Additionally, the code was designed to allow an effortless migration across computing systems, i.e. the same simulations can be executed in a local machine, in a computer cluster or a supercomputer, with straightforward resource allocation. 

### Getting started

The code was developed primarily for personal use, as part of a PhD thesis due to the need to perform similar experiments on very diverse systems. The goal was to perform numerical experiments, covering a broad range of complexity, in a fully transparent and reproducible manner, while making efficient use of computing resources. The code is imperfect and under active use / development. Despite our best efforts, it is still prone to errors, particularly due to the strict specificities of parameters..

#### Dependencies / Requirements

* Python 2.7
* [NEST](http://www.nest-simulator.org/) version 2.8.0 or higher
* numpy version 1.7.0 or higher 
* scipy version 0.12.0 or higher
* scikit-learn version 0.18.0 or higher
* matplotlib version 1.2.0 or higher

Optional (add functionality):
* [PySpike](http://mariomulansky.github.io/PySpike/) version 0.5.1
* h5py version 2.2.1 or higher

#### Installing

The code is currently available only in this GitHub repository. To use it, simply download the source or fork and clone the 
repository. To configure your system, source the configuration file
```
git clone https://github.com/
```

#### Setting system defaults

The defaults folder contains the ...
Modify the defaults to suit your specifications. 

### Running an experiment
A numerical simulation/experiment is specified as a global parameter file, a complete script parsing the parameters and executing the experiment (primarily usefull for development/debugging) and a similar function that executes the same series of commands but can be executed directly from main as:
```
python main.py -f {parameter_file} -c {function_name} --extra {additional_parameters}
```
All the details and specificities of the experiment are determined in complex parameters files. The main computation function then parses the contents of these files and, following the specifications, assembles and runs the simulation, using the framework's modules to build, simulate, analyse and plot. 

### Storing and harvesting data
After an experiment is complete, all relevant information is stored in 

#### Examples

Complete examples can be found in the examples folder. 


### Authors and contributors

* **[Renato Duarte](https://github.com/rcfduarte)**
* **[Barna Zajzon](https://github.com/zbarni)**


### Help and support
[r.duarte@fz-juelich.de](r.duarte@fz-juelich.de)

### Citing us
If you find NMT useful and use it in your research, please cite it as [zenodo]

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
