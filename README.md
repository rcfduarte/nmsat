# Neural Microcircuit Testbed 

NMT is a python package that provides a set of tools to build, simulate and
 analyse neuronal microcircuit models with any degree of complexity, as well as to probe the circuits with 
 arbitrarily complex input stimuli / signals and to analyse the relevant functional aspects of single neuron and 
 population dynamics. It provides a high-level wrapper for PyNEST 
 (which is used as the core simulation engine). As such, the complexity of the microcircuits analysed and their 
 building blocks (neuron and synapse models, circuit topology and connectivity, etc), are determined by the models 
 available in NEST. Additionally, the use of NEST allows efficient and highly scalable simulations of very large and 
 complex circuits, constrained only by the computational resources available.
The modular design allows the user to specify numerical experiments with varying degrees of complexity and 
focusing on different aspects.

### Getting started

The code was developed as part of a PhD thesis and its focus was primarily on research quality, validity
and reproducibility, as well as on conducting the main research required. As such, it is 
imperfect and prone to errors...

#### Dependencies

* Python 2.7
* [NEST](http://www.nest-simulator.org/) version 2.8.0 or higher - Core simulation engine
* numpy version 1.7.0 or higher 
* scipy version 0.12.0 or higher
* scikit-learn version 0.18.0 or higher
* matplotlib version 1.2.0

Optional (add functionality):
* [PySpike](http://mariomulansky.github.io/PySpike/) version 0.5.1
* h5py version 2.2.1 or higher
* mayavi 

#### Installing

The code is currently available only in the GitHub repository. To use it, simply download the source, e.g. clone the 
repository, source the configuration file...
```
git clone https://github.com/...
```

### Running an experiment

```
python main.py -f {parameter_file} -c {computation function} {additional_parameters}
```

### Authors and contributors

* **[Renato Duarte](https://github.com/rcfduarte)** - 
* **[Barna Zajzon](https://github.com/zbarni)** - 


### Help and support
[r.duarte@fz-juelich.de](r.duarte@fz-juelich.de)

### Citing us
If you use NMT in your research, please cite it as [zenodo]

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

* Hat tip to anyone who's code was used
