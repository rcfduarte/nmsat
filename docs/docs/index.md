# 

<!--# Neural Microcircuit Simulation and Analysis Toolkit (NMSAT)-->

![alt-text](/images/NMSAT.png)


## What is it?

NMSAT is a Python package that provides a set of tools to build, simulate and analyse neuronal microcircuit models with any degree of complexity, as well as to probe the circuits with arbitrarily complex input stimuli / signals and to analyse the relevant functional aspects of single neuron, population and network dynamics. It provides a high-level wrapper for PyNEST (which is used as
the core simulation engine). As such, the complexity of the microcircuits analysed and their building blocks (neuron and synapse models, circuit topology and connectivity, etc.), are determined by the models available in NEST. The use of NEST allows efficient and highly scalable simulations of very
large and complex circuits, constrained only by the computational resources available to the user.
The modular design allows the user to specify numerical experiments with varying degrees of
complexity depending on concrete research objectives. The generality of some of these experiments
allows the same types of measurements to be performed on a variety of different circuits, which can
be useful for benchmarking and comparison purposes. Additionally, the code was designed to allow
an effortless migration across computing systems, i.e. the same simulations can be executed in a
local machine, in a computer cluster or a supercomputer, with straightforward resource allocation
(see [kernel parameters](/parameters/#kernel)).


The code is licensed under GPLv3 and available on [GitHub](https://github.com/rcfduarte/nmsat).


## Disclaimer

The code was developed primarily for personal use, as part of a PhD project due to the need to perform similar experiments and analyses on very diverse systems. The goal was to use the same code to
run different, but specialized, numerical experiments, in a fully transparent and reproducible manner, while making efficient use of computing resources. Due to the inherent time pressures of a PhD project and the broad scope, the code and documentation are under active use / development. 


## Authors and contributors

* **[Renato Duarte](https://github.com/rcfduarte)**
* **[Barna Zajzon](https://github.com/zbarni)**


### Citing us
If you find NMSAT helpful and use it in your research, please cite it as [zenodo]

## Acknowledgements
This work was done in the **[Functional Neural Circuits](http://www.fz-juelich.de/inm/inm-6/EN/Forschung/Morrison/artikel.html)** group, at the Institute for Neuroscience and Medicine (INM-6) and Institute for Advanced Simulation (IAS-6), Jülich Research Centre, Jülich, Germany. 
We would like to thank Professor Abigail Morrison for her continued patience, advice and support and the **[Neurobiology of Language](http://www.mpi.nl/departments/neurobiology-of-language)** group, at the Max-Planck for Psycholinguistics, for valuable discussions and contributions.

We acknowledge partial support by the Erasmus Mundus Joint Doctoral Program EuroSPIN, the German Ministry for Education and Research (Bundesministerium für Bildung und Forschung) BMBF Grant 01GQ0420 to BCCN Freiburg, the Helmholtz Alliance on Systems Biology (Germany), the Initiative and Networking Fund of the Helmholtz Association, the Helmholtz Portfolio theme ‘Supercomputing and Modeling for the Human Brain’.
We additionally acknowledge the computing time granted by the JARA-HPC Vergabegremium on the supercomputer **[JURECA](https://jlsrf.org/index.php/lsf/article/view/121/pdf)** at Forschungszentrum Jülich, used during development.
