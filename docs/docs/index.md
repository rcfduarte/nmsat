# Neural Microcircuit Simulation and Analysis Toolkit (NMSAT)

## What is it?

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
(see [kernel parameters](/parameters/#kernel)).


The code is licensed under GPLv3 and available on [GitHub](https://github.com/zbarni/network_simulation_testbed).


## Getting started

The code was developed primarily for personal use, as part of a PhD thesis due to the need to perform
similar experiments and analyses on very diverse systems. The goal was to use the same code to
run diverse numerical experiments, covering a broad range of complexity, in a fully transparent and
reproducible manner, while making efficient use of computing resources. Due to the inherent time
pressures of a PhD project and the very broad scope, the code is imperfect and under active use /
development. Despite our best efforts, it is prone to errors and often difficult to understand, particularly
due to the strict specificities on the structure of the parameters dictionaries and how they are used.


## About the authors

Renato Duarte ...


