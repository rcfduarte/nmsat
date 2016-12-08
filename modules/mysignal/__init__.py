__author__ = 'duarte'
"""
====================================================================================
Signals Module
====================================================================================
(adapted and modified from NeuroTools.signals and stgen)

Collection of utilities and functions to create, use and manipulate signals (spike
data or analog data)

Classes:
---------------------
SpikeTrain            - object representing a spike train, for one cell. Useful for plots,
						calculations such as ISI, CV, mean rate(), ...
SpikeList             - object representing the activity of a population of neurons. Functions as a
						dictionary of SpikeTrain objects, with methods to compute firing rate,
						ISI, CV, cross-correlations, and so on.
AnalogSignal          - object representing an analog signal, with the corresponding data. ...
AnalogSignalList      - list of AnalogSignal objects, with all relevant methods (extensible)
VmList                - AnalogSignalList object specific for Vm traces
ConductanceList       - AnalogSignalList object used for conductance traces
CurrentList           - AnalogSignalList object used for current traces

Interval              - object to handle time intervals

PairsGenerator        -
AutoPairs             -
CustomPairs           -
RandomPairs           -

DistantDependentPairs -
StochasticGenerator   - object used to generate and handle stochastic input data
----------------------
"""
