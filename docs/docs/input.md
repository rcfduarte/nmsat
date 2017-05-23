The `input_architect.py` module handles everything related to input stimuli and signals. It is
designed to encompass a large variety of input stimuli / signals, patterned according to complex
specifications. However, at the moment, due to recent changes, not all variants have been tested.
None of the following components of the input construction process is strictly necessary. For example,
a signal can be generated using the signal_pars, without the need for a StimulusSet to be specified.
The main classes are:

* `StimulusSet` – hold and manipulate all the data pertaining to the input stimuli, labels, and
corresponding time series, can be divided into data sets (transient, train and test sets)
* `InputSignalSet` – container for all the relevant signals, divided into data sets (transient, train
and test sets)
* `InputSignal` – Generate and store AnalogSignal object referring to the structured input
signal u(t)
* `InputNoise` – Generate and store AnalogSignal object referring to the noise signal
