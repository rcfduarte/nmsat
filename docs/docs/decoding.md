### Specifying state variables
In the current implementation, one can simultaneously record and readout all relevant variables, for
comparison purposes (as long as they are recordable). To do so, we just specify which populations
to read, and which variable to read from. This is done in the dictionary that is passed as argument to the decoding defaults (see below), namely the key variables `decoded_population` and
`state_variable`.

These two variables need to be lists of equal length. A given decoded_population can be
specified as a sub-list, which means that we want to extract 1 state matrix from the combination of
the two referred populations (which are merged for this purpose). See example below.

### State sampling methods
The population responses to the stimuli can also be extracted in various different ways, by taking
samples of the state variable under consideration. In the current implementation this is simply done with the decoder variable global_sampling_times:

* One-sample at the stimulus offset (default, t* ):
    * `global_sampling_times = None`
    * one population state vector per stimulus
* Multiple samples taken at specific times (all stimuli must have the same duration):
    * global_sampling_times is given as a list of np.array of times (from stimulus onset) - length N*
    * one population state vector per sampling time
    * constructs N* state matrices that will be independently readout
* Sub-sampling responses at a fixed rate:
    * global_sampling_times = 1/10 is given as a fraction (one sample every 10 steps, step size being the input resolution)
    * constructs one long state matrix corresponding to the full response (in the limit, if global_sampling_times = 1, this corresponds to the entire activity history, sampled at the input resolution)
    * the target is also downsampled to the same rate, implementing an approximation to a continuous readout.

These different methods are used for different purposes, to assess different features of the responses. See Examples. Note that in the current version only the first sampling method is fully implemented and tested.

### Readouts
Currently available algorithms are:

* Direct pseudo-inverse - **'pinv'**
* Ridge Regression - **'ridge'**
* Logistic Regression - **'logistic'**
* Perceptron - **'perceptron'**
* Linear SVM - **'svm-linear'**
* Non-linear SVM (radial basis function) - **'svm-rbf'**
* Elastic Net - **'elastic'**
* Bayesian Ridge - **'bayesian_ridge'**


The decoding parameters should abide to the following structure:

```python
decoders = dict(
	decoded_population=['E', 'E', ['E', 'I'], ['E', 'I'], 'I', 'I'],
	state_variable=['V_m', 'spikes', 'V_m', 'spikes', 'V_m', 'spikes'],
	filter_time=filter_tau,
	readouts=readout_labels,
	readout_algorithms=['ridge', 'ridge', 'ridge', 'ridge', 'ridge', 'ridge'],
	global_sampling_times=state_sampling,
)
decoding_pars = set_decoding_defaults(default_set=1, 
                                      output_resolution=1., 
                                      to_memory=True, 
                                      kernel_pars=kernel_pars,
		                              **decoders)
```

