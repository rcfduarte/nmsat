The EncodingLayer class wraps all the encoding process. It’s main constituents are:

* `Generator` – consists of a NEST generator device (e.g. spike_generator, step_current_generator,
inh_poisson_generator). Note that in the parameter specifications, N refers to the num-
ber of unique devices in the setup not the number of devices of a certain type (this is later
acquired by the dimensionality of the input signal to be encoded - 1 unique generator device
per input dimension)
* `Encoder` – consists of a Population object containing a layer of spiking neurons or some other
mechanism that converts the continuous input into spike trains to be fed to the network