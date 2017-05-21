**NOTE:** to run this example, you need the NEST version containing the inhomogeneous Poisson generator (currently 
available as a [Pull Request here](https://github.com/nest/nest-simulator/pull/671))

This example illustrates one way how input signals can be transformed into spike patterns in the Encoding Layer to 
simulate synaptic bombardment of neuronal populations (here only a single neuron).  

![alt-text](/images/example2-01.png)

The set of input signals is composed of two independent noise channels that will later act as excitatory (blue) and 
inhibitory (red) input, modeled as independent Ornstein-Uhlenbeck 
processes. Each of the signals drives one inhomogeneous Poisson generator in the Encoding Layer: at given intervals, 
these input signals are sampled and the values are used to set / update the firing rate of the generators. This way, 
the generators transform the input signals into Poisson spike trains that are passed to the encoders. There are two encoders 
in this, consisting of a number of parrot neurons, which means they only forward the incoming spikes to the population they are connected to.

Note that, behind the scenes, there are as many NEST spike generators created as the dimensionality the Encoders, so 
each parrot neuron will receive and independent Poisson spike train as input.

To run this example execute:

```python
python main.py -f ./projects/examples/parameters/single_neuron_patterned_synaptic_input.py -c single_neuron_pattern_input --extra plot=True display=False save=True
```

After running the example you should see 3 figures: the input signal of the two channels and the main output reporting some statistics for the single neuron:
histogram of the inter-spike intervals (ISI), firing rate, coefficient of variation (CV_ISI), Fano Factor, and the time 
course of the synaptic current (I_syn) and membrane potential (V_m).  
 

![alt-text](/images/example2-02.png)
