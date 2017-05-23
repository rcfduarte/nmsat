This example consists of a balanced random network, driven by Poissonian external input, and is a commonly used setup to characterize population dynamics. The specific setup for this example is illustrated below:

![alt-text](/images/example3-01.png)

In this simple setup, one single generator, a `’poisson_generator’` is used and connects to all the neurons in the network (note that the generator draws independent realizations of Poisson spike trains at the specified rate for each target neuron). 

To run this example execute:

```python
python main.py -f projects/examples/parameters/noise_driven_dynamics.py -c population_noisedriven --extra plot=True display=True save=True
```

The output of this analysis, in the simplest case, consists of a raster plot displaying the population spiking activity, mean rates and (if an analog activity recorder is connected) an example of the input synaptic currents and membrane potential of a randomly chosen neuron. In addition, a summarized activity report is displayed.
The analysis parameters for this experiment allow the specification of which metrics are applied, as these vary in complexity and required computing time. This is accomplished with the parameter `’depth’`, as described in the comments. Note that the most complete characterization can be quite resource- and time-consuming.