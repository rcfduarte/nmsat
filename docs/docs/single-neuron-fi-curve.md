In this simple example, we perform a very simple and common experiment: determining the responses
of a single neuron to injected current at different amplitudes. The specific setup is illustrated below

![alt-text](/images/example1-01.png)

In this simple setup, one single generator, a `’step_current_generator’` is used and a single neuron composes the whole network.

To run this example execute:

```python
python main.py -f projects/examples/parameters/single_neuron_fI.py -c single_neuron_fIcurve --extra plot=True display=True save=True
```

The output should display the neuron's fI curve, as well as a sample of its response (to the first current amplitude that drives the neuron to fire). There are two additional plots that refer to an adaptation index, to determine the degree of irregularity in the inter-spike intervals (primarily for neurons that display an adaptive firing pattern). 