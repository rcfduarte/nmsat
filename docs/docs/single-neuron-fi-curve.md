In this simple example, we perform a very simple and common experiment: determining the responses
of a single neuron to injected current at different amplitudes. The specific setup is illustrated below

![alt-text](/images/example1-01.png)

In this simple setup, one single generator, a `’step_current_generator’` is used and a single neuron composes the whole network.

To run this example execute:

```python
python main.py -f projects/examples/parameters/single_neuron_fI.py -c single_neuron_fIcurve --extra plot=True display=True save=True
```
