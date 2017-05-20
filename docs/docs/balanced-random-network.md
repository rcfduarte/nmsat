In this simple example, we perform a very simple and common experiment: determining the responses
of a single neuron to injected current at different amplitudes. The specific setup is illustrated below

![alt-text](/images/example3-01.png)

In this simple setup, one single generator, a `’step_current_generator’` is used and a single neuron composes the whole network.

To run this example execute:

```python
python main.py -f projects/examples/parameters/noise_driven_dynamics.py -c population_noisedriven --extra plot=True display=True save=True
```
