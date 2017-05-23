One of the core tasks in each experiment is to set up a network of neuronal populations to be simulated. In NMSAT, 
  individual neurons are grouped into homogeneous populations and stored in `Population` objects. These, in turn, are
  organized into a single `Network` object that acts as a wrapper for all populations to allow their efficient handling. 
   
  Both classes are defined in the `net_architect.py` module.  
  
### Populations
  `Population` objects are used to handle each of the simulated neuronal populations and contain
  their parameters, such as name (unique among all populations), size, topology, etc. They also define what recording devices 
  (`spike_detector, multimeter`) should be connected to the populations, store the spiking and analog activities 
  in `SpikeLists` and `AnalogSignalLists`, and keep track of the attached [decoding layer](/decoding/).   

### Network

Generally there is only one `Network` object in each experiment, which keeps a list with all the populations building the 
  network, along with their connectivity properties. It provides routines for creating and connecting the populations, 
  attaching the recording devices and decoders, and extracting the activity from the simulator after termination. It is 
  also possible to merge different populations to create larger, heterogeneous clusters within the network. 

### Creating populations and networks

Three parameter dictionaries are used to define populations and networks (for detailed description follow the links):

* [`neuron_pars`](/parameters/#neuron) - neuron models and their parameters 
* [`net_pars`](/parameters/#network) - specifies the composition, topology and which variables to record
from the multiple populations. Note that there is no separate dictionary for the individual populations, they are all
defined in the lists of this dictionary.
* [`connection_pars`](/parameters/#connection) - parameter set defining connections among populations and their synaptic
properties (synapse model, weight distributions, delays, etc.). 


If these parameters are all correctly set, generating a network and initializing all connections is as simple as:


```python
# create Network object
net = Network(parameter_set.net_pars) 

# optional, example for merging two populations
# net.merge_subpopulations([net.populations[0], net.populations[1]], name='EI') 

# connect populations, for complex connectivity schemes it is worth setting 
# `progress=True` to accompany the connection progress
net.connect_populations(parameter_set.connection_pars, progress=False)

# attach devices
net.connect_devices()

# connect the decoders
net.connect_decoders(parameter_set.decoding_pars)
```