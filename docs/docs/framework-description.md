## NMSAT architecture overview:

###[!!!I think it'd be nice to have a global, yet quite general overview of what components there are, what they do and the flow through the layers / components!!! I tried (& failed) to start describing it. ]

![alt-text](/images/global_illustration-01.png)
The framework is built on a layered structure, which can be divided into three main elements (left to right):
  
  * **Input Architect:** handles all generation and preprocessing of the input stimuli and signals. Being quite flexible, it is difficult to describe a single workflow of Input Architect, but a general experiment looks like this: first, a set of input stimuli is generated. Afterwards, for each stimulus a corresponding signal is created, yielding a set of input signals [maybe add what types are possible? spike_input, ] to which various types of noise [e.g., GWN or OU processes] can be added if needed. The input signal set is then passed to the encoding layer, which has two types of components: generators and encoders. Generators can be of different types (...).
    
    However, you can design experiments without any stimuli ... [here I just wanted to highlight the versatility.. maybe a short description of an experiment without any stimuli / noise, just background noise?]
  
  * **Network Architect:** 
  
  * **Analysis:**

## Code structure

The code is organized as follows:


```
├── nmsat
│   ├── modules
│   │   ├── parameters.py
│   │   ├── input_architect.py
│   │   ├── net_architect.py
│   │   ├── signals.py
│   │   ├── analysis.py
│   │   ├── visualization.py
│   │   ├── io.py
│   ├── defaults
│   │   ├── paths.py
│   │   ├── matplotlib_rc
│   │   ├── cluster_templates
│   │   │   ├── cluster_jdf.sh
│   ├── data
│   ├── projects
│   │   ├── project_name
│   │   │   ├── computations
│   │   │   ├── parameters
│   │   │   │   ├── preset
│   │   │   ├── scripts
│   │   │   ├── read_data
│   ├── export
```


The core functionality lies in the modules packages, which contain all the relevant classes and
functions used. The specifics will be explained in greater detail below, but in general the modules
are responsible for:

* `parameters` - parsing and preparing all parameters files; retrieving stored parameter sets and
spaces and harvesting data
* `input_architect` - generating and setting up all the relevant input stimuli and signals; handling
input data; generating and connecting input encoding layers
* `net_architect` - generating specific networks and neuronal populations; generating all connectivity and topology features; 
connecting populations; ...
* `signals` - wrapping and processing the various signal types used in the framework (spiking activity,
analog variables, etc)
* `analysis` - post-processing and analysing population activity in various ways
* `visualization` - plotting routines
* `io` - loading and saving data
