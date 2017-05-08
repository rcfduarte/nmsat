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
