
The implementation of the simulator relies entirely on the structure of the parameters dictionaries,
which makes the correct specification of the parameters files the most sensitive (and error-prone)
aspect. The main modules simply extract the specifications described in these dictionaries and set-up
the experiment. So, the most critical step in setting up a numerical experiment with the current
implementation is the correct specification of all the complex parameter sets. In this section, we
explain the main features that all parameters must obey to and exemplify how the parameters file
containing all the dictionaries should be set up. However, different experiments can have very different
requirements and specifications (see examples). It is important to reinforce that the structure of these
dictionaries is the critical element to make everything work. If the naming conventions used are broken,
errors will follow.


Furthermore, modifications of the parameter values are accepted with some restrictions, depending
on what is currently implemented.

We start with a trimmed version of the parameter file in the [fI curve example](single-neuron-fi-curve/), to explain the 
 its basic components and structure:

```python
from defaults.paths import paths
from modules.parameters import ParameterSet, copy_dict

system_name = 'local'
data_label  = 'example1_singleneuron_fI'

# ######################################################################################################################
# PARAMETER RANGE declarations
# ======================================================================================================================
parameter_range = {
	'max_current': [800.]
}


def build_parameters(max_current):
	# ##################################################################################################################
	# DC input parameters
	# ==================================================================================================================
	total_time = 10000.        # total simulation time [ms]
	analysis_interval = 1000.  # duration of each current step [ms]
	min_current = 0.           # initial current amplitude [pA]


	# ##################################################################################################################
	# System / Kernel Parameters
	# ##################################################################################################################
	# system-specific parameters (resource allocation, simulation times)
	system_pars = dict(
		nodes=1,
	    ...
	)

	# main kernel parameter set
	kernel_pars = ParameterSet({
		'resolution': 0.1,
		...
	})
	# ##################################################################################################################
	# Recording devices
	# ##################################################################################################################
	multimeter = {
		'start': 0.,
		'stop': sys.float_info.max,
	    ...
	}

	# ##################################################################################################################
	# Neuron, Synapse and Network Parameters
	# ##################################################################################################################
	neuron_pars = {
		'AdEx': {
				'model': 'aeif_cond_exp',
				'C_m': 250.0,
		        ...
		}
	}

	net_pars = ParameterSet({
		'n_populations': len(neuron_pars.keys()),
		'pop_names': pop_names,
		...
	})
	
	neuron_pars = ParameterSet(neuron_pars)

	# ##################################################################################################################
	# Input/Encoding Parameters
	# ##################################################################################################################
	encoding_pars = ParameterSet({
		'generator': {
			'N': 1,
			'labels': ['DC_Input'],
			'models': ['step_current_generator'],
		}
	    ...
	})

	# ##################################################################################################################
	# Return dictionary of Parameters dictionaries
	# ==================================================================================================================
	return dict([('kernel_pars',    kernel_pars),
	             ('neuron_pars',    neuron_pars),
	             ('net_pars',       net_pars),
	             ('encoding_pars',  encoding_pars)])
```

Typically the following imports are necessary in a parameter file:

```python
from defaults.paths import paths  
from modules.parameters import ParameterSet
```

The experiment label and the system in which the experiments will run need to be specified and
will determine the system-specific paths to use as well as the label for data storage:

```python
system_name = 'local'
data_label  = 'example1_singleneuron_fI'
```

It is important to make sure that the `system_name` corresponds to a key in the `paths` dictionary.
Also, it is always advisable to provide appropriate labels for the data.

### Parameter range declarations and build function

NMSAT allows the definition of ranges for the values of various parameters of choice via the `parameter_range` dictionary,
in order simplify running the same experiment when only some parameter values change:
 
```python
parameter_range = {
	'max_current': [800.]
}

def build_parameters(max_current):
    ...
```
In this case, only one simulation will be executed where `max_current = 800.`. However, the following code 

```python
parameter_range = {
	'max_current': [800., 1200.],
	'total_time': [1000.]
}

def build_parameters(max_current, total_time):
    ...
```

will result in 2 separate runs, one for each parameter combination:

* `max_current = 800., total_time = 1000.`
* `max_current = 1200., total_time = 1000.`

The `build_parameters(...)` function is required in every parameter file. If the `parameter_range` dictionary is not 
empty, its keys (more precisely equivalent variable names) are passed as arguments to the function.

### Parameter types
The returned value of the `build_parameters(...)` function is a dictionary of `ParameterSets`, containing all the
necessary types of parameters to be used by the main experiment. These parameters defined in the function are themselves
`ParameterSets` dictionaries:

```python
return dict([('kernel_pars',   kernel_pars),
             ('neuron_pars',   neuron_pars),
             ('net_pars',      net_pars),
             ('encoding_pars', encoding_pars)])
```

Each key in the returned dictionary must match the name of the variable. Acceptable types (with examples of use) are: 

#### **Kernel** 
Specifies all relevant system and simulation parameters.

```python
kernel_pars = ParameterSet({
	'resolution': 0.1,		# simulation resolution
	'sim_time': 1000.,		# total simulation time (often not required)
	'transient_t': 0.,		# transient time 
	'data_prefix': data_label,
	'data_path': paths[system_name]['data_path'],
	'mpl_path': paths[system_name]['matplotlib_rc'],
	'overwrite_files': True,
	'print_time': (system_name == 'local'),
	'rng_seeds': range(msd + N_vp + 1, msd + 2 * N_vp + 1),
	'grng_seed': msd + N_vp,
	'total_num_virtual_procs': N_vp,
	'local_num_threads': 16,
	'np_seed': np_seed,

	'system': {
		'local': (system_name == 'local'),
		'system_label': system_name,
		'queueing_system': paths[system_name]['queueing_system'],
		'jdf_template': paths[system_name]['jdf_template'],
		'remote_directory': paths[system_name]['remote_directory'],
		'jdf_fields': {'{{ script_folder }}': '',
		               '{{ nodes }}': str(system_pars['nodes']),
		               '{{ ppn }}': str(system_pars['ppn']),
		               '{{ mem }}': str(system_pars['mem']),
		               '{{ walltime }}': system_pars['walltime'],
		               '{{ queue }}': system_pars['queue'],
		               '{{ computation_script }}': ''}}})
```
		               
If correctly specified, these parameters can be used during the initial setup of the main simulations:

```python
set_global_rcParams(parameter_set.kernel_pars['mpl_path'])
np.random.seed(parameter_set.kernel_pars['np_seed'])
nest.SetKernelStatus(extract_nestvalid_dict(kernel_pars.as_dict(), param_type='kernel'))
```

#### **Neuron** 
Neuron model parameters, must strictly abide by the naming conventions of the NEST model requested. For example:

```python
neuron_pars = {
	'neuron_population_name': {   # 
		'model': 'iaf_psc_exp',
		'C_m': 250.0,
		'E_L': 0.0,
		'V_reset': 0.0,
		'V_th': 15.,
		't_ref': 2.,
		'tau_syn_ex': 2.,
		'tau_syn_in': 2.,
		'tau_m': 20.}}
```

Typically the neuron parameters are only used within the parameters file, as they will be placed
in the network parameter dictionary, where the code will use them.

#### **Network** 
Network parameters, specifying the composition, topology and which variables to record
from the multiple populations:

```python
net_pars = {
	'n_populations': 2,			# total number of populations
	'pop_names': ['E', 'I'],    # names for each population, here 'E' for excitatory and 'I' for inhibitory
	'n_neurons': [8000, 2000],  # number of neurons in each population
	'neuron_pars': [
	        neuron_pars['E'],   # neuron parameters for population name 'E' 
	        neuron_pars['I']],  # neuron parameters for population name 'I'
	'randomize_neuron_pars': [  # randomize certain parameters if necessary
	        {'V_m': (np.random.uniform, {'low': 0.0, 'high': 15.})},  
	        {'V_m': (np.random.uniform, {'low': 0.0, 'high': 15.})}], 
	'topology': [False, False],     # does the network have topology? True or False
	'topology_dict': [None, None],  # dictionary with topology parameters, 
	                                # if topology is set to True for the population 
	'record_spikes': [True, True],  # whether to record spikes for each population
	'spike_device_pars': [          # parameters for spike recording devices
	        copy_dict(rec_devices,
                    {'model': 'spike_detector',
                     'record_to': ['memory'],
                     'label': ''}),
            copy_dict(rec_devices,
                    {'model': 'spike_detector',
                     'record_to': ['memory'],
                     'label': ''})],
	'record_analogs': [False, False],   # whether to record analog data for each population
	'analog_device_pars': [None, None]} # parameters for analog devices
```

Note that the dimensionality of all the list parameters has to be equal to `n_populations`. If
these parameters are all correctly set, generating a network is as simple as:


```python
net = Network(net_pars)
net.connect_devices()
```


#### **Connection** 
Connectivity parameters, specifying the composition, topology and connectivity of
the multiple populations:

```python
connection_pars = {
    # [int] - total number of connections to establish
	'n_synapse_types': 4,
	 
	 # [list of tuples] - each tuple corresponds to 1 connection and 
	 # has the form (tget_pop_label, src_pop_label)
	'synapse_types': [('E', 'E'), ('E', 'I'), ('I', 'E'), ('I', 'I')],
	
	# [list of bool] - whether synaptic connections are 
	# established relying on nest.topology module
	'topology_dependent':[False, False, False, False],
	 
	# [list of str] - names of the synapse models to realize on each synapse type
	'models': ['static_synapse', 'static_synapse', 'static_synapse', 'static_synapse'],  
	         
    # [list of dict] - each entry in the list contains the synapse
    # dictionary (if necessary) for the corresponding synapse model
	'model_pars': [synapse_pars_dict, synapse_pars_dict, 
	               synapse_pars_dict, synapse_pars_dict],
	     
    # [list of dict] - weight distribution parameters (in NEST format)
	'weight_dist': [common_syn_specs['weight'], common_syn_specs['weight'], 
                    common_syn_specs['weight'], common_syn_specs['weight']],
	
	# Provide an externally specified connectivity matrix if
    # pre-computed weights matrices are provided, delays also need 
    # to be pre-specified, as an array with the same dimensions as W, 
    # with delay values on each corresponding entry. Alternatively, 
    # if all delays are the same, just provide a single number.
	'pre_computedW': [None, None, None, None],

    # [list of dict] - delay distribution parameters (in NEST format)
	'delay_dist': [common_syn_specs['delay'], common_syn_specs['delay'],
				   common_syn_specs['delay'], common_syn_specs['delay']],

    # [list of dict] - for topological connections
	'conn_specs': [conn_specs, conn_specs, conn_specs, conn_specs],

    # [list of dict] - 
	'syn_specs': []
```

To connect the network, just provide this parameter set to the connect method of the `Network`
object (for complex connectivity schemes, it is worth setting the `progress=True` to accompany the
connection progress):

```python
net.connect_populations(parameter_set.connection_pars, progress=True)
```


#### **Stimulus** 
Stimulus or task parameters:

```python
stim_pars = {
		'n_stim': 10,		        # number of stimuli
		'elements': ['A', 'B',      #  
		             'C', ...],
		'grammar': None,            #!!
		'full_set_length': int(T + T_discard),  # total samples
		'transient_set_length': int(T_discard), # size of transient set (will be discarded)
		'train_set_length': int(0.8 * T),       # size of training set
		'test_set_length': int(0.2 * T)}        # size of test set
```

To generate a stimulus set:

```python
stim_set = StimulusSet()
stim_set.generate_datasets(stim_pars)
```


#### **Input** 
Specifies the stimulus transduction and/or the input signal properties:

```python
input_pars = {
	'signal': {
		'N': 3,		# Dimensionality of the signal
		 
        # Duration of each stimulus presentation. 
        # Various settings are allowed:
        #   - List of n_trials elements whose values correspond to the 
        #     duration of 1 stimulus presentation
        #   - List with one single element (uniform durations)
        #   - Tuple of (function, function parameters) specifying 
        #     distributions for these values, 
        #   - List of N lists of any of the formats specified... (*)
		'durations': [(np.random.uniform, {'low': 500., 'high': 500., 'size': n_trials})],

        # Inter-stimulus intervals - same settings as 'durations'.
		'i_stim_i':	[(np.random.uniform, {'low': 0., 'high': 0., 'size': n_trials-1})],

        # input mask - implemented options:
        #   box({}), exp({'tau'}), double_exp({'tau_1','tau_2'}), gauss({'mu', 'sigma'})
		'kernel': ('box', {}),	

		'start_time': 0.,			        # global signal onset time
		'stop_time': sys.float_info.max,	# global signal offset time
        
        # maximum signal amplitude - as in durations and i_stim_i,
        # can be a list of values, or a function with parameters
		'max_amplitude': [(np.random.uniform, {'low': 10., 'high': 100., 'size': n_trials})],

		'min_amplitude': 0.,	# minimum amplitude - will be added to the signal
		'resolution': 1.	    # temporal resolution
		},

	'noise': {
		'N': 3,	# Dimensionality of the noise component (common or 
			# multiple independent noise sources)
			
        # [list] - Type of noise:
        #  - Either a string for 'OU', 'GWN'
        #  - or a function, e.g. np.random.uniform
		'noise_source': ['GWN'],
			
		# [dict] Parameters (specific for each type of noise):
        # - 'OU' -> {'tau', 'sigma', 'y0'}
        # - 'GWN' -> {'amplitude', 'mean', 'std'}
        # - function parameters dictionary (see function documentation)
		'noise_pars': {'amplitude': 1., 'mean': 1., 'std': 0.1},
		
		'rectify': True,	# [bool] - rectify noise component
		 
		# global onset time (single value), 
		# or local onset times if multiple instances are required (list)
		'start_time': 0.,	
		
		# global offset time (single value), or local offset
		# times, if multiple instances are required
		'stop_time': sys.float_info.max,	
		
		'resolution': 1.,	                # signal resolution (dt)}}
```

**Note:** the chosen specifications of `durations`, `i_stim_i` and `max_amplitude`
must be consistent, i.e., if `durations` is provided as a single element list, the same format must
be applied to `i_stim_i` and `max_amplitude`. (test)

```python
inputs = InputSignalSet(parameter_set, stim_set, online=online)
inputs.generate_datasets(stim_set)
```

#### **Encoding** 
-

```python
enc_layer = EncodingLayer(parameter_set.encoding_pars, signal=inputs.full_set_signal, online=on
enc_layer.connect(parameter_set.encoding_pars, net)
```

#### **Decoding** 
-

```python
net.connect_decoders(parameter_set.decoding_pars)
enc_layer.connect_decoders(parameter_set.encoding_pars.input_decoder)
```

#### **Analysis** 
-

```python

```

### Parameter presets 
For convenience, to reduce the length and complexity of the parameter files and the likelihood of
accidentally changing the values of fixed, commonly used parameter sets, we typically include a set of
preset dictionaries and simple functions to retrieve them and to simplify the construction of parameters
files (see examples and upcoming project files).