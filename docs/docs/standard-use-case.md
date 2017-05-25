A numerical experiment in this framework consists of 2 or 3 main files (see Examples section):


* **parameters_file** - specifying all the complex parameter sets and dictionaries required to set up
the experiment.
* **experiment_script** - mostly used during development, for testing and debugging purposes.
These scripts parse the `parameters_file` and run the complete experiment
* **computation_file** - after the development and testing phase is completed, the experiment can
be copied to a computation_file, which can be used from the main... (*)


These files should be stored within a `project/` folder, and in the `parameters/`, `scripts/` and `computations/` 
folders, respectively.

## Running an experiment

The way in which an experiment is run depends on the system used:


### Local machine
Just go to the main `nmsat/` directory and execute the experiment as:

```bash
python main.py -f {parameters_file} -c {computation_function} --extra {extra_parameters}
```

where `parameters_file` refers to the (full or relative) path to the parameters file for the experiment,
`computation_function` is the name of the computation function to be executed on that parameter
set (must match the name of a file in the project’s `computations/` folder) and `extra_parameters` are
parameter=value pairs for different, extra parameters (specific to each computation).

###Cluster 
On a computer cluster or supercomputer, the execution of the framework has a slightly different meaning. 
Instead of executing the code, it generates a series of files that can be used to submit the jobs to the system’s scheduler.


To do this:

1. add an entry to `nmsat/defaults/paths.py` for your template (here ’Cluster’)
2. adapt the default cluster template in `nmsat/defaults/cluster_templates/Cluster_jdf.sh`
to match your cluster requirements
3. change `run=’local’` to `run=’Cluster’` in your parameter script
4. execute the following command from `nmsat/`

```python
python main.py -f {parameters_file} -c {computation_function} --extra {extra_parameters} --cluster=Blaustein
```

5. go to `nmsat/export/my_project_name/` and submit jobs  via

```python
python submit_jobs.py 0 1
```

###Simulation output

After a simulation is completed, all the relevant output data is stored in the pre-defined data path,
within a folder named after the `data_label` specified in the parameters file of the experiment. The output data 
structure is organized as follows:

```
── nmsat
│   ├── data
│   │   ├── data_label
│   │   │   ├── Figures
│   │   │   ├── Results
│   │   │   ├── Parameters
│   │   │   ├── Activity
│   │   │   ├── Input
│   │   │   ├── Output
│   │   ├── data_label_ParameterSpace.py # only created if running parameter scans
```

To illustrate this, consider the case of the first [example](/single-neuron-fi-curve): the data path set in `defaults/paths.py`
is `nmsat/data`. In the parameter file `nmsat/projects/examples/single-neuron-fi-curve.py` we define 
```python
data_label = 'example1'
```
so after running the experiment the output will be in `nmsat/data/example1/`. If we also ran a parameter scan, then 
a copy of the original parameter file would be created in nmsat/data/example1_ParameterScan.py`, which is not meant 
to be edited manually but is used when harvesting the results for post-processing ([see below](/standard-use-case/#harvesting-stored-results)). 



###Analysing and plotting
Analysis and plotting can be (and usually is) done within the main computation, so as to extract and
store only the information that is relevant for the specific experiment. Multiple, standard analysis and
plotting routines are implemented for various complex experiments, with specific objectives. Naturally,
this is highly mutable as new experiments always require specific analyses methods.


Alternatively, as all the relevant data is stored in the results dictionaries, you can read it and
process it offline, applying the same or novel visualization routines.


###Harvesting stored results
The Results folder stores all the simulation results for the given experiment, as pickle dictionaries.
Within each project, as mentioned earlier, a read_data folder should be included, which contains files
to parse and extract the stored results (see examples).


```python
project      = 'project_name'
data_path    = '/path/label/'
data_label   = 'example1'
results_path = data_path + data_label + '/Results/'

# set defaults and paths
set_project_paths(project)
set_global_rcParams(paths['local']['matplotlib_rc'])

# re-create ParameterSpace
pars_file = data_path + data_label + '_ParameterSpace.py'
pars = ParameterSpace(pars_file)

# print the full nested structure of the results dictionaries
pars.print_stored_keys(results_path)

# harvest a specific result based on the results structure
data_array = pars.harvest(results_path, key_set='dict_key1/dict_key1.1/dict_key1.1.1')
```