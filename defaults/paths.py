import os

NETWORK_HOME = os.environ.get("NETWORK_SIMULATION_HOME")

if NETWORK_HOME is None:
	print("Please set the project root directory environment variable! (source configure.sh)\nExiting.")
	exit(0)

paths = {
	'local': {
		'data_path': 			NETWORK_HOME + '/data/',
		'jdf_template': 		None,
		'matplotlib_rc': 		NETWORK_HOME + '/defaults/matplotlib_rc',
		'remote_directory': 	NETWORK_HOME + '/experiments/export/',
		'queueing_system':      None},

	'Cluster': {
		'data_path':            NETWORK_HOME + '/data/',
		'jdf_template':         NETWORK_HOME + '/defaults/cluster_templates/Cluster_jdf.sh',
		'matplotlib_rc':        NETWORK_HOME + '/defaults/matplotlib_rc',
		'remote_directory':     NETWORK_HOME + '/export/',
		'queueing_system':      'slurm'}, # slurm or sge
}