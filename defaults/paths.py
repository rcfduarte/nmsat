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
		'report_templates_path':NETWORK_HOME + '/defaults/report_templates/',
		'report_path': 			NETWORK_HOME + '/defaults/report_templates/',
		'queueing_system':      None},

	'Blaustein': {
		'data_path':            NETWORK_HOME + '/data/',
		'jdf_template':         NETWORK_HOME + '/defaults/cluster_templates/Blaustein_jdf.sh',
		'matplotlib_rc':        NETWORK_HOME + '/defaults/matplotlib_rc',
		'remote_directory':     NETWORK_HOME + '/export/',
		'queueing_system':      'slurm'},

	'Jureca': {
		'data_path':            '/work/jias61/jias6101/',
		'jdf_template':         NETWORK_HOME + '/defaults/cluster_templates/Jureca_jdf.sh',
		'matplotlib_rc':        NETWORK_HOME + '/defaults/matplotlib_rc',
		'remote_directory':     NETWORK_HOME + '/export/',
		'queueing_system':      'slurm'},

	'MPI': {
		'data_path':            NETWORK_HOME + '/data/',
		'jdf_template':         NETWORK_HOME + '/defaults/cluster_templates/MPI_jdf.sh',
		'matplotlib_rc':        NETWORK_HOME + '/defaults/matplotlib_rc',
		'remote_directory':     NETWORK_HOME + '/export/',
		'queueing_system':      'sge'},
}