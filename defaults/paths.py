"""
Defines paths to various directories and files specific to the running mode (locally or on a cluster).
Should be modified by users according to their particular needs.
"""

import os

NMSAT_HOME = os.environ.get("NMSAT_HOME")

# exit if environment variable is not set
if NMSAT_HOME is None:
	print("Please set the project root directory environment variable! (source configure.sh)\nExiting.")
	exit(0)

paths = {
	'local': {
		'data_path': 		NMSAT_HOME + '/data/',					# output directory, must be created before running
		'jdf_template': 	None,									# cluster template not needed
		'matplotlib_rc': 	NMSAT_HOME + '/defaults/matplotlib_rc',	# custom matplotlib configuration
		'remote_directory': NMSAT_HOME + '/experiments/export/',	# directory for export scripts to be run on cluster
		'queueing_system':  None},									# only when running on clusters

	'Cluster': {
		'data_path': 		NMSAT_HOME + '/data/',
		'jdf_template': 	NMSAT_HOME + '/defaults/cluster_templates/Cluster_jdf.sh', # cluster template
		'matplotlib_rc': 	NMSAT_HOME + '/defaults/matplotlib_rc',
		'remote_directory': NMSAT_HOME + '/export/',
		'queueing_system':  'slurm'}, # slurm or sge
}