import os
# sets the repository root directory, everything is relative to that
NETWORK_HOME = os.path.dirname(os.path.realpath(__file__)) + "/../../../../"
PROJECT_HOME = os.path.dirname(os.path.realpath(__file__)) + "/../../"

paths = {
	'local': {
		'data_path': 				PROJECT_HOME + '/data/',
		'modules_path': 			NETWORK_HOME + '/experiments/',
		'jdf_template': 			NETWORK_HOME + '/defaults/ClusterTemplates/bwUniCluster_jdf.sh',
		'matplotlib_rc': 			NETWORK_HOME + '/defaults/matplotlib_rc',
		'remote_directory': 		NETWORK_HOME + '/experiments/export/',
		'report_templates_path': 	NETWORK_HOME + '/defaults/ReportTemplates/',
		'report_path': 				PROJECT_HOME + '/data/'},

}