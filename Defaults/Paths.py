import os

NETWORK_HOME = os.environ.get("NETWORK_SIMULATION_HOME")

if NETWORK_HOME is None:
	print("Please set the project root directory environment variable! (source configure.sh)\nExiting.")
	exit(0)

paths = {
	'local': {
		'data_path': NETWORK_HOME + '/Data/',
		'modules_path': NETWORK_HOME + '/Experiments/',
		'jdf_template': NETWORK_HOME + '/Defaults/ClusterTemplates/bwUniCluster_jdf.sh',
		'matplotlib_rc': NETWORK_HOME + '/Defaults/matplotlib_rc',
		'remote_directory': NETWORK_HOME + '/Experiments/export/',
		'report_templates_path': NETWORK_HOME + '/Defaults/ReportTemplates/',
		'report_path': NETWORK_HOME + '/Data/'},

	'bwUniCluster': {
		'data_path': '/work/fr/fr_fr/fr_rd1000/',
		'modules_path': '/home/fr/fr_fr/fr_rd1000/NetworkSimulationTestbed',
		'jdf_template': '/home/fr/fr_fr/fr_rd1000/NetworkSimulationTestbed/Defaults/ClusterTemplates/bwUniCluster_jdf.sh',
		'matplotlib_rc': '/home/fr/fr_fr/fr_rd1000/NetworkSimulationTestbed/Defaults/matplotlib_rc2',
		'remote_directory': '/home/fr/fr_fr/fr_rd1000/NetworkSimulationTestbed/export/',
		'report_templates_path': '/home/fr/fr_fr/fr_rd1000/NetworkSimulationTestbed/Defaults/ReportTemplates/',
		'report_path': '/work/fr/fr_fr/fr_rd1000/'},

	'Blaustein': {
		'data_path': '/home/r.duarte/Data/',
		'modules_path': '/home/r.duarte/NetworkSimulationTestbed',
		'jdf_template': '/home/r.duarte/NetworkSimulationTestbed/Defaults/ClusterTemplates/Blaustein_jdf.sh',
		'matplotlib_rc': '/home/r.duarte/NetworkSimulationTestbed/Defaults/matplotlib_rc2',
		'remote_directory': '/home/r.duarte/NetworkSimulationTestbed/export/',
		'report_templates_path': '/home/r.duarte/NetworkSimulationTestbed/Defaults/ReportTemplates/',
		'report_path': '/home/r.duarte/Data/'},

	'Jureca': {
		'data_path': '/work/jias61/jias6101/',
		'modules_path': '/homea/jias61/jias6101/NetworkSimulationTestbed',
		'jdf_template': '/homea/jias61/jias6101/NetworkSimulationTestbed/Defaults/ClusterTemplates/Jureca_jdf.sh',
		'matplotlib_rc': '/homea/jias61/jias6101/NetworkSimulationTestbed/Defaults/matplotlib_rc2',
		'remote_directory': '/homea/jias61/jias6101/NetworkSimulationTestbed/export/',
		'report_templates_path': '/homea/jias61/jias6101/NetworkSimulationTestbed/Defaults/ReportTemplates/',
		'report_path': '/work/jias61/jias6101/'},
	'Jureca_mem': {
		'data_path': '/work/jias61/jias6101/',
		'modules_path': '/homea/jias61/jias6101/NetworkSimulationTestbed',
		'jdf_template': '/homea/jias61/jias6101/NetworkSimulationTestbed/Defaults/ClusterTemplates/Jureca_largeMem.sh',
		'matplotlib_rc': '/homea/jias61/jias6101/NetworkSimulationTestbed/Defaults/matplotlib_rc2',
		'remote_directory': '/homea/jias61/jias6101/NetworkSimulationTestbed/export/',
		'report_templates_path': '/homea/jias61/jias6101/NetworkSimulationTestbed/Defaults/ReportTemplates/',
		'report_path': '/work/jias61/jias6101/'}
}