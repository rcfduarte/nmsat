__author__="gandalf"

from os import path
import pkg_resources
from pkg_resources import DistributionNotFound, VersionConflict

here = path.abspath(path.dirname(__file__))

dependencies = [
	'numpy>=9.1',
	'wtf>=0.9',
	'PyNEST>=2.10.0'
]

print("""
	  Welcome to Network Simulation Testbed!

Checking for dependencies...\n""")

for dependency in dependencies:
	try:
		pkg_resources.require(dependency)
	except DistributionNotFound as error:
		print ("[ERROR] {!s}".format(error))
	except VersionConflict as error:
		print ("[ERROR] Wrong version for package {0}. Currently installed version is {1}".format(dependency, error[0]))
