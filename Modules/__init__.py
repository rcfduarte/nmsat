__author__ = 'duarte'
"""
===========================================================================
Modules Package
===========================================================================

Collection of modules comprising all the tools and objects necessary
to set up complex network simulations, input properties, connectivity
properties as well as develop complex, analysis, data processing
and plotting of the resulting data.

"""
import warnings
import pkgutil
import inspect

__all__ = ['analysis', 'io', 'parameters', 'input_architect',
           'signals', 'net_architect', 'visualization']

# __all__ = []
# for loader, name, is_pkg in pkgutil.walk_packages(__path__):
#     module = loader.find_module(name).load_module(name)
#
#     for name, value in inspect.getmembers(module):
#         if name.startswith('__'):
#             continue
#
#         globals()[name] = value
#         __all__.append(name)


def get_import_warning(name):
	return """** %s ** package is not installed. To have functions using %s please install the package.""" % (name,
	                                                                                                          name)


def check_dependency(name):
	"""
	verify if package is installed
	:param name: string with the name of the package to import
	:return:
	"""
	try:
		exec("import %s" % name)
		return True
	except ImportError:
		print get_import_warning(name)

