
def isiterable(x):
	"""
	Verify if input is iterable (list, dictionary, array...)
	:param x: input
	:return: boolean
	"""
	return hasattr(x, '__iter__') and not isinstance(x, basestring)
