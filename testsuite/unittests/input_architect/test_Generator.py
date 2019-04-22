"""
Test the `Generator` class in input_architect.py and all its methods.
"""

import sys
sys.path.append('../../../')
sys.path.append('../')

import numpy as np

# nest.init(sys.argv + ['--verbosity=QUIET', '--quiet'])  # turn off all NEST messages

from modules import parameters
from modules import input_architect as ia
from modules import signals as sg


########################################################################################################################
# Test functions
########################################################################################################################