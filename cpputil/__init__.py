"""This folder contains some c++ implementations that either make code run
faster or handles some numpy tricky issues.
"""
import ctypes as ct
import numpy as np
import os

# first, let's import the library
try:
    _FASTMATH = np.ctypeslib.load_library('libfastmath.so',
            os.path.join(os.path.dirname(__file__), '_cpp'))
except Exception, e:
    raise RuntimeError, "I cannot load libfastpool.so. please run make."

################################################################################
# fast pooling
################################################################################
_FASTMATH.fastpooling.restype = ct.c_int
_FASTMATH.fastpooling.argtypes = [ct.POINTER(ct.c_double), # image
                                      ct.c_int, # height
                                      ct.c_int, # width
                                      ct.c_int, # num_channels
                                      ct.c_int, # grid[0]
                                      ct.c_int, # grid[1]
                                      ct.c_int, # method
                                      ct.POINTER(ct.c_double) # output
                                     ]
fastpooling = _FASTMATH.fastpooling

