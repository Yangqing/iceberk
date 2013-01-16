"""This folder contains some c++ implementations that either make code run
faster or handles some numpy tricky issues.
"""
import ctypes as ct
import numpy as np
import os
from iceberk import mpi
from iceberk.mathutil import CHECK_IMAGE, CHECK_SHAPE

# first, let's import the library
try:
    _CPPUTIL = np.ctypeslib.load_library('libcpputil.so',
            os.path.join(os.path.dirname(__file__), '_cpp'))
except Exception, e:
    raise RuntimeError, "I cannot load libcpputil.so. please run make."

################################################################################
# fast pooling
################################################################################
_POOL_METHODS = {'max':0, 'ave': 1, 'rms': 2}
_CPPUTIL.fastpooling.restype = ct.c_int
_CPPUTIL.fastpooling.argtypes = [ct.POINTER(ct.c_double), # image
                                      ct.c_int, # height
                                      ct.c_int, # width
                                      ct.c_int, # num_channels
                                      ct.c_int, # grid[0]
                                      ct.c_int, # grid[1]
                                      ct.c_int, # method
                                      ct.POINTER(ct.c_double) # output
                                     ]
_CPPUTIL.fast_oc_pooling.restype = ct.c_int
_CPPUTIL.fast_oc_pooling.argtypes = [ct.POINTER(ct.c_double), # image
                                      ct.c_int, # grid[0]
                                      ct.c_int, # grid[1]
                                      ct.c_int, # num_channels
                                      ct.c_int, # method
                                      ct.POINTER(ct.c_double) # output
                                     ]

def fastpooling(image, grid, method, out = None):
    if out is None:
        output = np.empty((grid[0], grid[1], image.shape[-1]))
    else:
        output.resize(grid[0], grid[1], image.shape[-1])
    _CPPUTIL.fastpooling(
            image.ctypes.data_as(ct.POINTER(ct.c_double)),
            ct.c_int(image.shape[0]),
            ct.c_int(image.shape[1]),
            ct.c_int(image.shape[2]),
            ct.c_int(grid[0]),
            ct.c_int(grid[1]),
            ct.c_int(_POOL_METHODS[method]),
            output.ctypes.data_as(ct.POINTER(ct.c_double)))
    return output

def fast_oc_pooling(image, grid, method, out = None):
    num_output = grid[0] * (grid[0] + 1) * grid[1] * (grid[1] + 1) / 4
    if out is None:
        output = np.empty((num_output, image.shape[-1]))
    else:
        output.resize(num_output, image.shape[-1])
    if image.shape[0] != grid[0] or image.shape[1] != grid[1]:
        # do a first pass fast pooling
        image = fastpooling(image, grid, method)
    _CPPUTIL.fast_oc_pooling(
            image.ctypes.data_as(ct.POINTER(ct.c_double)),
            ct.c_int(grid[0]),
            ct.c_int(grid[1]),
            ct.c_int(image.shape[2]),
            ct.c_int(_POOL_METHODS[method]),
            output.ctypes.data_as(ct.POINTER(ct.c_double)))
    return output



################################################################################
# im2col operation
################################################################################
_CPPUTIL.im2col.restype = None
_CPPUTIL.im2col.argtype = [ct.POINTER(ct.c_double),
                           ct.POINTER(ct.c_int),
                           ct.POINTER(ct.c_int),
                           ct.c_int,
                           ct.POINTER(ct.c_double)]

def im2col(image, psize, stride, out = None):
    image = np.ascontiguousarray(np.atleast_3d(image), dtype=np.float64)
    imsize = np.asarray(image.shape, dtype = ct.c_int)
    psize = np.asarray(psize).astype(ct.c_int)
    stride = int(stride)
    if np.any(imsize[:2] < psize):
        raise ValueError, "No patch can be extracted."
    newsize = (imsize[:2] - psize) / stride + 1
    if out is None:
        out = np.empty((newsize[0], newsize[1], 
                        psize[0] * psize[1] * imsize[2]))
    else:
        CHECK_IMAGE(out)
        CHECK_SHAPE(out, (newsize[0], newsize[1], 
                          psize[0] * psize[1] * imsize[2]))
    _CPPUTIL.im2col(image.ctypes.data_as(ct.POINTER(ct.c_double)),
                    imsize.ctypes.data_as(ct.POINTER(ct.c_int)),
                    psize.ctypes.data_as(ct.POINTER(ct.c_int)),
                    ct.c_int(stride),
                    out.ctypes.data_as(ct.POINTER(ct.c_double)))
    return out

