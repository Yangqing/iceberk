"""This folder contains some c++ implementations that either make code run
faster or handles some numpy tricky issues.
"""
import ctypes as ct
import numpy as np
import os
from iceberk import mpi

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
_POOL_METHODS = {'max':0, 'ave': 1, 'rms': 2}

def fastpooling(image, grid, method):
    output = np.empty((grid[0], grid[1], image.shape[-1]))
    _FASTMATH.fastpooling(
            image.ctypes.data_as(ct.POINTER(ct.c_double)),
            ct.c_int(image.shape[0]),
            ct.c_int(image.shape[1]),
            ct.c_int(image.shape[2]),
            ct.c_int(grid[0]),
            ct.c_int(grid[1]),
            ct.c_int(_POOL_METHODS[method]),
            output.ctypes.data_as(ct.POINTER(ct.c_double)))
    return output


################################################################################
# fast mean and std
################################################################################
_FASTMATH.fastsumx2.restype = None
_FASTMATH.fastsumx2.argtypes = [ct.POINTER(ct.c_double),
                              ct.POINTER(ct.c_double),
                              ct.c_int,
                              ct.c_int,
                              ct.c_int,
                              ct.POINTER(ct.c_double)]

def fast_std_nompi(mat, axis, mean = None):
    """Equivalent to np.std(mean, axis) without duplicating the 
    matrix like numpy does. Will save some time when dealing with large 
    matrices. Pass in the precomputed mean, or we will compute it inside the
    function.
    
    This program does NOT use mpi.
    """
    if mat.flags['C_CONTIGUOUS'] != True \
            or mat.dtype != np.float64 or mat.ndim != 2:
        raise ValueError, "Unsupported input matrix."
    if mean is None or mean.dtype != np.float64:
        mean = np.mean(mat, axis)
    std = np.empty_like(mean)
    _FASTMATH.fastsumx2(mat.ctypes.data_as(ct.POINTER(ct.c_double)),
            mean.ctypes.data_as(ct.POINTER(ct.c_double)),
            ct.c_int(mat.shape[0]), 
            ct.c_int(mat.shape[1]),
            ct.c_int(axis),
            std.ctypes.data_as(ct.POINTER(ct.c_double)))
    std /= (mat.shape[axis])
    np.sqrt(std, out=std)
    return std

def column_meanstd(mat):
    """Computes the mean and std of the matrix, assuming that it is chopped
    along the rows and stored in a distributed fashion.
    """
    if mat.flags['C_CONTIGUOUS'] != True \
            or mat.dtype != np.float64 or mat.ndim != 2:
        raise ValueError, "Unsupported input matrix."
    num_data_local = mat.shape[0]
    # get the mean
    m_local = mat.sum(0)
    m = np.empty_like(m_local)
    num_data = mpi.COMM.allreduce(num_data_local)
    mpi.COMM.Allreduce(m_local, m)
    m /= num_data
    # get the std
    sumx2_local = np.empty_like(m)
    _FASTMATH.fastsumx2(mat.ctypes.data_as(ct.POINTER(ct.c_double)),
            m.ctypes.data_as(ct.POINTER(ct.c_double)),
            ct.c_int(mat.shape[0]), 
            ct.c_int(mat.shape[1]),
            ct.c_int(0),
            sumx2_local.ctypes.data_as(ct.POINTER(ct.c_double)))
    std = np.empty_like(sumx2_local)
    mpi.COMM.Allreduce(sumx2_local, std)
    std /= num_data
    np.sqrt(std, out=std)
    return m, std