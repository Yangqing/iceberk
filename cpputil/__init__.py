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
_POOL_METHODS = {'max':0, 'ave': 1, 'rms': 2}

def fastpooling(image, grid, method):
    output = np.empty((grid[0], grid[1], image.shape[-1]))
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


################################################################################
# fast mean and std
################################################################################
_CPPUTIL.fastsumx2.restype = None
_CPPUTIL.fastsumx2.argtypes = [ct.POINTER(ct.c_double),
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
    
    This program does NOT use mpi. Also, we do not check the correctness of the
    passed in mean values, so if you want to compute the length of each vector,
    simply pass in an all-0 mean vector.
    """
    raise NotImplementedError, "fastsumx2 large mem bug, need to be checked"
    if mat.flags['C_CONTIGUOUS'] != True \
            or mat.dtype != np.float64 or mat.ndim != 2:
        raise TypeError, "Unsupported input matrix: %s %s\n%s" % \
                (repr(mat.dtype), repr(mat.shape), repr(mat.flags), )
    if axis != 0 and axis != 1:
        raise ValueError, "Axis should either be 0 or 1."
    if mean is None or mean.dtype != np.float64:
        mean = np.mean(mat, axis)
    std = np.empty_like(mean)
    _CPPUTIL.fastsumx2(mat.ctypes.data_as(ct.POINTER(ct.c_double)),
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
    raise NotImplementedError, "fastsumx2 large mem bug, need to be checked"
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
    _CPPUTIL.fastsumx2(mat.ctypes.data_as(ct.POINTER(ct.c_double)),
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

################################################################################
# fast submatrix operation
################################################################################
_CPPUTIL.submatrix_add.restype = None
_CPPUTIL.submatrix_add.argtypes = [ct.POINTER(ct.c_double),
                                    ct.POINTER(ct.c_int),
                                    ct.POINTER(ct.c_int),
                                    ct.c_int,
                                    ct.c_int,
                                    ct.c_int,
                                    ct.c_int,
                                    ct.POINTER(ct.c_double)]

def submatrix_add(submat, rowid, colid, mat):
    """Adds a submatrix to the original matrix by giving the row and column ids.
    """
    if submat.flags['C_CONTIGUOUS'] != True or submat.dtype != np.float64 \
            or mat.flags['C_CONTIGUOUS'] != True or mat.dtype != np.float64:
        raise ValueError, "Unsupported input matrix."
    rowid = rowid.astype(ct.c_int)
    colid = colid.astype(ct.c_int)
    _CPPUTIL.submatrix_add(submat.ctypes.data_as(ct.POINTER(ct.c_double)),
                            rowid.ctypes.data_as(ct.POINTER(ct.c_int)),
                            colid.ctypes.data_as(ct.POINTER(ct.c_int)),
                            ct.c_int(mat.shape[0]),
                            ct.c_int(mat.shape[1]),
                            ct.c_int(rowid.size),
                            ct.c_int(colid.size),
                            mat.ctypes.data_as(ct.POINTER(ct.c_double))
                            )

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

################################################################################
# clip operation
################################################################################
_CPPUTIL.clip.restype = None
_CPPUTIL.clip.argtype = [ct.POINTER(ct.c_double),
                         ct.c_int,
                         ct.c_double,
                         ct.c_double,
                         ct.c_int]
_CLIP_LOWER = 1
_CLIP_UPPER = 2

def clip(arr, lower = None, upper = None):
    if arr.flags['C_CONTIGUOUS'] == False or arr.dtype != np.float64:
        raise TypeError, "clip cannot be used on such data type."
    # check the mode
    mode = 0
    if lower is not None:
        mode += _CLIP_LOWER
    else:
        lower = 0.
    if upper is not None:
        mode += _CLIP_UPPER
    else:
        upper = 0.
    _CPPUTIL.clip(arr.ctypes.data_as(ct.POINTER(ct.c_double)),
                  ct.c_int(arr.size),
                  ct.c_double(lower),
                  ct.c_double(upper),
                  ct.c_int(mode))
    return

