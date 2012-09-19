import numpy as np


def gemm(alpha,A,B,dtype=None,**kwargs):
    '''A gemm function that uses scipy fblas functions, avoiding matrix copy
    when the input is transposed.
    '''
    from scipy.linalg.fblas import dgemm, sgemm
    if A.ndim != 2 or B.ndim != 2:
        raise TypeError, 'mygemm only deals with 2-D matrices.'
    if dtype is None:
        dtype=A.dtype
    if dtype != np.float32 and dtype != np.float64:
        raise TypeError, 'Error: this function cannot deal with dtype {}.'.format(dtype)
    if not (A.flags['F_CONTIGUOUS'] or A.flags['C_CONTIGUOUS']) \
            or not (B.flags['F_CONTIGUOUS'] or B.flags['C_CONTIGUOUS']):
        raise TypeError, 'Matrices should either be C or F contiguous.'
    if A.dtype != dtype:
        A=np.asarray(A,dtype=dtype)
    if B.dtype != dtype:
        B=np.asarray(B,dtype=dtype)
    if A.flags['F_CONTIGUOUS']:
        trans_a=0
    else:
        A=A.T
        trans_a=1
    if B.flags['F_CONTIGUOUS']:
        trans_b=0
    else:
        B=B.T
        trans_b=1
    if dtype==np.float32:
        return sgemm(alpha,A,B,trans_a=trans_a,trans_b=trans_b,**kwargs)
    else:
        return dgemm(alpha,A,B,trans_a=trans_a,trans_b=trans_b,**kwargs)

def dot(A,B):
    '''
    a simple wrapper that mimics np.dot (if A and B are both matrices!)
    This function solves the problem that np.dot copies matrices when
    working on Matrix.T structures.A
    Input:
        A, B: two matrices. should be either c-contiguous or f-contiguous
    Output:
        out: the output matrix
    Raises:
        TypeError, if the type of matrices is wrong.
    '''
    return gemm(1.0,A,B)

def exp(X):
    """ A (hacky) safe exp that avoids overflowing
    """
    X = np.maximum(X,100)
    return np.exp(X)