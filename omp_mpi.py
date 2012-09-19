""" orthogonal matching pursuit training and prediction code.
"""

from jiayq_ice import mpi
import numpy as np

# minibatch is used to avoid excessive memory consumption
_MINIBATCH = 1000


def omp1_predict(X, centroids):
    ''' omp1 prediction
    
    This function does one-dimensional orthogonal matching pursuit.
    the returned values are simply going to be the indices and 
    inner products.
    '''
    idx = np.empty(X.shape[0], dtype=np.int)
    val = np.empty(X.shape[0])
    for start in range(0, X.shape[0], _MINIBATCH):
        end = min(start+_MINIBATCH, X.shape[0])
        dots = np.dot(X[start:end], centroids.T)
        idx[start:end] = np.argmax(np.abs(dots), axis=1)
        val[start:end] = dots[range(end-start), idx[start:end]]
    return idx, val


def omp1(X, k, max_iter=100, tol=1e-4):
    '''omp1 training with MPI
    
    Note that the X matrix passed should be the local data each node is hosting.
    '''
    # vdata is used for testing convergence
    Nlocal = X.shape[0]
    vdatalocal = np.sum(np.var(X, 0))
    N = mpi.COMM.allreduce(Nlocal)
    vdata = mpi.COMM.allreduce(vdatalocal)
    vdata /= N
    # random initialization
    centroids = np.random.randn(k, X.shape[1])
    centroids /= np.sqrt(np.sum(centroids**2, axis=1)).reshape(k, 1)
    centroids_all = mpi.COMM.gather(centroids)
    # make sure we are using the same centroids on all nodes
    if mpi.is_root():
        centroids_all = np.vstack(centroids_all)
        centroids[:] = centroids_all[\
                np.random.permutation(centroids_all.shape[0])[:k]]
    mpi.COMM.Bcast(centroids, root=0)

    for _ in range(max_iter):
        centroids_old = centroids.copy()
        labels, val = omp1_predict(X, centroids)
        centroids = omp1_maximize(X, labels, val, k)
        # check convergence on root
        if mpi.is_root():
            converged = np.sum((centroids_old - centroids) ** 2) < tol * vdata
        else:
            converged = None
        converged = mpi.COMM.bcast(converged)
        if converged:
            break
    return centroids

def omp1_maximize(X, labels, val, k):
    '''
    The MPI version of omp1_maximize
    Note that X is the local data hosted in each MPI node.
    '''
    dim = X.shape[1]
    centroids_local = np.zeros((k, dim))
    centroids_local_nonempty = np.zeros(k, dtype = np.int)
    # loop over the classes
    for q in range(k):
        center_mask = (labels == q)
        if np.any(center_mask):
            centroids_local[q] = np.dot(val[center_mask], X[center_mask])
            centroids_local_nonempty[q] = 1
    centroids_nonempty = np.zeros(k, dtype=np.int)
    mpi.COMM.Allreduce(centroids_local_nonempty, centroids_nonempty)
    # now, for those empty centroids, we need to randomly restart them
    for q in range(k):
        if centroids_nonempty[q] == 0 and mpi.is_president():
            centroids_local[q] = X[np.random.randint(X.shape[0])]
    # collect all centroids
    centroids = np.zeros((k, dim))
    mpi.COMM.Reduce(centroids_local, centroids)
    centroids /= (np.sqrt(np.sum(centroids**2,axis=1))+np.finfo(np.float64).eps).reshape(k,1)
    # broadcast to remove any numerical unstability
    mpi.COMM.Bcast(centroids)
    return centroids

