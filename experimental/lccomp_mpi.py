""" orthogonal matching pursuit training and prediction code.
"""

from iceberk import mpi, mathutil
import logging
import numpy as np

# minibatch is used to avoid excessive memory consumption
_MINIBATCH = 1000


def lccomp_predict(X, centroids, num_active, reg = 1e-8):
    ''' omp prediction
    Input:
        X: the data matrix
        centroids: the centroids matrix
        num_active: the number of active components
        reg: the regularization term for least square, default 1e-8
    '''
    idx = np.empty((X.shape[0], num_active), dtype=np.int)
    val = np.zeros((X.shape[0], num_active))
    # find the most active k
    for start in range(0, X.shape[0], _MINIBATCH):
        end = min(start+_MINIBATCH, X.shape[0])
        dots = np.dot(X[start:end], centroids.T)
        idx[start:end] = np.argsort(dots, axis=1)[:, -num_active:]
    # perform LCC-like local least square reconstruction
    Z = np.empty((num_active, centroids.shape[1]))
    for i in range(X.shape[0]):
        Z[:] = centroids[idx[i]]
        # local covariance
        C = mathutil.dot(Z, Z.T)
        # add regularization
        C.flat[::num_active+1] += reg * C.trace()
        val[idx[i]] = np.linalg.solve(C,np.dot(Z, X[i]))
    return idx, val

def lccomp_maximize(X, centroids_old, labels, val, k):
    '''
    The MPI version of lccomp_maximize
    Note that X is the local data hosted in each MPI node.
    '''
    dim = X.shape[1]
    num_active = labels.shape[1]
    centroids_local = np.zeros((k, dim))
    # compute the reconstruction, note that what we are computing are actually
    # the residuals
    Z = np.empty((num_active, dim))
    for i in range(X.shape[0]):
        Z[:] = centroids_old[labels[i]] * val[i][:, np.newaxis]
        # add the difference
        Z += X[i] - Z.sum(axis=0)
        # accumulate the values
        centroids_local[labels[i]] += Z
    centroids_local_nonempty = (np.abs(centroids_local).sum(axis=1) > 0)\
                                .astype(int)
    centroids_nonempty = np.empty_like(centroids_local_nonempty)
    mpi.COMM.Allreduce(centroids_local_nonempty, centroids_nonempty)
    # now, for those empty centroids, we need to randomly restart them
    for q in range(k):
        if centroids_nonempty[q] == 0 and mpi.is_president():
            centroids_local[q] = X[np.random.randint(X.shape[0])]
    # collect all centroids
    centroids = np.empty_like(centroids_local)
    mpi.COMM.Reduce(centroids_local, centroids)
    scale = np.sqrt((centroids ** 2).sum(1)) + np.finfo(np.float64).eps
    centroids /= scale[:, np.newaxis]
    return centroids

def lccomp(X, k, num_active, max_iter=100, tol=1e-4):
    '''lccomp training with MPI
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

    for iter_id in range(max_iter):
        logging.debug("OMP iteration %d" % (iter_id,))
        centroids_old = centroids.copy()
        labels, val = lccomp_predict(X, centroids, num_active)
        centroids = lccomp_maximize(X, centroids_old, labels, val, k)
        # check convergence on root
        if mpi.is_root():
            converged = np.sum((centroids_old - centroids) ** 2) < tol * vdata
        else:
            converged = None
        converged = mpi.COMM.bcast(converged)
        if converged:
            logging.debug("OMP has converged.")
            break
    return centroids
