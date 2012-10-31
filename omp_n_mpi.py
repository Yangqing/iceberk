""" orthogonal matching pursuit training and prediction code.
"""

from iceberk import mpi, mathutil, pipeline
import logging
import numpy as np

# minibatch is used to avoid excessive memory consumption
_MINIBATCH = 1000

def omp_n_predict(X, centroids, num_active, reg = 1e-8):
    ''' omp prediction
    Input:
        X: the data matrix
        centroids: the centroids matrix
        num_active: the number of active components
        reg: the regularization term for least square, default 1e-8
    '''
    idx = np.empty((X.shape[0], num_active), dtype=np.int)
    val = np.zeros((X.shape[0], num_active))
    C = mathutil.dot(centroids, centroids.T)
    # find the most active k
    dots = None
    dots_abs = None
    for start in range(0, X.shape[0], _MINIBATCH):
        end = min(start+_MINIBATCH, X.shape[0])
        batchsize = end - start
        if dots is None:
            dots = mathutil.dot(X[start:end], centroids.T)
            dots_abs = np.empty_like(dots)
        else:
            mathutil.dot(X[start:end], centroids.T, out=dots[:batchsize])
        # we only compute the dots once, and keep them for future use
        for i in range(num_active):
            np.abs(dots, out=dots_abs)
            idx[start:end, i] = np.argmax(dots_abs[:batchsize], axis=1)
            val[start:end, i] = dots[np.arange(batchsize), idx[start:end, i]]
            # remove the effect from dots
            dots[:batchsize] -= C[idx[start:end,i]] * val[start:end,i][:, np.newaxis]
    return idx, val

def omp_n_maximize(X, centroids_old, labels, val, k):
    '''Maximization of omp_n, with the given labels and vals using 
        D = (A'A)^-1 A'X
    
    Note that X is the local data hosted in each MPI node.
    '''
    dim = X.shape[1]
    # G is the gram matrix of the activations
    AtA_local = np.zeros((k,k))
    AtX_local = np.zeros((k, dim))
    
    A = None
    for start in range(0, X.shape[0], _MINIBATCH):
        end = min(start+_MINIBATCH, X.shape[0])
        batchsize = end - start
        if A is None:
            A = np.zeros((batchsize, k))
        else:
            A[:] = 0
        for i in range(batchsize):
            A[i, labels[start+i]] = val[start+i]
        AtA_local += mathutil.dot(A.T, A)
        AtX_local += mathutil.dot(A.T, X[start:end])
    """
    for i in range(X.shape[0]):
        cpputil.submatrix_add(val[i][:, np.newaxis] * val[i], 
                              labels[i], labels[i], AtA_local)
        AtX_local[labels[i]] += val[i][:, np.newaxis] * X[labels[i]]
    """
    AtA = np.empty_like(AtA_local)
    AtX = np.empty_like(AtX_local)
    mpi.COMM.Allreduce(AtA_local, AtA)
    mpi.COMM.Allreduce(AtX_local, AtX)
    # add a regularization term
    isempty = (np.diag(AtA) == 0)
    AtA.flat[::k+1] += 1e-8
    centroids = np.ascontiguousarray(np.linalg.solve(AtA, AtX))
    # let's deal with inactive guys
    for i in range(k):
        if isempty[i]:
            # randomly restart one
            centroids[i] = X[np.random.randint(X.shape[0])]
            mpi.COMM.Bcast(centroids[i], root = mpi.elect())
    scale = np.sqrt((centroids ** 2).sum(1)) + np.finfo(np.float64).eps
    centroids /= scale[:, np.newaxis]
    return centroids

def omp_n(X, k, num_active, max_iter=100, tol=1e-4):
    '''OMP training with MPI
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
        logging.debug("OMPN iteration %d" % (iter_id,))
        centroids_old = centroids.copy()
        labels, val = omp_n_predict(X, centroids, num_active)
        centroids = omp_n_maximize(X, centroids_old, labels, val, k)
        # check convergence on root
        if mpi.is_root():
            converged = np.sum((centroids_old - centroids) ** 2) < tol * vdata
        else:
            converged = None
        converged = mpi.COMM.bcast(converged)
        if converged:
            logging.debug("OMPN has converged.")
            break
    return centroids

    
if __name__ == "__main__":
    pass
    ## do some test
    #patches = np.load('cifar_patches_whitened.npy')
    #dictionary = omp_n(patches, 400, 10, 10)