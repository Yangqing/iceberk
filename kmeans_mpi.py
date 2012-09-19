"""kmeans_mpi is a simple implementation of kmeans
It is adjusted from scikits.learn so that we can carry out
kmeans under MPI.

The original copyright info is as follows.

# Authors: Gael Varoquaux <gael.varoquaux@normalesup.org>
#          Thomas Rueckstiess <ruecksti@in.tum.de>
#          James Bergstra <james.bergstra@umontreal.ca>
#          Jan Schlueter <scikit-learn@jan-schlueter.de>
#          Nelle Varoquaux
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
# License: BSD
"""

from jiayq_ice import mpi
import numpy as np
from sklearn import metrics


def kmeans(X, k, n_init=1, max_iter=300, tol=1e-4):
    """ K-means clustering algorithm.

    Parameters
    ----------
    X: ndarray
        A M by N array of M observations in N dimensions. X in every MPI node
        is the local data points it is responsible for.

    k: int or ndarray
        The number of clusters to form.

    n_init: int, optional, default: 1
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.

    max_iter: int, optional, default 300
        Maximum number of iterations of the k-means algorithm to run.

    tol: float, optional
        The relative increment in the results before declaring convergence.

    Returns
    -------
    centroid: ndarray
        A k by N array of centroids found at the last iteration of
        k-means.

    label: ndarray
        label[i] is the code or index of the centroid the
        i'th observation is closest to.

    inertia: float
        The final value of the inertia criterion

    """
    # do k-means training
    # vdata helps the stop criterion
    vdata = mpi.COMM.allreduce(np.mean(np.var(X, 0))) / mpi.SIZE
    best_inertia = np.infty

    # pre-compute squared norms of data points
    x_squared_norms = (X**2).sum(axis=1)
    for _ in range(n_init):
        # initialization
        centers = X[np.random.randint(X.shape[0], size = k)]
        centers_all = mpi.COMM.gather(centers)
        if mpi.is_root():
            centers_all = np.vstack(centers_all)
            centers[:] = centers_all[
                    np.random.permutation(centers_all.shape[0])[:k]]
        mpi.COMM.Bcast(centers)
        
        # iterations
        for _ in range(max_iter):
            centers_old = centers.copy()
            labels, inertia = _e_step(X, centers,
                                      x_squared_norms=x_squared_norms)
            inertia = mpi.COMM.allreduce(inertia)
            centers = _m_step(X, labels, k)
            # test convergence
            converged = (np.sum((centers_old - centers) ** 2) < tol * vdata)
            if mpi.agree(converged):
                break

        if inertia < best_inertia:
            best_labels = labels.copy()
            best_centers = centers.copy()
            best_inertia = inertia
    return best_centers, best_labels, best_inertia

def kmeans_predict(X, centers):
    """Does k-means prediction

    It is essentially a wrapper of the local e_step method

    Parameters
    ----------
    X: array, shape (n_samples, n_features)

    centers: array, shape (k, n_features)
        The cluster centers

    Returns
    -------z: array of shape(n)
        The resulting assignment

    inertia: float
        The value of the inertia criterion with the assignment
    """
    return _e_step(X, centers)

def _m_step(X, z, k):
    """M step of the K-means EM algorithm

    Computation of cluster centers/means

    Parameters
    ----------
    X: array, shape (n_samples, n_features)

    z: array, shape (n_samples)
        Current assignment

    k: int
        Number of desired clusters

    Returns
    -------
    centers: array, shape (k, n_features)
        The resulting centers
    """
    dim = X.shape[1]
    centers_local = np.zeros((k, dim))
    counts_local = np.zeros(k, dtype = int)
    centers = np.zeros((k, dim))
    counts = np.zeros(k, dtype = int)
    for q in range(k):
        center_mask = np.flatnonzero(z == q)
        counts_local[q] = len(center_mask)
        if counts_local[q] > 0:
            centers_local[q] = X[center_mask].sum(axis=0)
    mpi.COMM.Allreduce(counts_local, counts)
    mpi.COMM.Allreduce(centers_local, centers)
    for q in range(k):
        if counts[q] == 0:
            centers[q] = X[np.random.randint(X.shape[0])]
            mpi.COMM.Bcast(centers[q], root=mpi.elect())
            counts[q] = 1
    centers /= counts.reshape((centers.shape[0], 1))
    return centers

def _e_step(X, centers, x_squared_norms=None):
    """E step of the K-means EM algorithm

    Computation of the input-to-cluster assignment

    Parameters
    ----------
    X: array, shape (n_samples, n_features)

    centers: array, shape (k, n_features)
        The cluster centers

    x_squared_norms: array, shape (n_samples,), optional
        Squared euclidean norm of each data point, speeds up computations in
        case of precompute_distances == True. Default: None

    Returns
    -------z: array of shape(n)
        The resulting assignment

    inertia: float
        The value of the inertia criterion with the assignment
    """
    n_samples = X.shape[0]
    minibatch = 1000
    minid = np.empty(n_samples, dtype=np.int)
    inertia = 0.0
    if x_squared_norms is None:
        x_squared_norms = np.sum(X**2, axis=1)
    for start in range(0, n_samples, minibatch):
        end = min(n_samples, start + minibatch)
        distances = metrics.euclidean_distances(
                centers, X[start:end], x_squared_norms[start:end], squared=True)
        minid[start:end] = np.argmin(distances, axis=0)
        inertia += np.sum(distances[minid[start:end], range(end - start)]) 
    return minid, inertia

def demo_kmeans():
    """A simple kmeans demo
    """
    print 'Running kmeans demo'
    data = np.vstack((np.random.randn(500,2)+1,\
                      np.random.randn(500,2)-1))
    centers, labels, inertia = kmeans(data, 8, 
                                         n_init=1, 
                                         max_iter=5)
    print 'inertia =', inertia
    print 'centers = \n', centers
    try:
        from matplotlib import pyplot
        if mpi.is_root():
            pyplot.scatter(data[:,0],data[:,1],c=labels)
            pyplot.show()
        mpi.barrier()
    except Exception:
        print 'cannot show figure. will simply pass'
        pass

if __name__ == "__main__":
    demo_kmeans()