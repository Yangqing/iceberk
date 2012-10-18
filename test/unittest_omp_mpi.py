from iceberk import omp_mpi, mpi
import math
import numpy as np
import unittest

class TestOMP(unittest.TestCase):
    """Test the mpi module
    """
    def setUp(self):
        pass

    def test_omp1_predict(self):
        X = np.array([[0.4, 0.1], [0.4, 0.5], [1.0, 0.0]])
        centroids = np.array([[1.0, 0.0], [0.0, 1.0]])
        idx, val = omp_mpi.omp1_predict(X, centroids)
        self.assertTrue(np.all(idx == np.array([0, 1, 0])))
        self.assertTrue(np.all(val == np.array([0.4, 0.5, 1.0])))

    def test_omp1_maximize(self):
        X = np.array([[0.4, 0.1], [0.4, 0.5], [1.0, 0.0]])
        idx = np.array([0, 1, 0])
        val = np.array([0.4, 0.5, 1.0])
        centroids = omp_mpi.omp1_maximize(X, idx, val, 2)
        centroids_groundtruth = np.vstack((X[0]*0.4+X[2]*1.0, X[1]*0.5))
        centroids_groundtruth /= np.sqrt(np.sum(centroids_groundtruth**2, axis=1)).reshape(2, 1)
        #print centroids
        #print centroids_groundtruth
        np.testing.assert_array_almost_equal(centroids, centroids_groundtruth, 8)
        
    def test_omp(self):
        data = np.vstack((np.random.randn(100, 2)+1, \
                          np.random.randn(100, 2)-1))
        k = 20
        centers = omp_mpi.omp1(data, k, max_iter=10)
        self.assertEqual(centers.shape[0], k)
        self.assertEqual(centers.shape[1], data.shape[1])
    
    def test_omp_singlenode(self):
        """ Test a simple OMP where the result is apparent
        """
        data = np.vstack(( np.ones((100, 2)), \
                           np.tile([-1.,1.], (100,1))))
        k = 2
        centers = omp_mpi.omp1(data, k, max_iter=100)
        self.assertEqual(centers.shape[0], k)
        self.assertEqual(centers.shape[1], data.shape[1])
        np.testing.assert_array_almost_equal(np.abs(centers), 1./math.sqrt(2))
        self.assertTrue((centers[0,0] * centers[0,1] > 0) ^ \
                        (centers[1,0] * centers[1,1] > 0))
        
    def test_omp_multinode(self):
        if mpi.SIZE == 1:
            return
        if mpi.RANK % 2:
            data = np.ones((100, 2))
        else:
            data = np.tile([-1.,1.], (100,1))
        k = 2
        centers = omp_mpi.omp1(data, k, max_iter=100)
        self.assertEqual(centers.shape[0], k)
        self.assertEqual(centers.shape[1], data.shape[1])
        np.testing.assert_array_almost_equal(np.abs(centers), 1./math.sqrt(2))
        self.assertTrue((centers[0,0] * centers[0,1] > 0) ^ \
                        (centers[1,0] * centers[1,1] > 0))
        

if __name__ == '__main__':
    unittest.main()

