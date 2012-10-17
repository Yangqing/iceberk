from iceberk import kmeans_mpi, mpi
import numpy as np
import unittest

class TestKmeans(unittest.TestCase):
    """Test the mpi module
    """
    def setUp(self):
        pass

    def test_kmeans_predict(self):
        n = 10
        X = np.vstack((np.random.rand(n, 2),
                       -np.random.rand(n, 2)))
        centers= np.array([[1,1],
                           [-1,-1]])
        ground_truth = np.array([0]*n + [1]*n)
        predict, inertia = kmeans_mpi.kmeans_predict(X, centers)
        self.assertGreater(inertia, 0.)
        np.testing.assert_array_equal(ground_truth, predict)

    def test_kmeans_simple(self):
        n = 10
        X = np.vstack((np.random.rand(n, 2),
                       -np.random.rand(n, 2)))
        centers, labels, inertia = kmeans_mpi.kmeans(X, 3)
        self.assertEqual(centers.shape, (3,2))
        np.testing.assert_array_less(labels, 3)
        self.assertGreater(inertia, 0.)
        centers, labels, inertia = kmeans_mpi.kmeans(X, 3, n_init = 5)
        self.assertEqual(centers.shape, (3,2))
        np.testing.assert_array_less(labels, 3)
        self.assertGreater(inertia, 0.)
        
    def test_simple_m_step_mpi(self):
        X = np.ones((5,2)) * mpi.RANK
        labels = np.ones(5, dtype=int) * mpi.RANK
        centers_groundtruth = np.tile(np.arange(mpi.SIZE), (2, 1)).T
        centers = kmeans_mpi._m_step(X, labels, mpi.SIZE)
        self.assertEqual(centers.shape, (mpi.SIZE, 2))
        np.testing.assert_array_almost_equal(centers, centers_groundtruth)
        
if __name__ == '__main__':
    unittest.main()

