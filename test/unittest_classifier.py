from iceberk import classifier, mpi
import numpy as np
import unittest

class TestFeatureMeanStd(unittest.TestCase):
    def testFeatureMeanStd(self):
        mat = np.random.rand(100,50)
        m_test, std_test = classifier.feature_meanstd(mat)
        # use the naive approach to compute the mean and std
        mats = mpi.COMM.gather(mat)
        if mpi.is_root():
            mats = np.vstack(mats)
            m = mats.mean(0)
            std = mats.std(0)
        else:
            m = None
            std = None
        m = mpi.COMM.bcast(m)
        std = mpi.COMM.bcast(std)
        np.testing.assert_array_almost_equal(m, m_test)
        np.testing.assert_array_almost_equal(std, std_test)
        
class TestLoss(unittest.TestCase):
    def testLossRankHinge(self):
        weight = None
        Y = np.asarray([0])
        pred = np.asarray([[2,0,0,0]])
        f, g = classifier.Loss.loss_rank_hinge(Y, pred, weight)
        self.assertAlmostEqual(f, 0.)
        self.assertEqual(g.shape, (1,4))
        np.testing.assert_array_almost_equal(g, 0.)
        pred = np.asarray([[2,2,0,0]])
        f, g = classifier.Loss.loss_rank_hinge(Y, pred, weight)
        self.assertAlmostEqual(f, 1.)
        self.assertEqual(g.shape, (1,4))
        np.testing.assert_array_almost_equal(g,
                                             np.asarray([[-1., 1., 0., 0.]]))
        pred = np.asarray([[0,0,0,0]])
        f, g = classifier.Loss.loss_rank_hinge(Y, pred, weight)
        self.assertAlmostEqual(f, 3.)
        self.assertEqual(g.shape, (1,4))
        np.testing.assert_array_almost_equal(g,
                                             np.asarray([[-3., 1., 1., 1.]]))