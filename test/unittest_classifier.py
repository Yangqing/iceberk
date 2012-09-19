from jiayq_ice import classifier
import numpy as np
import unittest


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