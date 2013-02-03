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
        
class TestLoss2(unittest.TestCase):
    @staticmethod
    def basicTest(Y, pred, weight, loss1, loss2):
        """Test the loss function with gradient cache (loss2) against its naive
        counterpart (loss1)
        """
        f, g = loss1(Y, pred, None)
        g2 = np.empty_like(pred)
        f2, g2 = loss2(Y, pred, None, g2)
        np.testing.assert_almost_equal(f, f2)
        np.testing.assert_array_almost_equal(g, g2)
        f, g = loss1(Y, pred, weight)
        f2, g2 = loss2(Y, pred, weight, g2)
        np.testing.assert_almost_equal(f, f2)
        np.testing.assert_array_almost_equal(g, g2)

    def testLossL2(self):
        Y = np.random.rand(10,2)
        pred = np.random.rand(10,2)
        weight = np.random.rand(10)
        TestLoss2.basicTest(Y, pred, weight,
                classifier.Loss.loss_l2, classifier.Loss2.loss_l2)
        
    def testLossHinge(self):
        y = np.random.randint(5, size=100)
        while (y.max() < 4):
            y = np.random.randint(5, size=100)
        Y = classifier.to_one_of_k_coding(y)
        pred = np.random.rand(100,5)
        weight = np.random.rand(100)
        TestLoss2.basicTest(Y, pred, weight,
                classifier.Loss.loss_hinge, classifier.Loss2.loss_hinge)
        TestLoss2.basicTest(Y, pred, weight,
                classifier.Loss.loss_squared_hinge,
                classifier.Loss2.loss_squared_hinge)
        TestLoss2.basicTest(Y, pred * 10, weight,
                classifier.Loss.loss_hinge, classifier.Loss2.loss_hinge)
        TestLoss2.basicTest(Y, pred * 10, weight,
                classifier.Loss.loss_squared_hinge,
                classifier.Loss2.loss_squared_hinge)

    def testLossLogistic(self):
        y = np.random.randint(5, size=100)
        while (y.max() < 4):
            y = np.random.randint(5, size=100)
        Y = classifier.to_one_of_k_coding(y)
        pred = np.random.rand(100,5)
        weight = np.random.rand(100)
        TestLoss2.basicTest(Y, pred, weight,
                classifier.Loss.loss_multiclass_logistic,
                classifier.Loss2.loss_multiclass_logistic)
        TestLoss2.basicTest(Y, pred * 10, weight,
                classifier.Loss.loss_multiclass_logistic,
                classifier.Loss2.loss_multiclass_logistic)