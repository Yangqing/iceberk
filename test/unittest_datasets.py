from jiayq_ice import datasets, mpi, visiondata
import numpy as np
import os
import unittest

#_CIFAR10_FOLDER = '/Users/jiayq/Research/datasets/cifar-10-batches-py'
#_CIFAR100_FOLDER = '/Users/jiayq/Research/datasets/cifar-100-python'
_CIFAR10_FOLDER = '/disabled'
_CIFAR100_FOLDER = '/disabled'


class TestDatasetBasic(unittest.TestCase):
    """Test the basic dataset structures
    """
    def setUp(self):
        pass

    def testImageSet(self):
        self.assertIsNotNone(datasets.ImageSet())

    def testNdarraySet(self):
        input_array = np.random.rand(2,3,4,5)
        dataset = datasets.NdarraySet(input_array)
        self.assertEqual(dataset.size(), 2)
        self.assertEqual(dataset.dim(), (3,4,5))
        self.assertEqual(dataset.num_channels(), 5)
        self.assertEqual(dataset.size_total(), 2 * mpi.SIZE)
        for i in range(dataset.size()):
            np.testing.assert_array_equal(dataset.image(i), input_array[i])


class TestCifarDataset(unittest.TestCase):
    """Test the cifar dataset
    """
    def setUp(self):
        pass
    
    def testCifar10(self):
        if not os.path.exists(_CIFAR10_FOLDER):
            print 'Cifar 10 data not found, skipped'
            return
        cifar = visiondata.CifarDataset(_CIFAR10_FOLDER, is_training = True)
        self.assertEqual(cifar.size_total(), 50000)
        self.assertGreater(cifar.size(), 0)
        self.assertEqual(cifar.dim(), (32, 32, 3))
        self.assertEqual(cifar.num_channels(), 3)
        for i in range(cifar.size()):
            self.assertEqual(cifar.image(i).shape, (32, 32, 3))
            self.assertGreaterEqual(cifar.label(i), 0)
            self.assertLess(cifar.label(i), 10)
        cifar = visiondata.CifarDataset(_CIFAR10_FOLDER, is_training = False)
        self.assertEqual(cifar.size_total(), 10000)
        self.assertGreater(cifar.size(), 0)
        self.assertEqual(cifar.dim(), (32, 32, 3))
        self.assertEqual(cifar.num_channels(), 3)
        for i in range(cifar.size()):
            self.assertEqual(cifar.image(i).shape, (32, 32, 3))
            self.assertGreaterEqual(cifar.label(i), 0)
            self.assertLess(cifar.label(i), 10)

    def testCifar100(self):
        if not os.path.exists(_CIFAR100_FOLDER):
            print 'Cifar 100 data not found, skipped'
            return
        cifar = visiondata.CifarDataset(_CIFAR100_FOLDER, is_training = True)
        self.assertEqual(cifar.size_total(), 50000)
        self.assertGreater(cifar.size(), 0)
        self.assertEqual(cifar.dim(), (32, 32, 3))
        self.assertEqual(cifar.num_channels(), 3)
        for i in range(cifar.size()):
            self.assertEqual(cifar.image(i).shape, (32, 32, 3))
            self.assertGreaterEqual(cifar.label(i), 0)
            self.assertLess(cifar.label(i), 100)
        cifar = visiondata.CifarDataset(_CIFAR100_FOLDER, is_training = False)
        self.assertEqual(cifar.size_total(), 10000)
        self.assertGreater(cifar.size(), 0)
        self.assertEqual(cifar.dim(), (32, 32, 3))
        self.assertEqual(cifar.num_channels(), 3)
        for i in range(cifar.size()):
            self.assertEqual(cifar.image(i).shape, (32, 32, 3))
            self.assertGreaterEqual(cifar.label(i), 0)
            self.assertLess(cifar.label(i), 100)


class TestTwoLayerDataset(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(os.path.dirname(__file__),
                                 'data', 'twolayer')
        
    def testNonPrefetch(self):
        data = datasets.TwoLayerDataset(self.path, ['png'])
        self.assertFalse(data.dim())
        self.assertEqual(data.size_total(), 9)
        self.assertEqual(len(data.image(0).shape), 3)
        self.assertEqual(data.image(0).shape[-1], 3)
        self.assertEqual(data.num_channels(), 3)
    
    def testPrefetch(self):
        data = datasets.TwoLayerDataset(self.path, ['png'], prefetch=True)
        self.assertFalse(data.dim())
        self.assertEqual(data.size_total(), 9)
        self.assertEqual(len(data.image(0).shape), 3)
        self.assertEqual(data.image(0).shape[-1], 3)
        self.assertEqual(data.num_channels(), 3)
    
    def testTargetSize(self):
        target_size = (32,32)
        data = datasets.TwoLayerDataset(self.path, ['png'], prefetch=True,
                                        target_size = target_size)
        self.assertEqual(data.dim(), target_size + (3,))
        self.assertEqual(data.size_total(), 9)
        self.assertEqual(len(data.image(0).shape), 3)
        for i in range(data.size()):
            self.assertEqual(data.image(i).shape, target_size + (3,))
        self.assertEqual(data.num_channels(), 3)

if __name__ == '__main__':
    unittest.main()

            