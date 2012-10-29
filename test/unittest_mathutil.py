from iceberk import mathutil
import numpy as np
import unittest

class TestMathutil(unittest.TestCase):
    """Test the mathutil module
    """
    def setUp(self):
        self.test_matrices = [
            (np.random.rand(2,3), 
             np.random.rand(3,4)),
            (np.array(np.random.rand(4,5), order='f'),
             np.array(np.random.rand(5,2))),
            (np.array(np.random.rand(3,5)),
             np.array(np.random.rand(5,4), order='f')),
            (np.array(np.random.rand(5,3), order='f'),
             np.array(np.random.rand(3,6), order='f'))]
        self.test_matrices += \
            [(b.T, a.T) for a,b in self.test_matrices]
        self.test_matrices += \
            [(np.array(a, dtype=np.float32), np.array(b, dtype=np.float32))
                for a,b in self.test_matrices]
        self.test_matrices += \
            [(np.array(a, dtype=np.float32), b)
                for a,b in self.test_matrices]
        self.test_matrices += \
            [(a, np.array(b, dtype=np.float32))
                for a,b in self.test_matrices]
        pass

    def testgemm(self):
        for A, B in self.test_matrices:
            result = mathutil.gemm(1., A, B)
            result_ref = np.dot(A,B)
            self.assertTrue(result.flags['C_CONTIGUOUS'])
            np.testing.assert_array_almost_equal(result, result_ref)

    def testgemm_scale(self):
        for A, B in self.test_matrices:
            result = mathutil.gemm(2., A, B)
            result_ref = np.dot(A,B) * 2.
            self.assertTrue(result.flags['C_CONTIGUOUS'])
            np.testing.assert_array_almost_equal(result, result_ref)

    def testgemm_with_out(self):
        for A, B in self.test_matrices:
            result_ref = np.dot(A,B)
            result = np.empty(result_ref.shape, dtype = A.dtype)
            mathutil.gemm(1., A, B, out=result)
            self.assertTrue(result.flags['C_CONTIGUOUS'])
            np.testing.assert_array_almost_equal(result, result_ref)

    def testgemm_scale(self):
        for A, B in self.test_matrices:
            result_ref = np.dot(A,B) * 2.
            result = np.empty(result_ref.shape, dtype = A.dtype)
            mathutil.gemm(2., A, B, out=result)
            self.assertTrue(result.flags['C_CONTIGUOUS'])
            np.testing.assert_array_almost_equal(result, result_ref)

    def testdot(self):
        for A, B in self.test_matrices:
            result = mathutil.dot(A, B)
            result_ref = np.dot(A,B)
            self.assertTrue(result.flags['C_CONTIGUOUS'])
            np.testing.assert_array_almost_equal(result, result_ref)
    
    def testdot_image(self):
        A = np.random.rand(2,3,4)
        B = np.random.rand(4,5)
        result = mathutil.dot_image(A, B)
        self.assertTrue(result.flags['C_CONTIGUOUS'])
        self.assertEqual(result.shape[:-1], A.shape[:-1])
        self.assertEqual(result.shape[-1], B.shape[-1])
        
        B = np.array(B, order='f')
        result = mathutil.dot_image(A, B)
        self.assertTrue(result.flags['C_CONTIGUOUS'])
        self.assertEqual(result.shape[:-1], A.shape[:-1])
        self.assertEqual(result.shape[-1], B.shape[-1])
    
    def testreservoir_sampler(self):
        # test size
        sampler = mathutil.ReservoirSampler(100)
        sampler.consider(np.random.rand(40,10))
        self.assertEqual(sampler.get().shape, (40, 10))
        sampler.consider(np.random.rand(40,10))
        self.assertEqual(sampler.get().shape, (80,10))
        sampler.consider(np.random.rand(40,10))
        self.assertEqual(sampler.get().shape, (100,10))
        # test if replacing does work
        sampler = mathutil.ReservoirSampler(10)
        sampler.consider(np.zeros((10,10)))
        np.testing.assert_array_equal(sampler.get(), 0)
        # with very high probabibility, one sample will be nonzero
        sampler.consider(np.ones((100,10)))
        self.assertGreater(sampler.get().sum(), 0)
        
        
if __name__ == '__main__':
    unittest.main()

