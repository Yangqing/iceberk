import cPickle as pickle
from jiayq_ice import mpi
import numpy as np
import unittest

_MPI_TEST_DIR = '/tmp/mpi_test_dir'
_MPI_DUMP_TEST_FILE = '/tmp/jiayq_ice.test.unittest_mpi.dump.pickle'

class TestMPI(unittest.TestCase):
    """Test the mpi module
    """
    def setUp(self):
        pass

    def testBasic(self):
        self.assertIsNotNone(mpi.COMM)
        
    def testMkdir(self):
        import os
        mpi.mkdir(_MPI_TEST_DIR)
        self.assertTrue(os.path.exists(_MPI_TEST_DIR))
        
    def testAgree(self):
        self.assertTrue(mpi.agree(True))
        self.assertFalse(mpi.agree(False))
        self.assertTrue(mpi.agree(mpi.RANK == 0))
        self.assertFalse(mpi.agree(mpi.RANK != 0))
        self.assertFalse(mpi.agree(mpi.RANK))
    
    def testElect(self):
        result = mpi.elect()
        self.assertLess(result, mpi.SIZE)
        all_results = mpi.COMM.allgather(result)
        self.assertEqual(len(set(all_results)), 1)
        num_presidents = mpi.COMM.allreduce(mpi.is_president())
        self.assertEqual(num_presidents, 1)
        
    def testIsRoot(self):
        if mpi.RANK == 0:
            self.assertTrue(mpi.is_root())
        else:
            self.assertFalse(mpi.is_root())
    
    def testBarrier(self):
        import time
        # sleep for a while, and resume
        time.sleep(mpi.RANK)
        mpi.barrier()
        self.assertTrue(True)
    
    def testDistribute(self):
        data_list = [np.ones(100), np.ones((100,2)), np.ones((100,2,3))]
        for data in data_list:
            distributed = mpi.distribute(data)
            self.assertTrue(isinstance(distributed, np.ndarray))
            np.testing.assert_array_almost_equal(distributed,
                                                 np.ones(distributed.shape),
                                                 8)
            total_number = mpi.COMM.allreduce(distributed.shape[0])
            self.assertEqual(total_number, data.shape[0])
    
    def testDistributeList(self):
        lengths = range(5)
        for length in lengths:
            source = range(length) * mpi.SIZE
            result = mpi.distribute_list(source)
            self.assertEqual(len(result), length)
            for i in range(length):
                self.assertEqual(result[i],i)
    
    def testDumpLoad(self):
        local_size = 2
        mat_sources = [np.random.rand(local_size),
                       np.random.rand(local_size,2),
                       np.random.rand(local_size, 2,3)]
        for mat in mat_sources:
            mpi.dump_matrix(mat, _MPI_DUMP_TEST_FILE)
            if mpi.is_root():
                mat_dumped = pickle.load(open(_MPI_DUMP_TEST_FILE,'r'))
                self.assertEqual(mat_dumped.shape,
                                 (local_size * mpi.SIZE,) + mat.shape[1:])
            mat_read = mpi.load_matrix(_MPI_DUMP_TEST_FILE)
            self.assertEqual(mat.shape, mat_read.shape)
        
if __name__ == '__main__':
    unittest.main()

