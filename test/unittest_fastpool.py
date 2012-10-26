import ctypes as ct
import numpy as np
import os
import unittest
import iceberk as ice

from iceberk import cpputil

class TestFastpool(unittest.TestCase):
    """Test the mpi module
    """
    
    def testPoolingSingle(self):
        data = np.random.rand(10,10,3)
        data_2d = data.reshape((100,3))
        pooled_max = data_2d.max(axis=0)
        pooled_ave = data_2d.mean(axis=0)
        pooled_rms = np.sqrt((data_2d**2).mean(axis=0))
        np.testing.assert_almost_equal(pooled_max,
                cpputil.fastpooling(data, (1,1), 'max').flatten())
        np.testing.assert_almost_equal(pooled_ave,
                cpputil.fastpooling(data, (1,1), 'ave').flatten())
        np.testing.assert_almost_equal(pooled_rms,
                cpputil.fastpooling(data, (1,1), 'rms').flatten())
        
    def testPoolingMultiple(self):
        height = 10
        width = 10
        data = np.random.rand(height, width, 3)
        data_2d = data.reshape(100,3)
        grids = [(2,2), (3,3), (2,3), (4,4), (2,4), (5,5)]
        for grid in grids:
            index = np.zeros((height, width), dtype=int)
            for i in range(height):
                for j in range(width):
                    h_id = i * grid[0] / height;
                    w_id = j * grid[1] / width;
                    index[i,j] = h_id * grid[1] + w_id
            index = index.flatten()
            num_pool = grid[0]*grid[1]
            pooled_max = np.hstack(
                    [data_2d[index==i].max(axis=0) for i in range(num_pool)])
            pooled_ave = np.hstack(
                    [data_2d[index==i].mean(axis=0) for i in range(num_pool)])
            pooled_rms = np.hstack(
                    [(data_2d[index==i]**2).mean(axis=0) \
                     for i in range(num_pool)])
            pooled_rms = np.sqrt(pooled_rms)
            np.testing.assert_almost_equal(pooled_max,
                    cpputil.fastpooling(data, grid, 'max').flatten())
            np.testing.assert_almost_equal(pooled_ave,
                    cpputil.fastpooling(data, grid, 'ave').flatten())
            np.testing.assert_almost_equal(pooled_rms,
                    cpputil.fastpooling(data, grid, 'rms').flatten())

    def testPoolingShapes(self):
        heights = [16, 31, 32, 33, 37, 40]
        widths = [16, 31, 32, 33, 37, 40]
        channels = [1, 3, 5, 10]
        grids = [(2, 2), (3, 3), (2, 3), (4, 4), (2, 4), (5, 5)]
        for height in heights:
            for width in widths:
                for channel in channels:
                    data = np.random.rand(height, width, channel)
                    for grid in grids:
                        shape = cpputil.fastpooling(data, grid, 'max').shape
                        self.assertEqual(shape, grid + (channel,))

if __name__ == '__main__':
    unittest.main()
