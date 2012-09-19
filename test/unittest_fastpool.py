import ctypes as ct
import numpy as np
import os
import unittest
import jiayq_ice as ice


class TestFastpool(unittest.TestCase):
    """Test the mpi module
    """
    def setUp(self):
        self.methods = {'max':0, 'ave': 1, 'rms': 2}
        # fast pooling C library
        self.fp = np.ctypeslib.load_library('libfastpool.so',
                                            os.path.dirname(ice.__file__))
        self.fp.fastpooling.restype = ct.c_int
        self.fp.fastpooling.argtypes = [ct.POINTER(ct.c_double), # image
                                        ct.c_int, # height
                                        ct.c_int, # width
                                        ct.c_int, # num_channels
                                        ct.c_int, # grid[0]
                                        ct.c_int, # grid[1]
                                        ct.c_int, # method
                                        ct.POINTER(ct.c_double) # output
                                       ]
    
    def wrapper(self, image, grid, method):
        output = np.empty((grid[0], grid[1], image.shape[-1]))
        self.fp.fastpooling(
                image.ctypes.data_as(ct.POINTER(ct.c_double)),
                ct.c_int(image.shape[0]),
                ct.c_int(image.shape[1]),
                ct.c_int(image.shape[2]),
                ct.c_int(grid[0]),
                ct.c_int(grid[1]),
                ct.c_int(method),
                output.ctypes.data_as(ct.POINTER(ct.c_double)))
        return output.flatten(), output.shape
    
    def testPoolingSingle(self):
        data = np.random.rand(10,10,3)
        data_2d = data.reshape((100,3))
        pooled_max = data_2d.max(axis=0)
        pooled_ave = data_2d.mean(axis=0)
        pooled_rms = np.sqrt((data_2d**2).mean(axis=0))
        np.testing.assert_almost_equal(pooled_max,
                self.wrapper(data, (1,1), self.methods['max'])[0])
        np.testing.assert_almost_equal(pooled_ave,
                self.wrapper(data, (1,1), self.methods['ave'])[0])
        np.testing.assert_almost_equal(pooled_rms,
                self.wrapper(data, (1,1), self.methods['rms'])[0])
        
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
                    self.wrapper(data, grid, self.methods['max'])[0])
            np.testing.assert_almost_equal(pooled_ave,
                    self.wrapper(data, grid, self.methods['ave'])[0])
            np.testing.assert_almost_equal(pooled_rms,
                    self.wrapper(data, grid, self.methods['rms'])[0])

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
                        shape = self.wrapper(data, grid, self.methods['max'])[1]
                        self.assertEqual(shape, grid + (channel,))

if __name__ == '__main__':
    unittest.main()
