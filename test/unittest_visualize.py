from iceberk import visualize, mpi
import numpy as np
import unittest


class TestExtractor(unittest.TestCase):
    def setUp(self):
        self.visualizer = visualize.PatchVisualizer(gap=1)
        
    def testGetPatchShape(self):
        patch = np.random.rand(16)
        self.assertEqual(self.visualizer.get_patch_shape(patch), (4,4))
        patch = np.random.rand(16*3)
        self.assertEqual(self.visualizer.get_patch_shape(patch), (4,4,3))
        wrongpatches = [np.random.rand(16*5), np.random.rand(15)]
        for patch in wrongpatches:
            try:
                self.visualizer.get_patch_shape(patch)
                self.assertTrue(False,
                    "When calling a wrong patch, an error should be raised.")
            except Exception, err:
                self.assertTrue(isinstance(err, ValueError))
        return
    
    def testShowSingle(self):
        patch = np.random.rand(16)
        self.assertEqual(self.visualizer.show_single(patch).shape, (4,4))
        patch = np.random.rand(16*3)
        self.assertEqual(self.visualizer.show_single(patch).shape, (4,4,3))
        patch = np.random.rand(4,4)
        self.assertEqual(self.visualizer.show_single(patch).shape, (4,4))
        patch = np.random.rand(4,4,3)
        self.assertEqual(self.visualizer.show_single(patch).shape, (4,4,3))
        
    def testShowMultiple(self):
        patches = np.random.rand(10,16)
        self.assertEqual(self.visualizer.show_multiple(patches).shape, (14,19))
        patches = np.random.rand(10,4,4)
        self.assertEqual(self.visualizer.show_multiple(patches).shape, (14,19))
        patches = np.random.rand(10,16*3)
        self.assertEqual(self.visualizer.show_multiple(patches).shape, (14,19,3))
        patches = np.random.rand(10,4,4,3)
        self.assertEqual(self.visualizer.show_multiple(patches).shape, (14,19,3))
    
    def testShowChannels(self):
        patches = np.random.rand(4,4,10)
        self.assertEqual(self.visualizer.show_channels(patches).shape, (14,19))
      
if __name__ == "__main__":
    unittest.main()