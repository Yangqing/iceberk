from iceberk import pipeline, datasets, mpi
import numpy as np
import unittest


class TestExtractor(unittest.TestCase):
    def setUp(self):
        self._sample_number = 1000
        self._dim = 32
        self._patchsize = 6
        mat = np.random.rand(100, self._dim, self._dim, 3)
        mat_sc = np.random.rand(100, self._dim, self._dim)
        mat_mc = np.random.rand(100, self._dim, self._dim, 6)
        self.data = datasets.NdarraySet(mat)
        self.data_sc = datasets.NdarraySet(mat_sc)
        self.data_mc = datasets.NdarraySet(mat_mc)
        self.extractor = pipeline.PatchExtractor(self._patchsize, 1)
    
    def testSample(self):
        patches = self.extractor.sample(self.data, self._sample_number)
        self.assertEqual(patches.shape[0], self._sample_number)
        self.assertEqual(patches.shape[1], self._patchsize*self._patchsize*3)
        
        patches = self.extractor.sample(self.data_sc, self._sample_number)
        self.assertEqual(patches.shape[0], self._sample_number)
        self.assertEqual(patches.shape[1], self._patchsize*self._patchsize)
        
        patches = self.extractor.sample(self.data_mc, self._sample_number)
        self.assertEqual(patches.shape[0], self._sample_number)
        self.assertEqual(patches.shape[1], self._patchsize*self._patchsize*6)

    def testProcess(self):
        patches = self.extractor.process(self.data.image(0))
        self.assertEqual(patches.shape, (self._dim - self._patchsize + 1, 
                                         self._dim - self._patchsize + 1,
                                         self._patchsize*self._patchsize*3))
        patches = self.extractor.process(self.data_sc.image(0))
        self.assertEqual(patches.shape, (self._dim - self._patchsize + 1, 
                                         self._dim - self._patchsize + 1,
                                         self._patchsize*self._patchsize))
        patches = self.extractor.process(self.data_mc.image(0))
        self.assertEqual(patches.shape, (self._dim - self._patchsize + 1, 
                                         self._dim - self._patchsize + 1,
                                         self._patchsize*self._patchsize*6))


class TestNormalizer(unittest.TestCase):
    def setUp(self):
        self.test_patches = [np.random.rand(100,36),
                             np.random.rand(27, 27, 36),
                             np.random.rand(10, 27, 27, 36)]
        
    def testBaseNormalizer(self):
        normalizer = pipeline.Normalizer({'test':'dummy'})
        self.assertTrue(isinstance(normalizer, pipeline.Component))
    
    def testMeanvarNormalizer(self):
        normalizer = pipeline.MeanvarNormalizer({})
        for patch in self.test_patches:
            output = normalizer.process(patch)
            self.assertEqual(patch.shape, output.shape)
            np.testing.assert_almost_equal(output.mean(axis=-1), 0.)
            np.testing.assert_almost_equal(np.mean(output**2, axis=-1), 1.)
        
    def testL1Normalizer(self):
        normalizer = pipeline.L1Normalizer({})
        for patch in self.test_patches:
            output = normalizer.process(patch)
            self.assertEqual(patch.shape, output.shape)
            np.testing.assert_almost_equal(output.sum(axis=-1), 1.)
            
    def testL2Normalizer(self):
        normalizer = pipeline.L2Normalizer({})
        for patch in self.test_patches:
            output = normalizer.process(patch)
            self.assertEqual(patch.shape, output.shape)
            np.testing.assert_almost_equal((output**2).sum(axis=-1), 1.)
            
class TestDictTrainer(unittest.TestCase):
    def setUp(self):
        self.test_patches = np.random.rand(1000,36)
        
    def testPcaTrainer(self):
        trainer = pipeline.PcaTrainer({})
        (W, b), (eigval, eigvec) = trainer.train(self.test_patches)
        np.testing.assert_equal(W.shape[0], self.test_patches.shape[1])
        np.testing.assert_equal(W.shape[1], self.test_patches.shape[1])
        if mpi.SIZE == 1:
            # we test values as well
            np.testing.assert_array_almost_equal(
                    self.test_patches.mean(axis=0), -b)
            covmat = np.cov(np.dot(self.test_patches+b, W), rowvar=0)
            covmat -= np.diag(np.diag(covmat))
            np.testing.assert_array_almost_equal(covmat, 0.)
        
    def testZcaTrainer(self):
        trainer = pipeline.ZcaTrainer({})
        W, b = trainer.train(self.test_patches)[0]
        np.testing.assert_equal(W.shape[0], self.test_patches.shape[1])
        np.testing.assert_equal(W.shape[1], self.test_patches.shape[1])
        if mpi.SIZE == 1:
            np.testing.assert_array_almost_equal(
                    self.test_patches.mean(axis=0), -b)
            covmat = np.cov(np.dot(self.test_patches+b, W), rowvar=0)
            np.testing.assert_array_almost_equal(np.diag(covmat), 1., 3)
            covmat -= np.diag(np.diag(covmat))
            np.testing.assert_array_almost_equal(covmat, 0.)
            
    def testKmeansTrainer(self):
        specs = {'k': 100}
        trainer = pipeline.KmeansTrainer(specs)
        centroids, (label, inertia) = trainer.train(self.test_patches)
        np.testing.assert_equal(centroids.shape[0], specs['k'])
        np.testing.assert_equal(centroids.shape[1], self.test_patches.shape[1])
        np.testing.assert_array_less(label, specs['k'])
        np.testing.assert_array_less(-label, 1)
        self.assertGreater(inertia, 0.)
        
    def testOMPTrainer(self):
        specs = {'k': 100}
        trainer = pipeline.OMPTrainer(specs)
        centroids = trainer.train(self.test_patches)[0]
        np.testing.assert_equal(centroids.shape[0], specs['k'])
        np.testing.assert_equal(centroids.shape[1], self.test_patches.shape[1])
        np.testing.assert_array_almost_equal((centroids**2).sum(axis=1), 1.)


class TestEncoder(unittest.TestCase):
    def setUp(self):
        self.training_patches= [np.random.rand(1000,36),
                                np.random.rand(1000,1),
                                np.random.rand(1000,100)]
        self.test_images = [np.random.rand(1000,36),
                             np.random.rand(32,32,1),
                             np.random.rand(32,32,100)]
        
    def testLinearEncoder(self):
        trainer = pipeline.PcaTrainer({})
        encoder = pipeline.LinearEncoder({}, trainer = trainer)
        for patches, image in zip(self.training_patches, self.test_images):
            encoder.train(patches)
            W, b = trainer.train(patches)[0]
            output = encoder.process(image)
            self.assertEqual(output.shape, image.shape)
            np.testing.assert_array_almost_equal(output, np.dot(image + b, W))
            
    def testInnerProductEncoder(self):
        trainer = pipeline.OMPTrainer({'k': 10})
        encoder = pipeline.InnerProductEncoder({}, trainer = trainer)
        for patches, image in zip(self.training_patches, self.test_images):
            encoder.train(patches)
            output = encoder.process(image)
            self.assertEqual(output.shape[:-1], image.shape[:-1])
            self.assertEqual(output.shape[-1], trainer.specs['k'])
            np.testing.assert_array_almost_equal(output,
                    np.dot(image, encoder.dictionary.T))

    def testThresholdEncoder(self):
        trainer = pipeline.OMPTrainer({'k': 10})
        encoder = pipeline.ThresholdEncoder({'alpha': 0.25, 'twoside': True},
                                            trainer = trainer)
        for patches, image in zip(self.training_patches, self.test_images):
            encoder.train(patches)
            output = encoder.process(image)
            self.assertEqual(output.shape[:-1], image.shape[:-1])
            self.assertEqual(output.shape[-1], trainer.specs['k'] * 2)
            np.testing.assert_array_less(-np.finfo(np.float64).eps,
                                         output)
        encoder = pipeline.ThresholdEncoder({'alpha': 0.25, 'twoside': False},
                                            trainer = trainer)
        for patches, image in zip(self.training_patches, self.test_images):
            encoder.train(patches)
            output = encoder.process(image)
            self.assertEqual(output.shape[:-1], image.shape[:-1])
            self.assertEqual(output.shape[-1], trainer.specs['k'])
            np.testing.assert_array_less(-np.finfo(np.float64).eps,
                                         output)
            
    def testTriangleEncoder(self):
        trainer = pipeline.KmeansTrainer({'k': 10})
        encoder = pipeline.TriangleEncoder({}, trainer = trainer)
        for patches, image in zip(self.training_patches, self.test_images):
            encoder.train(patches)
            output = encoder.process(image)
            self.assertEqual(output.shape[:-1], image.shape[:-1])
            self.assertEqual(output.shape[-1], trainer.specs['k'])
            np.testing.assert_array_less(-np.finfo(np.float64).eps,
                                         output)
    
    def testLLCEncoder(self):
        trainer = pipeline.KmeansTrainer({'k': 100})
        encoder = pipeline.LLCEncoder({'k': 10, 'reg': 0.01}, trainer = trainer)
        for patches, image in zip(self.training_patches, self.test_images):
            encoder.train(patches)
            output = encoder.process(image)
            self.assertEqual(output.shape[:-1], image.shape[:-1])
            self.assertEqual(output.shape[-1], trainer.specs['k'])
            np.testing.assert_array_less((output > 0).sum(axis=-1),
                                         encoder.specs['k']+1)


class TestSpatialPooler(unittest.TestCase):
    def setUp(self):
        heights = [16, 31, 32, 33, 37, 40]
        widths = [16, 31, 32, 33, 37, 40]
        channels = [1, 3, 5, 10]
        self.test_data = []
        for height, width in zip(heights, widths):
            for channel in channels:
                self.test_data.append(np.random.rand(height, width, channel))
    
    def testSpatialPooler(self):
        """ We only test the size of the pooler. The correctness of the values
        are tested in fastop
        """
        grids = [(2, 2), (3, 3), (2, 3), (4, 4), (2, 4), (5, 5)]
        methods = ['max','ave','rms']
        for grid in grids:
            for method in methods:
                pooler = pipeline.SpatialPooler(
                        {'method': method, 'grid': grid})
                for data in self.test_data:
                    output = pooler.process(data)
                    self.assertEqual(output.shape, grid + (data.shape[-1],))

if __name__ == '__main__':
    unittest.main()

