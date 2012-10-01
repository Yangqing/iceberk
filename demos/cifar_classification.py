'''
Created on Sep 20, 2012

@author: jiayq
'''
import cPickle as pickle
from jiayq_ice import mpi, visiondata, pipeline

_CIFAR_FOLDER = '/u/vis/x1/common/CIFAR/cifar-10-batches-py'
_MODEL_FILE = './conv.pickle'

print 'Loading cifar data...'
cifar = visiondata.CifarDataset(_CIFAR_FOLDER, is_training=True)
conv = pipeline.ConvLayer([
        pipeline.PatchExtractor([6,6], 1), # extracts patches
        pipeline.MeanvarNormalizer({'reg': 10}), # normalizes the patches
        pipeline.LinearEncoder({},
                trainer = pipeline.ZcaTrainer({'reg': 0.01})), # Does whitening
        pipeline.TriangleEncoder({},
                trainer = pipeline.KmeansTrainer({'k': 800})), # does encoding
        pipeline.SpatialPooler({'grid': (2,2), 'method': 'ave'}) # average pool
        ])
print 'Training the pipeline...'
conv.train(cifar, 400000)
if mpi.is_root():
    fid = open(_MODEL_FILE,'w')
    pickle.dump(conv, fid)
    fid.close()

print 'Extracting features...'

Xtrain = conv.process_dataset(cifar)
Ytrain = cifar.labels()


