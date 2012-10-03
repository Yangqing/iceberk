'''
Created on Sep 20, 2012

@author: jiayq
'''
import cPickle as pickle
import cProfile
import gflags
import logging
from jiayq_ice import mpi, visiondata, pipeline, classifier
import numpy as np
import sys

gflags.DEFINE_string("root", "",
                     "The root to the cifar dataset")
gflags.RegisterValidator('root', lambda x: x != "",
                         message='--root must be provided.')
gflags.DEFINE_string("output_dir", ".",
                     "The output directory")
gflags.DEFINE_string("model_file", "conv.pickle",
                     "The output model file")
gflags.DEFINE_string("feature_file", "cnn_features",
                     "The output feature files")
gflags.DEFINE_string("svm_file", "svm.pickle",
                     "The output svm file")
gflags.DEFINE_string("profile", "",
                     "If set, do cProfile analysis")
FLAGS = gflags.FLAGS
def demo():
    mpi.mkdir(FLAGS.output_dir)
    logging.info('Loading cifar data...')
    cifar = visiondata.CifarDataset(FLAGS.root, is_training=True)
    cifar_test = visiondata.CifarDataset(FLAGS.root, is_training=False)
    
    conv = pipeline.ConvLayer([
            pipeline.PatchExtractor([6,6], 1), # extracts patches
            pipeline.MeanvarNormalizer({'reg': 10}), # normalizes the patches
            pipeline.LinearEncoder({},
                    trainer = pipeline.ZcaTrainer({'reg': 0.01})), # Does whitening
            pipeline.TriangleEncoder({},
                    trainer = pipeline.KmeansTrainer({'k': 800})), # does encoding
            pipeline.SpatialPooler({'grid': (2,2), 'method': 'ave'}) # average pool
            ])
    logging.info('Training the pipeline...')
    conv.train(cifar, 400000)
    if mpi.is_root():
        with open(FLAGS.model_file,'w') as fid:
            pickle.dump(conv, fid)
            fid.close()
    
    logging.info('Extracting features...')
    Xtrain = conv.process_dataset(cifar, as_2d = True)
    mpi.dump_matrix_multi(Xtrain, FLAGS.feature_file+'_train')
    Ytrain = cifar.labels()
    Xtest = conv.process_dataset(cifar_test, as_2d = True)
    mpi.dump_matrix_multi(Xtest, FLAGS.feature_file+'_test')
    Ytest = cifar_test.labels()
    
    # normalization
    m = Xtrain.mean(axis=1)
    std = Xtrain.std(asix=1)
    Xtrain -= m
    Xtrain /= std
    Xtest -= m
    Xtest /= std
    
    w, b = classifier.l2svm_onevsall(Xtrain, Ytrain, 0.01)
    with open(FLAGS.svm_file, 'w') as fid:
        pickle.dump({'m': m, 'std': std, 'w': w, 'b': b}, fid)
    accu = np.sum(Ytrain == (np.dot(Xtrain,w)+b).argmax(axis=1)) \
            / float(len(Ytrain))
    accu_test = np.sum(Ytrain == (np.dot(Xtrain,w)+b).argmax(axis=1)) \
            / float(len(Ytrain))
    
    logging.info('Training accuracy: %f' % accu)
    logging.info('Testing accuracy: %f' % accu_test)

if __name__ == "__main__":
    gflags.FLAGS(sys.argv)
    if mpi.is_root():
        logging.basicConfig(level=logging.DEBUG)
        if FLAGS.profile != "":
            cProfile.run('demo()', FLAGS.profile)
        else:
            demo()
    else:
        demo()
