'''
This script uses the jiayq_ice pipeline to perform a cifar classification demo
using parameter settings idential to Adam Coates' AISTATS paper (except for the
number of kmeans centers, which we set to 800 for speed considerations).

You need to specify "--root=/path/to/cifar-data" to run the code. For other
optional flags, run the script with --help or --helpshort.

@author: jiayq
'''

import cPickle as pickle
import cProfile
import gflags
import logging
from jiayq_ice import mpi, visiondata, pipeline, classifier
import numpy as np
import os
import sys

gflags.DEFINE_string("root", "",
                     "The root to the cifar dataset (python format)")
gflags.RegisterValidator('root', lambda x: x != "",
                         message='--root must be provided.')
gflags.DEFINE_string("output_dir", ".",
                     "The output directory that stores dumped feature and"
                     "classifier files.")
gflags.DEFINE_string("model_file", "conv.pickle",
                     "The filename to output the CNN model.")
gflags.DEFINE_string("feature_file", "cnn_features",
                     "The filename to output the features.")
gflags.DEFINE_string("svm_file", "svm.pickle",
                     "The filename to output the trained svm parameters.")
gflags.DEFINE_string("profile_file", "",
                     "If set, do cProfile analysis, and output the result"
                     "to the filename given by the flag.")
FLAGS = gflags.FLAGS

def cifar_demo():
    """Performs a demo classification on cifar
    """
    mpi.mkdir(FLAGS.output_dir)
    logging.info('Loading cifar data...')
    cifar = visiondata.CifarDataset(FLAGS.root, is_training=True)
    cifar_test = visiondata.CifarDataset(FLAGS.root, is_training=False)
    
    conv = pipeline.ConvLayer([
            pipeline.PatchExtractor([6,6], 1), # extracts patches
            pipeline.MeanvarNormalizer({'reg': 10}), # normalizes the patches
            pipeline.LinearEncoder({},
                    trainer = pipeline.ZcaTrainer({'reg': 0.1})), # Does whitening
            pipeline.TriangleEncoder({},
                    trainer = pipeline.KmeansTrainer(
                            {'k': 800, 'max_iter':100})), # does encoding
            pipeline.SpatialPooler({'grid': (2,2), 'method': 'ave'}) # average pool
            ])
    logging.info('Training the pipeline...')
    conv.train(cifar, 400000)
    if mpi.is_root():
        with open(os.path.join(FLAGS.output_dir, FLAGS.model_file),'w') as fid:
            pickle.dump(conv, fid)
            fid.close()
    
    logging.info('Extracting features...')
    Xtrain = conv.process_dataset(cifar, as_2d = True)
    mpi.dump_matrix_multi(Xtrain,
                          os.path.join(FLAGS.output_dir, 
                                       FLAGS.feature_file+'_train'))
    Ytrain = cifar.labels().astype(np.int)
    Xtest = conv.process_dataset(cifar_test, as_2d = True)
    mpi.dump_matrix_multi(Xtest,
                          os.path.join(FLAGS.output_dir, 
                                       FLAGS.feature_file+'_test'))
    Ytest = cifar_test.labels().astype(np.int)
    
    # normalization
    m, std = classifier.feature_meanstd(Xtrain)
    Xtrain -= m
    Xtrain /= std
    Xtest -= m
    Xtest /= std
    
    w, b = classifier.l2svm_onevsall(Xtrain, Ytrain, 0.01)
    if mpi.is_root():
        with open(os.path.join(FLAGS.output_dir, FLAGS.svm_file), 'w') as fid:
            pickle.dump({'m': m, 'std': std, 'w': w, 'b': b}, fid)
    accu = np.sum(Ytrain == (np.dot(Xtrain,w)+b).argmax(axis=1)) \
            / float(len(Ytrain))
    accu_test = np.sum(Ytest == (np.dot(Xtest,w)+b).argmax(axis=1)) \
            / float(len(Ytest))
    
    logging.info('Training accuracy: %f' % accu)
    logging.info('Testing accuracy: %f' % accu_test)

if __name__ == "__main__":
    gflags.FLAGS(sys.argv)
    if mpi.is_root():
        logging.basicConfig(level=logging.DEBUG)
        if FLAGS.profile != "":
            cProfile.run('cifar_demo()', FLAGS.profile)
        else:
            cifar_demo()
    else:
        cifar_demo()
