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
from jiayq_ice import mpi, datasets, pipeline, classifier, dsift
import numpy as np
import os
import sys

gflags.DEFINE_string("root", "",
                     "The root to the cifar dataset (python format)")
gflags.RegisterValidator('root', lambda x: x != "",
                         message='--root must be provided.')
gflags.DEFINE_string("feature_dir", ".",
                     "The directory that stores dumped features.")
gflags.DEFINE_string("model_file", "conv.pickle",
                     "The filename to output the model.")
gflags.DEFINE_string("feature_file", "features",
                     "The filename to output the features.")
gflags.DEFINE_string("label_file", "labels",
                     "The filename to output the labels.")
gflags.DEFINE_integer("sift_size", 16,
                      "The sift patch size")
gflags.DEFINE_integer("sift_stride", 6,
                      "The dense sift stride")
gflags.DEFINE_integer("dict_size", 1024,
                      "The LLC dictionary size")
gflags.DEFINE_integer("llc_k", 5,
                       "The LLC number of neighbors")
gflags.DEFINE_integer("num_tries", 10,
                      "The number of random train/test split tries")
gflags.DEFINE_float("gamma", 0.01,
                     "The SVM regularization term")
FLAGS = gflags.FLAGS

def compute_caltech_features():
    caltech = datasets.TwoLayerDataset(FLAGS.root,
                                       ['jpg'])
    conv = pipeline.ConvLayer([
            dsift.DsiftExtractor(FLAGS.sift_size, FLAGS.sift_stride),
            pipeline.LLCEncoder({'k': FLAGS.llc_k},
                    trainer = pipeline.KmeansTrainer({'k':FLAGS.dict_size})),
            pipeline.PyramidPooler({'level': 3, 'method': 'max'})])
    conv.train(caltech, 400000)
    feat = conv.process_dataset(caltech, as_2d = True)
    
    mpi.mkdir(FLAGS.feature_dir)
    if mpi.is_root():
        with(open(os.path.join(FLAGS.feature_dir, FLAGS.model_file),'w')) as fid:
            pickle.dump(conv, fid)
    
    mpi.dump_matrix_multi(feat, 
                          os.path.join(FLAGS.feature_dir, 
                                       FLAGS.feature_file))
    mpi.dump_matrix_multi(caltech.labels(),
                          os.path.join(FLAGS.feature_dir,
                                       FLAGS.label_file))

if __name__ == "__main__":
    gflags.FLAGS(sys.argv)
    if mpi.is_root():
        logging.basicConfig(level=logging.DEBUG)
        compute_caltech_features()
    else:
        compute_caltech_features()
