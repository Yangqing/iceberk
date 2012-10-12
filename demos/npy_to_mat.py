'''
Performs transformation from numpy npy format to mat format.

@author: jiayq
'''

import gflags
from jiayq_ice import mpi
import numpy as np
from scipy import io
import sys

gflags.DEFINE_string("output", "",
                   "If set, combine multiple files and output to the given "
                   "filename. The files should contain matrices of the same "
                   "shape beyond the first dimension, and they will be combined"
                   " in alphabetical order.")
gflags.DEFINE_string("name", "mat",
                     "The variable name used in the output matlab file.")
FLAGS = gflags.FLAGS

def convert(files):
    files.sort()
    if len(files) == 0:
        return
    if FLAGS.output != "":
        # combine features
        mat = np.vstack([np.load(f) for f in files])
        output = FLAGS.output
        if not output.endswith('.mat'):
            output += '.mat'
        io.savemat(output, {FLAGS.name: mat}, oned_as='column')
    else:
        for filename in files:
            mat = np.load(filename)
            output = filename
            if output.endswith('.npy'):
                output = output[:-4]
            output += '.mat'
            io.savemat(output, {FLAGS.name:mat}, oned_as='column')

if __name__ == "__main__":
    files = gflags.FLAGS(sys.argv)[1:]
    convert(files)