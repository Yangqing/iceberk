'''The STL-10 dataset 
'''

import cPickle as pickle
import h5py
from jiayq_ice import datasets, mpi
import numpy as np
import os
from scipy import io

class STL10Dataset(datasets.ImageSet):
    """The STL-10 dataset
    """
    # some constants
    _image_dim = (96, 96, 3)
    _num_channels = 3
    
    def __init__(self, root, mode, is_gray = False):
        """Loads the STL dataset. mode should be either 'train', 'test', or 
        'unlabeled'
        """
        if mode == 'train':
            self._data, self._label = \
                    STL10Dataset.get_data(os.path.join(root, 'train.mat'))
        elif mode == 'test':
            self._data, self._label = \
                    STL10Dataset.get_data(os.path.join(root, 'test.mat'))
        elif mode == 'unlabeled':
            # h5py allows us to directly read part of the matrix, so each
            # node will work on his own
            matdata = h5py.File(os.path.join(root, 'unlabeled.mat'),'r')
            segments = mpi.get_segments(matdata['X'].shape[1])
            # read
            X = matdata['X'][:, segments[mpi.RANK]:segments[mpi.RANK+1]]
            X.resize(STL10Dataset._image_dim[::-1] + (X.shape[1],))
            self._data = np.ascontiguousarray(np.transpose(X))
            self._label = None
        else:
            raise ValueError, "Unrecognized mode."
        if is_gray:
            self._data = self._data.mean(axis=-1)
            self._dim = STL10Dataset._image_dim[:2]
            self._channels = 1
        else:
            self._dim = STL10Dataset._image_dim
            self._channels = STL10Dataset._num_channels

        self._prefetch = True
    
    @staticmethod
    def get_data(filename):
        """This is a wrapper function that returns the images in the right
        axes order
        """
        if mpi.is_root():
            matdata = io.loadmat(filename)
            X = matdata['X'].reshape(\
                    (matdata['X'].shape[0],) + STL10Dataset._image_dim[::-1])
            # make it contiguous so we can do mpi distribute
            X = np.ascontiguousarray(np.transpose(X, axes=[0,3,2,1]),
                                     dtype = X.dtype)
            Y = matdata['y'].astype(int).flatten()
        else:
            X = None
            Y = None
        return mpi.distribute(X), mpi.distribute(Y)

def demo_read(root):
    from jiayq_ice import visualize
    vis = visualize.PatchVisualizer()
    print 'Loading training data...'
    traindata = STL10Dataset(root, 'train')
    print 'My training data size:', traindata.size()
    print 'Loading testing data...'
    testdata = STL10Dataset(root, 'test')
    print 'My testing data size:', testdata.size()
    print 'Loading unlabeled data...'
    unlabeleddata = STL10Dataset(root, 'unlabeled')
    print 'My unlabeled data size:', unlabeleddata.size()
    if mpi.is_root():
        vis.pyplot.figure()
        vis.show_multiple(traindata.raw_data()[:25])
        vis.pyplot.title('Sample training images.')
        vis.pyplot.figure()
        vis.show_multiple(testdata.raw_data()[:25])
        vis.pyplot.title('Sample testing images.')
        vis.pyplot.figure()
        vis.show_multiple(unlabeleddata.raw_data()[:25])
        vis.pyplot.title('Sample unlabeled images.')
        vis.pyplot.show()
    mpi.barrier()

if __name__ == "__main__":
    import sys
    demo_read(sys.argv[1])
    