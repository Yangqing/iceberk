'''The Cifar dataset 
'''

import cPickle as pickle
from jiayq_ice import datasets, mpi
import numpy as np
import os


class CifarDataset(datasets.ImageSet):
    """The Cifar dataset
    """
    # some cifar constants
    __num_train = 50000
    __num_batches = 5 # for cifar 10
    __batchsize = 10000 # for cifar 10
    __num_test = 10000
    __image_dim = (32, 32, 3)
    __num_channels = 3
    __image_size = 1024
    __flat_dim = 3072
    
    def __init__(self, rootfolder, is_training, is_gray = False):
        super(CifarDataset, self).__init__()
        # we will automatically determine if the data is cifar-10 or cifar-100
        if os.path.exists(rootfolder+os.sep+'batches.meta'):
            self.name = 'CIFAR-10'
            self.load_cifar10(rootfolder, is_training)
        elif os.path.exists(rootfolder+os.sep+'meta'):
            self.name = 'CIFAR-100'
            self.load_cifar100(rootfolder, is_training)
        else:
            raise IOError, 'Cannot understand the dataset format.'
        if is_gray:
            self._data = self._data.mean(axis=-1)
            self._dim = CifarDataset.__image_dim[:2]
            self._channels = 1
        else:
            self._dim = CifarDataset.__image_dim
            self._channels = CifarDataset.__num_channels
        
    @staticmethod
    def get_images_from_matrix(mat):
        """ converts the order of the loaded matrix so each pixel is stored
        consistently
        """
        mat = mat.reshape((mat.shape[0],
                           CifarDataset.__num_channels,
                           CifarDataset.__image_size))
        images = mat.swapaxes(1, 2).reshape((mat.shape[0],) + \
                                            CifarDataset.__image_dim)
        return images.astype(np.float64)
    
    def load_cifar100(self, rootfolder, is_training):
        """loads the cifar-100 dataset
        """
        if mpi.is_root():
            if is_training:
                filename = 'train'
            else:
                filename = 'test'
            with open(rootfolder + os.sep + filename) as fid:
                batch = pickle.load(fid)
            self._data = CifarDataset.get_images_from_matrix(batch['data'])
            self._coarselabel = np.array(batch['coarse_labels'])
            self._label = np.array(batch['fine_labels'])
        else:
            self._data = None
            self._coarselabel = None
            self._label = None
        self._data = mpi.distribute(self._data)
        self._coarselabel = mpi.distribute(self._coarselabel)
        self._label = mpi.distribute(self._label)
    
    def coarse_labels(self):
        return self._coarselabel.copy()
    
    def load_cifar10(self, rootfolder, is_training):
        """loads the cifar-10 dataset
        """
        if mpi.is_root():
            if is_training:
                self._data = np.empty((CifarDataset.__num_train,) + \
                                      CifarDataset.__image_dim)
                self._label = np.empty(CifarDataset.__num_train)
                # training batches
                for i in range(CifarDataset.__num_batches):
                    with open(os.path.join(rootfolder,
                            'data_batch_{0}'.format(i+1)),'r') as fid:
                        batch = pickle.load(fid)
                    start_idx = CifarDataset.__batchsize * i
                    end_idx = CifarDataset.__batchsize * (i+1)
                    self._data[start_idx:end_idx] = \
                            CifarDataset.get_images_from_matrix(batch['data'])
                    self._label[start_idx:end_idx] = np.array(batch['labels'])
            else:
                with open(os.path.join(rootfolder, 'test_batch'), 'r') as fid:
                    batch = pickle.load(fid)
                self._data = CifarDataset.get_images_from_matrix(batch['data'])
                self._label = np.array(batch['labels'])
        else:
            self._data = None
            self._label = None
        self._data = mpi.distribute(self._data)
        self._label = mpi.distribute(self._label)
