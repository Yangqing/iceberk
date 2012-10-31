'''The MNIST dataset 
'''

from iceberk import datasets, mpi
import numpy as np
import os

class MNISTDataset(datasets.ImageSet):
    __num_train = 60000
    __num_test = 10000
    __image_dim = (28,28)
    
    def __init__(self, rootfolder, is_training):
        super(MNISTDataset, self).__init__()
        if mpi.is_root():
            # root loads the data
            if is_training:
                self._data = self._read_byte_data(
                        os.path.join(rootfolder,'train-images-idx3-ubyte'), 
                        16, (MNISTDataset.__num_train,) + \
                                MNISTDataset.__image_dim)
                self._label = self._read_byte_data(
                        os.path.join(rootfolder,'train-labels-idx1-ubyte'),
                        8, [MNISTDataset.__num_train]).astype(np.int)
            else:
                self._data = self._read_byte_data(
                        os.path.join(rootfolder,'t10k-images-idx3-ubyte'),
                        16, (MNISTDataset.__num_test,) + \
                                MNISTDataset.__image_dim)
                self._label = self._read_byte_data(
                        os.path.join(rootfolder,'t10k-labels-idx1-ubyte'),
                        8, [MNISTDataset.__num_test]).astype(np.int)
        else:
            self._data = None
            self._label = None
        self._data = mpi.distribute(self._data)
        self._label = mpi.distribute(self._label)
        self._dim = MNISTDataset.__image_dim
        self._channels = 1
    
    def _read_byte_data(self, filename, skipbytes, shape):
        fid = open(filename, 'rb')
        fid.seek(skipbytes)
        nbytes = np.prod(shape)
        rawdata = fid.read(nbytes)
        fid.close()
        #convert rawdata to data
        data = np.zeros(nbytes)
        for i in range(nbytes):
            data[i] = ord(rawdata[i])
        data.resize(shape)
        return data