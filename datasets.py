"""Datasets implements some basic structures to deal with a dataset
"""
import glob
from jiayq_ice import mpi
import logging
import numpy as np
import os
from PIL import Image
from scipy import misc


def imread_rgb(fname):
    '''This imread deals with occasional cases when scipy.misc.imread fails to
    load an image correctly.
    '''
    return np.asarray(Image.open(fname,'r').convert('RGB'))


class ImageSet(object):
    """The basic structure that stores data. This class should be MPI ready.
    
    Each datum in this dataset would have to be a 3-dimensional image of size
    (height * width * num_channels) even if the number of channels is 1.
    """
    def __init__(self):
        """ You should write your own initialization code!
        """
        self._data = None
        self._label = None
        self._dim = False
        self._channels = False
        self._prefetch = True
    
    def size(self):
        """Return the size of the dataset hosted on the current node
        """
        return len(self._data)
    
    def size_total(self):
        """Return the size of the dataset hosted on all nodes
        """
        return mpi.COMM.allreduce(self.size())
    
    def _read(self, idx):
        """reads a datum given by idx, if not prefetched.
        """
        raise NotImplementedError
    
    def image(self, idx):
        """ Returns datum 
        
        Note that you should almost never use data that is hosted on other
        nodes - every node should deal with its data only.
        """
        if self._prefetch:
            return self._data[idx]
        else:
            return self._read(idx)
    
    def raw_data(self):
        """ Returns the raw data
        
        Make sure you know what is stored in self._data if you use this
        """
        return self._data
    
    def label(self, idx):
        """ Returns the label for the corresponding datum
        """
        return self._label[idx]

    def labels(self):
        """ Returns the label vector for all the data I am hosting
        """
        return np.array(self._label)
    
    def dim(self):
        """Returns the dimension of the data if they have the same dimension
        Otherwise, return False
        """
        return self._dim
    
    def num_channels(self):
        """ Returns the number of channels
        """
        return self._channels
        
        
class NdarraySet(ImageSet):
    """Wraps an Ndarray using the dataset interface
    """
    def __init__(self, input_array, label = None, copy=False):
        """Initializtion
        
        If copy is true, copy the data
        """
        super(NdarraySet, self).__init__()
        if copy:
            self._data = input_array.copy()
        else:
            self._data = input_array
        if label is None:
            self._label = np.zeros(input_array.shape[0])
        elif len(label) != input_array.shape[0]:
            raise ValueError, \
                  "The number of input images and labels should be the same."
        else:
            self._label = label.copy()
        self._dim = self._data.shape[1:]
        # The number of channels. If the data has less than 4 dims, we
        # set the num of channels to 1 (in the case of e.g. grayscale images)
        if len(self._data.shape) < 4:
            self._channels = 1
        else:
            self._channels = self._dim[-1]

class MirrorSet(ImageSet):
    def __init__(self, original_set):
        """Create a mirrored dataset from the original data set.
        """
        super(MirrorSet, self).__init__()
        self._original = original_set

    def size(self):
        """Return the size of the dataset hosted on the current node
        """
        return self._original.size() * 2

    def image(self, idx):
        """ Returns datum 
        
        Note that you should almost never use data that is hosted on other
        nodes - every node should deal with its data only.
        """
        if idx < self._original.size():
            return self._original.image(idx)
        else:
            im = self._original.image(idx - self._original.size())
            return np.ascontiguousarray(im[:, ::-1])

    def label(self, idx):
        """ Returns the label for the corresponding datum
        """
        return self._original.label(idx % self._original.size())

    def labels(self):
        """ Returns the label vector for all the data I am hosting
        """
        return np.hstack((self._original.labels, self._original.labels))
    
    def dim(self):
        """Returns the dimension of the data if they have the same dimension
        Otherwise, return False
        """
        return self._original.dim()
    
    def num_channels(self):
        """ Returns the number of channels
        """
        return self._original.num_channels()


class TwoLayerDataset(ImageSet):
    """Builds a dataset composed of two-layer storage structures similar to
    Caltech-101
    """
    def __init__(self, root_folder, extensions, prefetch = False, 
                 target_size = None, max_size = None):
        """ Initialize from a two-layer storage
        Input:
            root_folder: the root that contains the data. Under root_folder
                there should be a list of folders, under which there should be
                a list of files
            extensions: the list of extensions that should be used to filter the
                files. Should be like ['png', 'jpg']. It's case insensitive.
            prefetch: if True, the images are prefetched to avoid disk read. If
                you have a large number of images, prefetch would require a lot
                of memory.
            target_size: if provided, all images are resized to the size 
                specified. Should be a list of two integers, like [640,480].
            max_size: if provided, any image that is larger than the max size
                is scaled so that its larger edge has max_size. if target_size
                is set, it overrides max_size.
        """
        super(TwoLayerDataset, self).__init__()
        if mpi.agree(not os.path.exists(root_folder)):
            raise OSError, "The specified folder does not exist."
        logging.debug('Loading from %s' % (root_folder,))
        if type(extensions) is str:
            extensions = [extensions]
        if mpi.is_root():
            # get files first
            files = glob.glob(os.path.join(root_folder, '*', '*'))
            # select those that fits the extension
            files = [f for f in files  if any([
                            f.lower().endswith(ext) for ext in extensions])]
            # get raw labels
            labels = [os.path.split(f)[-2] for f in files]
            classnames = list(set(labels))
            # sort so we get a reasonable class order
            classnames.sort()
            name2val = dict(zip(classnames, range(len(classnames))))
            labels = [name2val[label] for label in labels]
        else:
            files = None
            classnames = None
            labels = None
        self._data = mpi.distribute_list(files)
        self._prefetch = prefetch
        self.target_size = target_size
        self._max_size = max_size
        if target_size != None:
            self._dim = tuple(target_size) + (3,)
        else:
            self._dim = False
        self._channels = 3
        if prefetch:
            self._data = [self._read(idx) for idx in range(len(self._data))]
        self._label = mpi.distribute_list(labels)
        self._classnames = mpi.COMM.bcast(classnames)
    
    def _read(self, idx):
        if self.target_size is not None:
            return misc.imresize(imread_rgb(self._data[idx]),
                                 self.target_size)
        else:
            img = imread_rgb(self._data[idx])
            if self._max_size is not None and \
                    max(img.shape[:2]) > self._max_size:
                ratio = self._max_size / float(max(img.shape[:2]))
                img = misc.imresize(img, ratio)
            return img
