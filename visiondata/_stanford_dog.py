"""The Stanford Dog Dataset
"""
from iceberk import datasets, mpi
import numpy as np
import os
from scipy import misc, io
import xml.etree.ElementTree as ET

class StanfordDogDataset(datasets.ImageSet):
    def __init__(self, root, is_training, crop = False,
                 prefetch = False, target_size = None):
        """Load the dataset.
        Input:
            root: the root folder of the CUB_200_2011 dataset.
            is_training: if true, load the training data. Otherwise, load the
                testing data.
            crop: if False, does not crop the bounding box. If a real value,
                crop is the ratio of the bounding box that gets cropped.
                e.g., if crop = 1.5, the resulting image will be 1.5 * the
                bounding box area.
            prefetch: if True, the images are prefetched to avoid disk read. If
                you have a large number of images, prefetch would require a lot
                of memory.
            target_size: if provided, all images are resized to the size 
                specified. Should be a list of two integers, like [640,480].
            
        Note that we will use the python indexing (labels start from 0).
        """
        if is_training:
            mat_filename = 'train_list.mat'
        else:
            mat_filename = 'test_list.mat'
        if mpi.is_root():
            matfile = io.loadmat(os.path.join(root, mat_filename))
            labels = np.array(matfile['labels'].flatten()-1, dtype=np.int)
            files = [f[0][0] for f in matfile['file_list']]
        else:
            labels = None
            files = None
        self._data = mpi.distribute_list(files)
        self._label = mpi.distribute(labels)
        self._root = root
        self._prefetch = prefetch
        self._crop = crop
        self._target_size = target_size
        if target_size is not None:
            self._dim = tuple(target_size) + (3,)
        else:
            self._dim = False
        self._channels = 3
        if self._prefetch:
            self._data = [self._read(i) for i in range(len(self._data))]
            
    def _read(self, idx):
        image = datasets.imread_rgb(os.path.join(self._root, 
                                    "Images", self._data[idx]))
        if self._crop != False:
            imheight, imwidth = image.shape[:2]
            annotation = ET.parse(os.path.join(self._root, 
                    "Annotation", self._data[idx][:-4])).getroot()
            objects = [c for c in annotation if c.tag == "object"]
            bbox = objects[0].find('bndbox')
            xmin = int(bbox.find('xmin').text) - 1
            ymin = int(bbox.find('ymin').text) - 1
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            # expand
            width = xmax - xmin
            height = ymax - ymin
            centerx = xmin + width / 2.
            centery = ymin + height / 2.
            xoffset = width * self._crop / 2.
            yoffset = height * self._crop / 2.
            xmin = max(int(centerx - xoffset + 0.5), 0)
            ymin = max(int(centery - yoffset + 0.5), 0)
            xmax = min(int(centerx + xoffset + 0.5), imwidth - 1)
            ymax = min(int(centery + yoffset + 0.5), imheight - 1)
            image = image[ymin:ymax, xmin:xmax].copy()
        if self._target_size is not None:
            image = misc.imresize(image, self._target_size)
        return image
