"""This module implements several common datasets used in vision following the
format defined in iceberk.datasets
"""

import logging

from iceberk.visiondata._cifar import CifarDataset
from iceberk.visiondata._cub import CUBDataset
from iceberk.visiondata._stanford_dog import StanfordDogDataset
from iceberk.visiondata._mnist import MNISTDataset
try:
    from iceberk.visiondata._stl_10 import STL10Dataset
except ImportError, e:
    logging.warning("Warning: The code for STL-10 dataset needs the h5py package, "\
                    "which is not available.")
    STL10Dataset = None
from iceberk.visiondata.jitter import JitterSet
from iceberk.datasets import TwoLayerDataset

# some alias
Caltech101Dataset = TwoLayerDataset
ILSVRCDataset = TwoLayerDataset
