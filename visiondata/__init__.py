"""This module implements several common datasets used in vision following the
format defined in iceberk.datasets
"""

from iceberk.visiondata._cifar import CifarDataset
from iceberk.visiondata._cub import CUBDataset
from iceberk.visiondata._stl_10 import STL10Dataset
from iceberk.visiondata.jitter import JitterSet