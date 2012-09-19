"""
ICE is Yangqing's image classification engine mainly written to reproduce the
results in the CVPR 2012 paper:

Y. Jia, C. Huang, T. Darrell. 
Beyond Spatial Pyramids: Receptive Field Learning for Pooled Image Features. 
CVPR 2012

External dependencies
---------------------
    PIL 
      \-Image
    matplotlib 
      \-cm
      \-pyplot
    mpi4py 
      \-MPI
    numpy
    scipy 
      \-misc
    sklearn 
      \-metrics
"""

from jiayq_ice import pipeline, datasets, visualize