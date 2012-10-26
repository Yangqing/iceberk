"""
ICEBERK is Yangqing's image classification engine mainly written to reproduce
the results in the CVPR 2012 paper:

Y. Jia, C. Huang, T. Darrell. 
Beyond Spatial Pyramids: Receptive Field Learning for Pooled Image Features. 
CVPR 2012

It then evolved to have more codes integrated. Use with caution.

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

from iceberk import pipeline, datasets, visualize