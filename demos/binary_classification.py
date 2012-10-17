'''Demos how to do binary classification.

@author: jiayq
'''
from matplotlib import pyplot
import numpy as np
from iceberk import classifier

X = np.vstack([np.random.randn(100,2)-1, np.random.randn(100,2)+1])
Y = np.hstack([np.ones(100), -np.ones(100)])
w,b = classifier.svm_binary(X,Y,0.1)
print "classifying two gaussian distributions located at [-1,-1] and [1,1]"
print 'Weight:', w, 'Bias:', b
