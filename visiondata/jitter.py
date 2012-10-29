'''JITTER implements a toy jittering effect for images

@author: jiayq
'''

from iceberk import datasets
import numpy as np
from scipy import interpolate

def jitter(img, translation, rotation, scaling):
    """ Jittering an image with the given translation, rotation and scaling
    """
    # we write the program in a general fashion assuming that the image is 
    # multi channel.
    img = np.atleast_3d(img)
    img_size = np.array(img.shape[:2])
    center = img_size / 2. - 0.5
    hh, ww = np.meshgrid(np.arange(img_size[0]), np.arange(img_size[1]))
    old_coor = np.hstack((hh.reshape(hh.size, 1), ww.reshape(ww.size, 1)))\
                - center
    
    rotation_matrix = np.asarray([[ np.cos(rotation), -np.sin(rotation)],
                                  [ np.sin(rotation), np.cos(rotation)]])
    new_coor = np.dot(old_coor, rotation_matrix)
    new_coor -= translation
    new_coor *= 2. ** (- scaling)
    new_coor += center
    img_jittered = np.empty_like(img)
    # we use linear interpolation to create the image for better quality, and
    # use the nearest values for pixels outside the image
    for i in range(img.shape[2]):
        model = interpolate.RectBivariateSpline(np.arange(img_size[0]),
                                                np.arange(img_size[1]),
                                                img[:,:,i])
        out = model.ev(new_coor[:,0], new_coor[:,1])
        img_jittered[:,:,i] = out.reshape(img_size[1],img_size[0]).T
    # finally, if it's a single channel image, we will just return a single
    # channel image
    if img_jittered.shape[2] == 1:
        img_jittered.reshape(img_size)
    return img_jittered

def randn_jitter(img, stdval):
    """Randomly jitter an image by providing the translation, rotation and
    scaling standard deviations. The exact parameters are sampled from Gaussian
    distributions.
    
    Definition of std:
        translation: in the number of pixels
        rotation: in degrees
        scaling: in log_2 scale, e.g. scaling=0.5 means 2^0.5 times larger
    """
    translation = np.random.randn(2) * stdval[0]
    rotation = np.random.randn() * np.pi * stdval[1] / 180.
    scaling = np.random.randn() * stdval[2]
    return jitter(img, translation, rotation, scaling)

def rand_jitter(img, maxval):
    """Randomly jitter an image by providing the translation, rotation and
    scaling standard deviations. The exact parameters are sampled from uniform
    distributions bounded by [-max, max].
    
    Definition of max:
        translation: in the number of pixels
        rotation: in degrees
        scaling: in log_2 scale, e.g. scaling=0.5 means 2^0.5 times larger
    """
    translation = (np.random.rand(2) * 2. - 1.) * maxval[0]
    rotation = (np.random.rand() * 2. - 1.) * np.pi * maxval[1] / 180.
    scaling = (np.random.rand() * 2. - 1.) * maxval[2]
    return jitter(img, translation, rotation, scaling)


class JitterSet(datasets.ImageSet):
    """JitterSet provides a jittered version of the dataset by randomly 
    adding jitter effect to the images. 
    """
    def __init__(self, original_set, factor, params, method = 'uniform'):
        """Create a mirrored dataset from the original data set.
        Input:
            original_set: the original dataset
            factor: a positive number. the dataset is expanded 
                factor times - for each image we will have the original image
                and factor jittered images. Note that the parameters of the 
                jittered images are randomly regenerated every time you call
                image, so if you call image(n) two times with the same n, the
                results will still be different - this is by design since it
                makes stochastic approaches easier.
                
                If factor = 0, the original images are not returned, and all 
                images will be jittered.
            params: the 3-dimensional array containing the parameters for 
                jittering
            method: either 'uniform' or 'gaussian'.
        """
        super(JitterSet, self).__init__()
        self._original = original_set
        self._factor = factor
        self._params = np.asarray(params)
        if method == 'uniform':
            self._jitter = rand_jitter
        elif method == 'gaussian':
            self._jitter = randn_jitter
        else:
            raise ValueError, "Unrecognized method name: %s" % (method)
        

    def size(self):
        """Return the size of the dataset hosted on the current node
        """
        return self._original.size() * (self._factor + 1)

    def image(self, idx):
        """ Returns datum 
        
        Note that you should almost never use data that is hosted on other
        nodes - every node should deal with its data only.
        """
        if self._factor > 0 and idx < self._original.size():
            return self._original.image(idx)
        else:
            im = self._original.image(idx % self._original.size())
            return self._jitter(im, self._params)

    def label(self, idx):
        """ Returns the label for the corresponding datum
        """
        return self._original.label(idx % self._original.size())

    def labels(self):
        """ Returns the label vector for all the data I am hosting
        """
        return np.hstack([self._original.labels() 
                          for _ in range(self._factor + 1)])
    
    def dim(self):
        """Returns the dimension of the data if they have the same dimension
        Otherwise, return False
        """
        return self._original.dim()
    
    def num_channels(self):
        """ Returns the number of channels
        """
        return self._original.num_channels()

if __name__ == "_main__":
    pass
