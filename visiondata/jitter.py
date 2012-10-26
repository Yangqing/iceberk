'''JITTER implements a toy jittering effect for images

@author: jiayq
'''

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

def random_jitter(img, translation_std, rotation_std, scaling_std):
    """Randomly jitter an image by providing the translation, rotation and
    scaling standard deviations
    
    Definition of std:
        translation: in the number of pixels
        rotation: in degrees
        scaling: in log_2 scale, e.g. scaling=0.5 means 2^0.5 times larger
    """
    translation = np.random.randn(2) * translation_std
    rotation = np.random.randn() * np.pi * rotation_std / 180.
    scaling = np.random.randn() * scaling_std
    return jitter(img, translation, rotation, scaling)

if __name__ == "_main__":
    pass
