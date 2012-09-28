"""Yangqing's refactoring of the CVPR'12 algorithm

In this program we will represent any image (and encoded tensors) with a 
width * height * nchannels numpy matrix, which is always preserved as a
contiguous array in C-order so we can more efficiently solve most of the
problems.
"""
import ctypes as ct
from jiayq_ice import kmeans_mpi, mpi, omp_mpi, mathutil
import logging
import numpy as np
import os
from PIL import Image
from sklearn import metrics


class Component(object):
    """ The common interface to process an input image
    
    Components could be used to stack up a big convolutional layer. See 
    ConvLayer for more details.
    """
    def __init__(self, specs):
        """ initialize with some specifications
        
        Input:
            specs: a dictionary containing param keywords and values
        """
        self.specs = specs
    
    
    def process(self, image):
        """ The interface that processes the input from the last component,
        and outputs the input for the next component.
        
        Input:
            image: an Ndarray where the feature is along the last axis
        Output:
            output: an Ndarray, whose size only differes from image on the
                last dimension. It should be the output to the next layer.
        """
        raise NotImplementedError

    def train(self, patches):
        """ The interface that trains the component.
        
        Input:
            patches: a 2-D numpy array (unlike image!)
        Output:
            output: the 2-D numpy array as the output of this component
        """
        raise NotImplementedError

class ConvLayer(list):
    """ ConvLayer is one big layer in the convolutional pipeline.
    
    It starts with a patch extractor, followed by several feature processing
    components, and ends with a spatial pooler.
    """
    def train(self, dataset, num_patches):
        """ train the convolutional layer
        
        Note that we do not train the first element (patch extractor),
        and stop when we see the spatial pooler. There might be some post
        processing components after the pooler, but they should not require
        any training (if they do, you may want to move them to the next layer
        """
        logging.debug("Training convolutional layer...")
        if not isinstance(self[0], PatchExtractor):
            raise ValueError, "The first component should be a patch extractor!"
        patches = self[0].sample(dataset, 
                                 int(round(num_patches / mpi.SIZE + 0.5)))
        for component in self[1:]:
            if isinstance(component, Pooler):
                # if we've reached pooler, stop training
                break
            patches = component.train(patches)
        
    def process(self, image):
        output = image
        for element in self:
            output = element.process(output)
        return output
        
    def process_dataset(self, dataset):
        """Processes a whole dataset and returns an numpy ndarray
        """
        
        return np.asarray([self.process(dataset.image(i))
                           for i in range(dataset.size())])
            
class PatchExtractor(object):
    """The patch extractor. It densely extracts overlapping patches, and 
    convert them as an NDarray which could be passed on to different 
    components.
    """
    def __init__(self, psize, stride):
        """Initialize a patch extractor
        
        Input:
        psize: the patch size, [w,h] for rectangular patches, or a single int
            for square patches.
        stride: the stride for dense extraction
        """
        if type(psize) is int:
            self.psize = [psize,psize]
        else:
            self.psize = psize
        self.stride = stride
        
    def sample(self, dataset, num_patches):
        """ randomly sample num_patches from the dataset.
        
        The returned patches would be a 2-dimensional ndarray of size
            [num_patches, psize[0] * psize[1] * num_channels]
        """
        imids = np.random.randint(dataset.size(), size=num_patches)
        # sort the ids so we don't need to re-read images when sampling
        imids.sort()
        patches = np.empty((num_patches, 
                            self.psize[0] * 
                            self.psize[1] * 
                            dataset.num_channels()))
        current_im = -1
        if dataset.dim() is not None:
            # all images have the same dim, making random sampling easier
            dim = dataset.dim()
            rowids = np.random.randint(dim[0]-self.psize[0], size=num_patches)
            colids = np.random.randint(dim[1]-self.psize[1], size=num_patches)
            precomputed = True  
        else:
            precomputed = False
        for i in range(num_patches):
            if imids[i] != current_im:
                im = dataset.image(imids[i])
                current_im = imids[i]
            if not precomputed:
                rowid = np.random.randint(im.shape[0]-self.psize[0])
                colid = np.random.randint(im.shape[1]-self.psize[1])
            else:
                rowid = rowids[i]
                colid = colids[i]
            patches[i] = im[rowid:rowid+self.psize[0], \
                            colid:colid+self.psize[1]].flat
        return patches
        
    def process(self, image):
        '''process an image
        
        The returned image would be a 3-dimensional ndarray of size
            [new_height, new_width, psize[0] * psize[1] * num_channels]
        '''
        imheight = image.shape[0]
        imwidth = image.shape[1]
        try:
            num_channels = image.shape[2]
        except IndexError:
            num_channels = 1
        stride = self.stride
        idxh = range(0,imheight-self.psize[0]+1,stride)
        idxw = range(0,imwidth-self.psize[1]+1,stride)
        new_height, new_width = len(idxh), len(idxw)
        num_patches= len(idxh) * len(idxw)
        if num_patches == 0:
            raise ValueError, "The image is too small for dense extraction!"
        patches = np.empty((new_height, new_width, 
                            self.psize[0] * 
                            self.psize[1] * 
                            num_channels))
        for i in idxh:
            for j in idxw:
                patches[i,j] = image[i:i+self.psize[0],j:j+self.psize[1]].flat
        return patches
        

class Normalizer(Component):
    """ Normalizer are those layers that do not need training
    """
    
    def process(self, image):
        raise NotImplementedError
    
    def train(self, patches):
        """ For normalizers, no training should be needed.
        """
        return self.process(patches)
    

class MeanvarNormalizer(Normalizer):
    """Normalizes the patches to mean zero and standard deviation 1
    
    Specs:
        'reg': the regularization term added to the norm.
    """
    def process(self, image):
        """ normalizes the patches.
        """
        reg = self.specs.get('reg', np.finfo(np.float64).eps)
        image_out = image - image.mean(axis=-1).\
                                reshape(image.shape[:-1] + (1,))
        image_out /= (np.sqrt(np.mean(image_out**2, axis = -1)) + reg).\
                            reshape(image.shape[:-1] + (1,))
        return image_out
            
        
class L2Normalizer(Normalizer):
    """Normalizes the patches so they lie on a unit ball.
    
    Specs:
        'reg': the regularization term added to the norm.
    """
    def process(self, image):
        """ normalizes the patches
        """
        reg = self.specs.get('reg', np.finfo(np.float64).eps)
        image_out = image / (np.sqrt(np.sum(image**2, axis = -1)) + reg).\
                                reshape(image.shape[:-1] + (1,))
        return image_out


class L1Normalizer(Normalizer):
    """Normalizes the patches so each patch sums to 1
    
    Specs:
        'reg': the regularization term added to the norm.
    """
    def process(self, image):
        """ normalizes the patches
        """
        reg = self.specs.get('reg', np.finfo(np.float64).eps)
        image_out = image / (np.sum(image, axis = -1) + reg).\
                                reshape(image.shape[:-1] + (1,))
        return image_out
    

class DictionaryTrainer(object):
    """The dictionary trainer
    """
    def __init__(self, specs):
        """ initialize with some specifications
        
        Input:
            specs: a dictionary containing param keywords and values
        """
        self.specs = specs
        
    def train(self, incoming_patches):
        """ train a dictionary, and return the necessary dictionary parameters
        
        Input:
            incoming_patches: a 2-d matrix each row being a patch feature vector
            specs: a dictionary of specification parameters
        Output:
            dictionary: the dictionary items
            misc: misc variables that might be useful in inspection.
        """
        raise NotImplementedError
    
class PcaTrainer(DictionaryTrainer):
    """Performs PCA training
    """
    def train(self, incoming_patches):
        size = mpi.COMM.allreduce(incoming_patches.shape[0])
        b = - mpi.COMM.allreduce(np.sum(incoming_patches,axis=0)) / size
        # remove the mean from data
        patches = incoming_patches + b
        covmat = mpi.COMM.allreduce(mathutil.dot(patches.T, patches)) / size
        if mpi.RANK == 0:
            eigval, eigvec = np.linalg.eigh(covmat)
            reg = self.specs.get('reg', np.finfo(np.float64).eps)
            W = eigvec * 1.0 / (np.sqrt(np.maximum(eigval, 0.0)) + reg)
        else:
            eigval, eigvec, W = None, None, None
        W = mpi.COMM.bcast(W)
        eigval = mpi.COMM.bcast(eigval)
        eigvec = mpi.COMM.bcast(eigvec)
        return (W, b), (eigval, eigvec)

class ZcaTrainer(PcaTrainer):
    """Performs ZCA training
    """
    def train(self, incoming_patches):
        (W, b), (eigval, eigvec) = PcaTrainer.train(self, incoming_patches)
        W = np.dot(W, eigvec.T)
        return (W, b), (eigval, eigvec)

class KmeansTrainer(DictionaryTrainer):
    def train(self, incoming_patches):
        centroid, label, inertia = \
            kmeans_mpi.kmeans(incoming_patches, 
                              self.specs['k'],
                              n_init = self.specs.get('n_init', 1),
                              max_iter = self.specs.get('max_iter', 100),
                              tol = self.specs.get('tol', 0.0001))
        return centroid, (label, inertia)


class OMPTrainer(DictionaryTrainer):
    """Orthogonal Matching Pursuit
    """
    def train(self, incoming_patches):
        centroid = omp_mpi.omp1(incoming_patches,
                                self.specs['k'],
                                max_iter = self.specs.get('max_iter', 100),
                                tol = self.specs.get('tol', 0.0001)
                                )
        return centroid, ()
        
        
class FeatureEncoder(Component):
    """The feature encoder.
    
    The old PatchPreprocessor in jiayq.imageclassify is now part of 
    PatchEncoder. The old DictTrainer is now absorbed into PatchEncoder.
    """
    def __init__(self, specs, trainer = None):
        self.trainer = trainer
        self.dictionary = None
        super(FeatureEncoder, self).__init__(specs)
        
    def process(self, image):
        raise NotImplementedError
        
    def train(self, incoming_patches):
        if self.trainer is not None:
            self.dictionary = self.trainer.train(incoming_patches)[0]
        return self.process(incoming_patches)

class LinearEncoder(FeatureEncoder):
    """A linear encoder that does output = W (input + b)
    """
    def process(self, image):
        W, b = self.dictionary
        return np.dot(image + b, W)
        
class InnerProductEncoder(FeatureEncoder):
    """ An innner product encoder that does output = np.dot(input, dictionary)
    """
    def process(self, image):
        return mathutil.dot_image(image, self.dictionary.T)

class ThresholdEncoder(FeatureEncoder):
    """ Like inner product encoder, but does thresholding to zero-out small
    values.
    """
    def process(self, image):
        # 0.25 is the default value used in Ng's paper
        alpha = self.specs.get('alpha', 0.25)
        output = mathutil.dot_image(image, self.dictionary.T)
        # check if we would like to do two-side thresholding. Default yes.
        if self.specs.get('twoside', True):
            # concatenate, and make sure to be C_CONTIGUOUS
            imshape = output.shape[:-1]
            N = output.shape[-1]
            output.resize((np.prod(imshape), N))
            temp = np.empty((np.prod(imshape), N*2))
            temp[:,:N] = output
            temp[:,N:] = -output
            output = temp.reshape(imshape + (N*2,))
        else:
            # otherwise, we will take the absolute value
            output = np.abs(output)
        return np.maximum(output-alpha, 0.)

class TriangleEncoder(FeatureEncoder):
    """ Does triangle encoding as described in Coates and Ng's AISTATS paper
    """
    def process(self, image):
        shape = image.shape[:-1]
        num_channels = image.shape[-1]
        image_2d = image.reshape((np.prod(shape), num_channels))
        distance = metrics.euclidean_distances(image_2d, self.dictionary)
        mu = np.mean(distance, axis=1)
        encoded = np.maximum(0., mu.reshape(mu.size, 1) - distance)
        return encoded.reshape(shape + (encoded.shape[-1],))
                    
class LLCEncoder(FeatureEncoder):
    """Encode with LLC
    """
    def process(self, image):
        '''Performs llc encoding.
        '''
        K = self.specs['k']
        reg = self.specs['reg']
        D = self.dictionary
        shape = image.shape[:-1]
        X = image.reshape((np.prod(shape), image.shape[-1]))
        # D_norm is the precomputed norm of the entries
        if 'D_norm' not in self.specs:
            self.specs['D_norm'] = np.sum(D**2,1) / 2.
        D_norm = self.specs['D_norm']
        # similarity is not the distance, but the 
        # order is preserved (reversed order as distance).
        similarity = np.dot(X, D.T) - D_norm
        IDX = np.argsort(similarity,1)
    
        # do LLC approximate coding
        coeff = np.zeros((X.shape[0], D.shape[0]))
        ONES = np.ones(K)
        for i in range(X.shape[0]):
            # shift to origin
            z = D[IDX[i, :K]] - X[i]
            # local covariance
            C = np.dot(z,z.T)
            # regularization
            C.flat[::K+1] = reg * C.trace()
            w = np.linalg.solve(C,ONES)
            coeff[i][IDX[i,:K]] = w / w.sum()
        return coeff.reshape(shape + (coeff.shape[1],))

class Pooler(Component):
    """Pooler is just an abstract class that holds all pooling subclasses
    """
    def train(self, incoming_patches):
        raise RuntimeError,\
            "You should not call the train() function of a pooler."

class SpatialPooler(Pooler):
    """ The spatial Pooler that does spatial pooling on a regular grid.
    """
    _METHODS = {'max':0, 'ave': 1, 'rms': 2}
    # fast pooling C library
    _FASTPOOL = np.ctypeslib.load_library('libfastpool.so',
                                          os.path.dirname(__file__))
    _FASTPOOL.fastpooling.restype = ct.c_int
    _FASTPOOL.fastpooling.argtypes = [ct.POINTER(ct.c_double), # image
                                          ct.c_int, # height
                                          ct.c_int, # width
                                          ct.c_int, # num_channels
                                          ct.c_int, # grid[0]
                                          ct.c_int, # grid[1]
                                          ct.c_int, # method
                                          ct.POINTER(ct.c_double) # output
                                         ]
    
    def process(self, image):
        if not (image.flags['C_CONTIGUOUS'] and image.dtype == np.float64):
            logging.warning("Warning: the image is not contiguous.")
            image = np.ascontiguousarray(image, dtype=np.float64)
        # do fast pooling
        grid = self.specs['grid']
        if type(grid) is int:
            grid = (grid, grid)
        output = np.empty((grid[0], grid[1], image.shape[-1]))
        SpatialPooler._FASTPOOL.fastpooling(\
                image.ctypes.data_as(ct.POINTER(ct.c_double)),
                ct.c_int(image.shape[0]),
                ct.c_int(image.shape[1]),
                ct.c_int(image.shape[2]),
                ct.c_int(grid[0]),
                ct.c_int(grid[1]),
                ct.c_int(SpatialPooler._METHODS[self.specs['method']]),
                output.ctypes.data_as(ct.POINTER(ct.c_double)))
        return output


class WeightedPooler(Pooler):
    """WeightedPooler does weighted sum (or rms) of the incoming image
    """
    def process(self, image):
        image = np.atleast_3d(image)
        height, width, channels = image.shape
        maps = self.specs['maps']
        num_maps = len(maps)
        maps_rescaled = []
        for i in range(num_maps):
            weightmap = Image.fromarray(maps[i])
            map_rescaled = np.asarray(weightmap.resize((height,width), \
                                                       Image.BILINEAR) \
                                     ).flatten()
            map_rescaled /= map_rescaled.sum() + np.finfo(np.float64).eps
            maps_rescaled.append(map_rescaled)
        image = image.reshape((height*width, channels))
        output = np.empty((num_maps, channels))
        if self.specs['method'] == 'ave':
            for i, weightmap in enumerate(maps_rescaled):
                output[i] = (image * weightmap[:,np.newaxis]).sum(axis=0)
        elif self.specs['method'] == 'rms':
            image **= 2
            for i, weightmap in enumerate(maps_rescaled):
                output[i] = np.sqrt(
                        (image * weightmap[:,np.newaxis]).sum(axis=0))
        else:
            raise ValueError, \
                'The method %s cannot be recognized.' % (self.specs['method'])
        return output
        
if __name__ == "__main__":
    print "It works!"