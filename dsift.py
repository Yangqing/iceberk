'''dsift.py implements the dense sift feature extraction code.

The descriptors are defined in a similar way to the one used in
Svetlana Lazebnik's Matlab implementation, which could be found
at:

http://www.cs.unc.edu/~lazebnik/

Yangqing Jia, jiayq@eecs.berkeley.edu
'''

import numpy as np
from scipy import signal
from jiayq_ice import pipeline, mpi
import logging

"""Default SIFT feature parameters
"""
_NUM_ANGLES = 8
_NUM_BINS = 4
_NUM_SAMPLES = _NUM_BINS**2
_ALPHA = 9.0
_ANGLES = np.array(range(_NUM_ANGLES))*2.0*np.pi/_NUM_ANGLES

def gen_dgauss(sigma,fwid=None):
    '''generating a derivative of Gauss filter on both the X and Y
    direction.
    '''
    if fwid is None:
        fwid = np.int(2*np.ceil(sigma))
    else:
        # in the code below, fwid is half the size of the returned filter
        # i.e. after the following line, if fwid is 2, the returned filter
        # will have size 5.
        fwid = int(fwid/2)
    sigma += np.finfo(np.float64).eps
    G = np.array(range(-fwid,fwid+1))**2
    G = G.reshape((G.size,1)) + G
    G = np.exp(- G / 2.0 / sigma / sigma)
    G /= np.sum(G)
    GH,GW = np.gradient(G)
    GH *= 2.0/np.sum(np.abs(GH))
    GW *= 2.0/np.sum(np.abs(GW))
    return GH,GW

class DsiftExtractor(pipeline.Extractor):
    '''
    The class that does dense sift feature computation.
    Sample Usage:
        extractor = DsiftExtractor(gridSpacing,patchSize,[optional params])
        feat,positions = extractor.process_image(Image)
    '''
    def __init__(self, psize, stride, specs = {}):
        '''
        stride: the spacing for sampling dense descriptors
        psize: the size for each sift patch
        specs:
            nrml_thres: low contrast normalization threshold
            sigma_edge: the standard deviation for the gaussian smoothing
                before computing the gradient
            sift_thres: sift thresholding (0.2 works well based on
                Lowe's SIFT paper)
        '''
        self.gS = stride
        self.pS = psize
        self.nrml_thres = specs.get('nrml_thres', 1.0)
        self.sigma = specs.get('sigma_edge', 0.8)
        self.sift_thres = specs.get('sift_thres', 0.2)
        # compute the weight contribution map
        sample_res = self.pS / np.double(_NUM_BINS)
        sample_p = np.array(range(self.pS))
        sample_ph, sample_pw = np.meshgrid(sample_p,sample_p)
        sample_ph.resize(sample_ph.size)
        sample_pw.resize(sample_pw.size)
        bincenter = np.array(range(1,_NUM_BINS*2,2)) \
                    / 2.0 / _NUM_BINS * self.pS - 0.5
        bincenter_h, bincenter_w = np.meshgrid(bincenter,bincenter)
        bincenter_h.resize((bincenter_h.size,1))
        bincenter_w.resize((bincenter_w.size,1))
        dist_ph = abs(sample_ph - bincenter_h)
        dist_pw = abs(sample_pw - bincenter_w)
        weights_h = dist_ph / sample_res
        weights_w = dist_pw / sample_res
        weights_h = (1-weights_h) * (weights_h <= 1)
        weights_w = (1-weights_w) * (weights_w <= 1)
        # weights is the contribution of each pixel to the corresponding bin
        # center
        self.weights = weights_h * weights_w
        #pyplot.imshow(self.weights)
        #pyplot.show()
        
    def sample(self, dataset, num_patches):
        """ randomly sample num_patches from the dataset.
        
        The returned patches would be a 2-dimensional ndarray of size
            [num_patches, psize[0] * psize[1] * num_channels]
        When we sample patches, we need to process all the images, which might
        not be a very efficient way
        """
        num_patches = np.maximum(int(num_patches / float(mpi.SIZE) + 0.5), 1)
        data = np.empty((num_patches, _NUM_SAMPLES*_NUM_ANGLES))
        curr = 0
        for i in range(dataset.size()):
            feat = self.process(dataset.image(i))
            feat.resize(np.prod(feat.shape[:2]), feat.shape[2])
            # we perform approximate reservoir sampling
            if curr < num_patches:
                num_to_add = np.minimum(num_patches - curr, feat.shape[0])
                data[curr:curr + num_to_add] = feat[:num_to_add]
                curr += num_to_add
            else:
                # do random replacement
                curr += feat.shape[0]
                to_replace = (np.random.rand(feat.shape[0]) \
                              > num_patches / float(curr))
                replace_num = to_replace.sum()
                if replace_num > 0:
                    replace_id = np.random.randint(feat.shape[0],
                                                   size=replace_num)
                    data[replace_id] = feat[to_replace]
        return data
        
    def process(self, image):
        '''
        processes a single image, return the locations
        and the values of detected SIFT features.
        image: a M*N image which is a numpy 2D array. If you 
            pass a color image, it will automatically be converted
            to a grayscale image.
        
        Return values:
            feat
        '''
        image = image.astype(np.double)
        if image.max() > 1:
            # The image is between 0 and 255 - we need to convert it to [0,1]
            image /= 255;
        if image.ndim == 3:
            # we do not deal with color images.
            image = np.mean(image,axis=2)
        # compute the grids
        H,W = image.shape
        if H < self.pS or W < self.pS:
            logging.warning("Image size is smaller than patch size.")
            return np.zeros((0,0,_NUM_SAMPLES*_NUM_ANGLES))
        gS = self.gS
        pS = self.pS
        remH = np.mod(H-pS, gS)
        remW = np.mod(W-pS, gS)
        offsetH = remH/2
        offsetW = remW/2
        rangeH = np.arange(offsetH,H-pS+1,gS)
        rangeW = np.arange(offsetW, W-pS+1, gS)
        logging.debug('Image: w {}, h {}, gs {}, ps {}, nFea {}'.\
                      format(W,H,gS,pS,len(rangeH)*len(rangeW)))
        feat = self.calculate_sift_grid(image,rangeH,rangeW)
        feat = self.normalize_sift(feat)
        return feat

    def calculate_sift_grid(self,image,rangeH,rangeW):
        '''This function calculates the unnormalized sift features
        It is called by process_image().
        '''
        H,W = image.shape
        feat = np.zeros((len(rangeH), len(rangeW), _NUM_SAMPLES*_NUM_ANGLES))

        # calculate gradient
        GH,GW = gen_dgauss(self.sigma)
        IH = signal.convolve2d(image,GH,mode='same')
        IW = signal.convolve2d(image,GW,mode='same')
        Imag = np.sqrt(IH**2+IW**2)
        Itheta = np.arctan2(IH,IW)
        Iorient = np.zeros((_NUM_ANGLES,H,W))
        for i in range(_NUM_ANGLES):
            Iorient[i] = Imag * np.maximum(np.cos(Itheta - _ANGLES[i])**_ALPHA,
                                           0)
        
        currFeature = np.zeros((_NUM_ANGLES,_NUM_SAMPLES))
        for i, hs in enumerate(rangeH):
            for j, ws in enumerate(rangeW):
                for k in range(_NUM_ANGLES):
                    currFeature[k] = np.dot(self.weights,
                                            Iorient[k,
                                                    hs:hs+self.pS,
                                                    ws:ws+self.pS
                                                   ].flatten())
                feat[i,j] = currFeature.flat
        return feat

    def normalize_sift(self,feat):
        '''
        This function does sift feature normalization
        following David Lowe's definition (normalize length ->
        thresholding at 0.2 -> renormalize length)
        '''
        siftlen = np.sqrt(np.sum(feat**2,axis=-1))
        hcontrast = (siftlen >= self.nrml_thres)
        siftlen[siftlen < self.nrml_thres] = self.nrml_thres
        # normalize with contrast thresholding
        feat /= siftlen[:,:,np.newaxis]
        # suppress large gradients
        feat[feat>self.sift_thres] = self.sift_thres
        # renormalize high-contrast ones
        feat[hcontrast] /= np.sqrt(np.sum(feat[hcontrast]**2,axis=-1))\
                [:, np.newaxis]
        return feat
