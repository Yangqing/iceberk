import glob
import numpy as np
from iceberk import mpi
import logging

def CHECK_IMAGE(img):
    if (type(img) is np.ndarray) and (img.ndim == 3) \
            and (img.dtype == np.float64):
        pass
    else:
        raise RuntimeError, "The image format is incorrect."

def CHECK_SHAPE(img, shape):
    if (type(img) is not np.ndarray):
        raise RuntimeError, "The image is not a numpy array."
    if img.shape != shape:
        raise RuntimeError, "The shapes do not equal."

def gemm(alpha, A, B, dtype=None, out=None):
    '''A gemm function that uses scipy fblas functions, avoiding matrix copy
    when the input is transposed.
    
    The returned matrix is designed to be C_CONTIGUOUS.
    '''
    from scipy.linalg.fblas import dgemm, sgemm
    if A.ndim != 2 or B.ndim != 2:
        raise TypeError, 'gemm only deals with 2-D matrices.'
    if dtype is None:
        dtype=A.dtype
    if dtype != np.float32 and dtype != np.float64:
        raise TypeError, 'Error: this function cannot deal with dtype {}.'\
                .format(dtype)
    if not (A.flags['F_CONTIGUOUS'] or A.flags['C_CONTIGUOUS']) \
            or not (B.flags['F_CONTIGUOUS'] or B.flags['C_CONTIGUOUS']):
        raise TypeError, 'Matrices should either be C or F contiguous.'
    if A.dtype != dtype:
        A=np.asarray(A,dtype=dtype)
    if B.dtype != dtype:
        B=np.asarray(B,dtype=dtype)
    
    # In fact, what we are doing here is (1) compute B*A, and (2) transpose the
    # result. The reason is that fblas returns F_CONTINUOUS matrices, so doing 
    # this enables us to get a final output that is C_CONTIGUOUS.
    if not B.flags['F_CONTIGUOUS']:
        B = B.T
        trans_b=0
    else:
        trans_b=1
    if not A.flags['F_CONTIGUOUS']:
        A = A.T
        trans_a=0
    else:
        trans_a=1
    if dtype == np.float32:
        fblas_gemm = sgemm
    else:
        fblas_gemm = dgemm
    if out is None:
        return fblas_gemm(alpha,B,A,trans_a=trans_b,trans_b=trans_a).T
    else:
        if out.dtype != dtype:
            raise TypeError, "The output matrix should have type %s"\
                    % repr(dtype)
        if not out.flags['C_CONTIGUOUS']:
            raise TypeError, "The output matrix should be C contiguous."
        fblas_gemm(alpha, B, A, 0.0, out.T, trans_b, trans_a, True)
        return out


def dot(A, B, out=None):
    '''
    a simple wrapper that mimics np.dot (if A and B are both matrices!)
    This function solves the problem that np.dot copies matrices when
    working on transposed matrices.
    Input:
        A, B: two matrices. should be either c-contiguous or f-contiguous
    Output:
        out: the output matrix
    Raises:
        TypeError, if the type of matrices is wrong.
    '''
    return gemm(1.0, A, B, out=out)


def dot_image(image, B, out=None):
    """ A wrapper that does dot for a multidimensional image that is often used
    in the pipeline. The input image should be C-contiguous.
    """
    
    imshape = image.shape
    if not image.flags['C_CONTIGUOUS']:
        raise TypeError, 'Error: cannot deal with non-C-contiguous image'
    if out is None:
        out = np.empty((np.prod(imshape[:-1]), B.shape[1]))
    else:
        out.resize((np.prod(imshape[:-1]), B.shape[1]))
    out = gemm(1.0, image.reshape((np.prod(imshape[:-1]), imshape[-1])), B,
                  out=out)
    out.resize(imshape[:-1] + (B.shape[1],))
    return out


def exp(X, out = None):
    """ A (hacky) safe exp that avoids overflowing
    Input:
        X: the input ndarray
        out: (optional) the output ndarray. Could be in-place.
    Output:
        out: the output ndarray
    """
    if out is None:
        out = np.empty_like(X)
    np.clip(X, -np.inf, 100, out = out)
    # we do in-place exp
    np.exp(out, out = out)
    return out

def log(X, out = None):
    """ A (hacky) safe log that avoids nans
    
    Note that if there are negative values in the input, this function does not
    throw an error. Handle these cases with care.
    """
    if out is None:
        out = np.empty_like(X)
    np.clip(X, np.finfo(np.float64).eps, np.inf, out = out)
    np.log(out, out = out)
    return out


def wolfe_line_search_adagrad(x, func, alpha = 1., eta = 0., c1 = 0.01, c2 = 0.9, tau = 0.8):
    """Perform line search using the Wolfe's condition. The search direction
    will be determined as if we are doing the first step of adagrad. Note that this
    will yield a direction different from the gradient.
    Input:
        x: the initial point
        func: the function to minimize, should be in the form
            function_value, gradient = func(x)
        alpha: the initial step size
        eta: the initial value to start the gradient accumulation.
        c1, c2: the constants in the Wolfe's condition.
        tau: the shrinkage factor. If conditions are not met, then we set
            alpha <- alpha * tau
    """
    f0, g0 = func(x)
    # copy g0 so calling func again does not modify it
    g0 = g0.copy()
    direction = - g0 / np.sqrt(g0 * g0 + eta * eta + np.finfo(np.float64).eps)
    logging.debug('wolfe ls: f = %f.' % (f0))
    alpha /= tau
    while True:
        alpha *= tau
        f, g = func(x + alpha * direction)
        if f > f0 + c1 * alpha * np.dot(direction, g0):
            logging.debug('wolfe ls: a = %f, f = %f, condition 1 not met' % \
                    (alpha, f))
            continue
        elif np.dot(direction, g) < c2 * np.dot(direction, g0):
            logging.debug('wolfe ls: a = %f, f = %f, condition 2 not met' % \
                    (alpha, f))
            continue
        else:
            # both conditions met!
            break
    logging.debug('wolfe ls: a = %f, f = %f, finished.' % (alpha, f))
    return alpha

class ReservoirSampler(object):
    """reservoir_sampler implements the reservoir sampling method based on numpy
    matrices. It does NOT use mpi - each mpi node does sampling on its own.
    """
    def __init__(self, num_samples):
        """Initializes the sampler by giving the number of data points N
        """
        self._num_samples = num_samples
        self._current = 0
        self._data = None
        
    def consider(self, feature):
        """Consider a feature batch. feature.shape[1:] should be the same for
        any batch.
        """
        if self._data is None:
            self._data = np.empty((self._num_samples,) + feature.shape[1:],
                                  dtype=feature.dtype)
        elif self._data.shape[1:] != feature.shape[1:]:
            raise ValueError, \
                    "Input data has the wrong size, should be %s " \
                    % str(feature.shape[1:])
        batch_size = feature.shape[0]
        if self._current >= self._num_samples:
            # to make sure we have unbiased sampling, we do 2 steps: first
            # decide whether to use one sample or not, then decide which one it
            # should replace
            # we need to decide if we want to keep the feature
            selected = np.random.rand(batch_size) < \
                    (float(self._num_samples) / np.arange(self._current + 1,
                            self._current + batch_size + 1))
            count = selected.sum()
            self._data[np.random.randint(self._num_samples, size = count)] = \
                    feature[selected]
            self._current += batch_size
        else:
            # we need to fill the data first, and then deal with remaining
            # features
            count = min(self._num_samples - self._current, batch_size)
            self._data[self._current:self._current+count] = feature[:count]
            self._current += count
            if count < batch_size:
                # we have remaining features to consider
                self.consider(feature[count:])
    
    def num_considered(self):
        """Return the number of considered samples
        """
        return self._current
    
    def get(self):
        """After considering all samples, call get() to get the sampled
        features
        """
        if self._current < self._num_samples:
            # not enough data considered. Only return valid ones
            return self._data[:self._current]
        else:
            return self._data


class MinibatchSampler(object):
    """MinibatchSampler is the general class that performs minibatch sampling
    from data
    """
    def sample(self, batch_size):
        """return a minibatch sample. Your sampler should implement this.
        """
        raise NotImplementedError

class PostProcessSampler(MinibatchSampler):
    """PostProcessSampler does not do sampling directly, but uses a basic 
    sampler to get minibatches. What it does is to then apply a set of
    postprocessing functions to the minibatch, such as data conversion and
    normalization.
    """
    def __init__(self, basic_sampler, funcs):
        """Initialize the postprocesing sampler
        Input:
            basic_sampler: the basic sampler that generates the minibatch.
            funcs: a list of functions, one for each minibatch entry.
        """
        self._basic = basic_sampler
        self._funcs = funcs

    def sample(self, batch_size):
        batch = self._basic.sample(batch_size)
        output = []
        for mat, func in zip(batch, self._funcs):
            if func is None:
                output.append(mat)
            else:
                output.append(func(mat))
        return output


class NdarraySampler(MinibatchSampler):
    """This sampler initializes with a list or tuple of ndarrays, and for each
    sample, return a list of the same length, and each entry will be a minibatch
    from the corresponding input. For the input list, each entry should have
    the same shape[0] (the sampling will be carried out along the first axis)
    or be None, in which case the returned list will have a corresponding None
    entry as well.
    
    This sampler takes the distributed storage in consideration. If the
    program is run over multiple instances, each mpi instance will only return
    a proportion of the minibatch, and the total size will be equal to
    minibatch.
    """
    def __init__(self, arrays):
        self._arrays = arrays
        lengths = [t.shape[0] for t in arrays if t is not None]
        if not all([x == lengths[0] for x in lengths]):
            raise ValueError, \
                    "The input ndarrays should have the same shape[0]."
        self._num_data_local = lengths[0]
        self._num_data = mpi.COMM.allreduce(self._num_data_local)
        # initialize some bookkeeping values for the sampler
        self._indices = np.arange(self._num_data_local, dtype = np.int)
        np.random.shuffle(self._indices)
        self._pointer = 0
    
    def sample(self, batch_size):
        # compute the local batch size, and make sure the sampling is done
        # proportional to the number of data points that is hosted locally.
        batch_size = int(batch_size * self._num_data_local / \
                   float(self._num_data))
        if (self._num_data_local < batch_size):
            raise ValueError, "I can't do such a big batch size!"
        if (self._num_data_local - self._pointer < batch_size):
            # The remaining data are not enough, reshuffle
            np.random.shuffle(self._indices)
            old_pointer = 0
            self._pointer = batch_size
        else:
            old_pointer = self._pointer
            self._pointer += batch_size
        output = []
        for array in self._arrays:
            if array is None:
                output.append(None)
            else:
                batch_idx = self._indices[old_pointer:self._pointer]
                output.append(array[batch_idx].copy())
        return output
        

class FileSampler(MinibatchSampler):
    """FileSampler takes in a set of files stored in a distributed fasion, and
    use memory maps to access the files and do sampling, in order to save 
    memory. Currently, you will need to make sure all file parts for an array
    is loadable from the running machine.
    """
    def __init__(self, filenames):
        """Initialize the sampler.
        Input:
            filenames: a list of filenames containing the data. Each filename
                could be a specific name like "label.npy", or a name that
                contains wildcards like "Xtrain-*-of-*.npy" in case the data
                is stored in chunks (see iceberk.mpi). In the latter case, we
                will read the files one by one using their string order. The
                list could contain None, in which case we will simply return
                None for the corresponding sample.
        """
        self._datamaps = []
        # precheck the files to understand the storage structure
        for fname in filenames:
            if fname is None:
                self._datamaps.append(None)
            else:
                files = glob.glob(fname)
                files.sort()
                if len(files) == 0:
                    raise ValueError, "Cannot find file: %s" % fname
                segments = []
                start_id = 0
                for f in files:
                    mat = np.load(f, mmap_mode='r')
                    end_id = mat.shape[0] + start_id
                    segments.append((f, start_id, end_id, mat.shape[1:], mat.dtype))
                    start_id = end_id

                self._datamaps.append(segments)
        # find the number of data
        num_data = [m[-1][2] for m in self._datamaps]
        if not all(n == num_data[0] for n in num_data):
            raise ValueError, \
                    "Files %s do not have the same number of data."
        self._num_data = num_data[0]
        self._indices = np.arange(self._num_data, dtype = np.int)
        np.random.shuffle(self._indices)
        self._pointer = 0
        return

    def sample(self, batch_size):
        if (batch_size > self._num_data):
            raise ValueError, "I can't do such a big batch size!"
        batch_size = batch_size / mpi.SIZE
        if (self._num_data - self._pointer < batch_size):
            # The remaining data are not enough, reshuffle
            np.random.shuffle(self._indices)
            old_pointer = 0
            self._pointer = batch_size
        else:
            old_pointer = self._pointer
            self._pointer += batch_size
        batch_idx = self._indices[old_pointer:self._pointer]
        batch_idx.sort()
        return [self._sample_single(m, batch_idx) for m in self._datamaps]

    def _sample_single(self, datamap, indices):
        """Sample the numpy array based on the current filename
        """
        if datamap is None:
            return None
        else:
            batchsize = len(indices)
            output = np.empty((batchsize,) + datamap[0][-2],
                              dtype = datamap[0][-1])
            startid = 0
            for m in datamap:
                # determine my end location
                endid = startid
                while endid < batchsize and indices[endid] < m[2]:
                    endid += 1
                if endid > startid:
                    mat = np.load(m[0], mmap_mode='r')
                    output[startid:endid] = mat[indices[startid:endid] - m[1]]
                startid = endid
            return output
                 
###############################################################################
# MPI-related utils are implemented here.
###############################################################################

def mpi_mean(data):
    """An mpi implementation of the mean over different nodes along axis 0
    """
    s_local = data.sum(0)
    m = np.empty_like(s_local)
    mpi.COMM.Allreduce(s_local, m)
    num_data = mpi.COMM.allreduce(data.shape[0])
    m /= float(num_data)
    return m

def mpi_meanstd(data):
    """An mpi implementation of the std over different nodes along axis 0
    """
    m = mpi_mean(data)
    # since we need to compute the square, we cannot do in-place subtraction
    # and addition.
    try:
        data_centered = data - m
        data_centered **= 2
        std_local = data_centered.sum(0)
        std_local_computed = 1
    except MemoryError:
        std_local_computed = 0
    # let's check if some nodes did not have enough memory
    if mpi.COMM.allreduce(std_local_computed) < mpi.SIZE:
        # we need to compute the std_local in a batch-based way
        std_local = np.zeros_like(data[0])
        # we try to get a reasonable minibatch size
        minibatch = max(int(data.shape[0] / 10), 1)
        data_batch = np.empty_like(data[:minibatch])
        for start in range(0, data.shape[0], minibatch):
            end = min(data.shape[0], start + minibatch)
            data_batch[:end-start] = data[start:end] - m
            data_batch **= 2
            std_local += data_batch.sum(axis=0)
    std = np.empty_like(std_local)
    mpi.COMM.Allreduce(std_local, std)
    num_data = mpi.COMM.allreduce(data.shape[0])
    std /= float(num_data)
    np.sqrt(std, out=std)
    return m, std

def mpi_std(data):
    return mpi_meanstd(data)[1]

def mpi_meancov(data, copydata = False):
    """An mpi implementation of the covariance matrix over different nodes
    """
    m = mpi_mean(data)
    if copydata:
        # copy the data and avoid numerical instability
        data = data - m
    else:
        data -= m
    cov_local = dot(data.T, data)
    covmat = np.empty_like(cov_local)
    mpi.COMM.Allreduce(cov_local, covmat)
    num_data = mpi.COMM.allreduce(data.shape[0])
    covmat /= float(num_data)
    if not copydata:
        data += m
    return m, covmat

def mpi_cov(data, copydata = False):
    return mpi_meancov(data, copydata)[1]
