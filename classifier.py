'''
mpiclassify
====
Provides an MPI interface that trains linear classifiers that can be represented
by
    \min_w     1/N * sum_n L(y_n,w'x_n+b) + gamma * Reg(w)

This algorithm only deals with the primal case (no dual), assuming that there 
are more data points than the number of feature dimension (if not, you might 
want to look for dual solvers to your problem). We use L-BFGS as the default
solver, and if the loss function or regularizer is not differentiable everywhere
(like the v-style L1 regularizer), we will use the subgradient methods.
'''

from iceberk import cpputil, mathutil, mpi, util
import inspect
import logging
import numpy as np
# The inner1d function is imported here to do more memory-efficient sum of
# squares. For example, if a.size = [300,100], inner1d(a,a) is equivalent to
# (a**2).sum(axis=1) but does not create additional space.
from numpy.core.umath_tests import inner1d
from scipy import optimize
from sklearn import metrics

_FMIN = optimize.fmin_l_bfgs_b

def to_one_of_k_coding(Y, fill = -1):
    '''Convert the vector Y into one-of-K coding. The element will be either
    fill (-1 in default) or 1
    '''
    if Y.ndim > 1:
        raise ValueError, "The input Y should be a vector."
    K = mpi.COMM.allreduce(Y.max(), op=max) + 1
    Yout = np.ones((len(Y), K)) * fill
    Yout[np.arange(len(Y)), Y.astype(int)] = 1
    return Yout

def feature_meanstd(mat, reg = None):
    '''
    Utility function that does distributed mean and std computation
    Input:
        mat: the local data matrix, each row is a feature vector and each 
             column is a feature dim
        reg: if reg is not None, the returned std is computed as
            std = np.sqrt(std**2 + reg)
    Output:
        m:      the mean for each dimension
        std:    the standard deviation for each dimension
    
    The implementation is actually moved to iceberk.mathutil now, we leave the
    code here just for backward compatibility
    '''
    m, std = mathutil.mpi_meanstd(mat)

    if reg is not None:
        std = np.sqrt(std**2 + reg)
    return m, std


class Solver(object):
    '''
    Solver is the general solver to deal with bookkeeping stuff
    '''
    def __init__(self, gamma, loss, reg,
                 args = {}, lossargs = {}, regargs = {}, fminargs = {}):
        '''
        Initializes the solver.
        Input:
            gamma: the regularization parameter
            loss: the loss function. Should accept three variables Y, X and W,
                where Y is a vector in {labels}^(num_data), X is a matrix of size
                [num_data,nDim], and W is a vector of size nDim. It returns
                the loss function value and the gradient with respect to W.
            reg: the regularizaiton func. Should accept a vector W of
                shape nDim and returns the regularization term value and
                the gradient with respect to W.
            args: the arguments for the solver in general.
            lossargs: the arguments that should be passed to the loss function
            regargs: the arguments that should be passed to the regularizer
            fminargs: additional arguments that you may want to pass to fmin.
                you can check the fmin function to see what arguments can be
                passed (like display options: {'disp':1}).
        '''
        self._gamma = gamma
        self.loss = loss
        self.reg = reg
        self._args = args.copy()
        self._lossargs = lossargs.copy()
        self._regargs = regargs.copy()
        self._fminargs = fminargs.copy()
        self._add_default_fminargs()
    
    def _add_default_fminargs(self):
        '''
        This function adds some default args to fmin, if we have not explicitly
        specified them.
        '''
        self._fminargs['maxfun'] = self._fminargs.get('maxfun', 1000)
        self._fminargs['disp'] = self._fminargs.get('disp', 1)
        # even when fmin displays outputs, we set non-root display to none
        if not mpi.is_root():
            self._fminargs['disp'] = 0
            
    @staticmethod
    def obj(wb, solver):
        """The objective function to be used by fmin
        """
        raise NotImplementedError
    
    def presolve(self, X, Y, weight, param_init):
        """This function is called before we call lbfgs. It should return a
        vector that is the initialization of the lbfgs, and does any preparation
        (such as creating caches) for the optimization.
        """
        raise NotImplementedError
    
    def postsolve(self, lbfgs_result):
        """This function deals with the post-processing of the lbfgs result. It
        should return the optimal parameter for the classifier.
        """
        raise NotImplementedError
    
    def solve(self, X, Y, weight = None, param_init = None, presolve = True):
        """The solve function
        """
        if presolve:
            param_init = self.presolve(X, Y, weight, param_init)
        logging.debug('Solver: running lbfgs...')
        result = _FMIN(self.__class__.obj, param_init, 
                       args=[self], **self._fminargs)
        return self.postsolve(result)


class SolverMC(Solver):
    '''SolverMC is a multi-dimensional wrapper
    For the input Y, it could be either a vector of the labels
    (starting from 0), or a matrix whose values are -1 or 1. You 
    need to manually make sure that the input Y format is consistent
    with the loss function though.
    '''
    @staticmethod
    def flatten_params(params):
        if type(params) is np.array:
            return params
        elif type(params) is list or type(params) is tuple:
            return np.hstack((p.flatten() for p in params))
        else:
            raise TypeError, "Unknown input type: %s." % (repr(type(params)))

    def presolve(self, X, Y, weight, param_init):
        self._iter = 0
        self._X = X.reshape((X.shape[0],np.prod(X.shape[1:])))
        if len(Y.shape) == 1:
            self._K = mpi.COMM.allreduce(Y.max(), op=max) + 1
        else:
            # We treat Y as a two-dimensional matrix
            Y = Y.reshape((Y.shape[0],np.prod(Y.shape[1:])))
            self._K = Y.shape[1]
        self._Y = Y
        self._weight = weight
        # compute the number of data
        if weight is None:
            self._num_data = mpi.COMM.allreduce(X.shape[0])
        else:
            self._num_data = mpi.COMM.allreduce(weight.sum())
        self._dim = self._X.shape[1]
        self._pred = np.empty((X.shape[0], self._K), dtype = X.dtype)
        if param_init is None:
            param_init = np.zeros(self._K * (self._dim+1))
        else:
            # the initialization is w and b
            param_init = SolverMC.flatten_params(param_init) 
        # gradient cache
        self._glocal = np.empty(param_init.shape)
        self._g = np.empty(param_init.shape)
        # depending on the loss function, we choose whether we want to do
        # gpred cache
        if len(inspect.getargspec(self.loss)[0]) == 5:
            logging.debug('Using gpred cache')
            self.gpredcache = True
            self._gpred = np.empty((X.shape[0], self._K))
            self._gpredcache = []
        else:
            self.gpredcache = False
        # just to make sure every node is on the same page
        mpi.COMM.Bcast(param_init)
        # for debugging, we report the initial function value.
        f = SolverMC.obj(param_init, self)[0]
        logging.debug("Initial function value: %f." % f)
        return param_init
    
    def unflatten_params(self, wb):
        K = self._K
        w = wb[: K * self._dim].reshape(self._dim, K).copy()
        b = wb[K * self._dim :].copy()
        return w, b
    
    def postsolve(self, lbfgs_result):
        wb = lbfgs_result[0]
        logging.debug("Final function value: %f." % lbfgs_result[1])
        return self.unflatten_params(wb)
    
    @staticmethod
    def obj(wb,solver):
        '''
        The objective function used by fmin
        '''
        # obtain w and b
        K = solver._K
        dim = solver._dim
        w = wb[:K*dim].reshape((dim, K))
        b = wb[K*dim:]
        # pred is a matrix of size [num_datalocal, K]
        mathutil.dot(solver._X, w, out = solver._pred)
        solver._pred += b
        # compute the loss function
        if solver.gpredcache:
            flocal,gpred = solver.loss(solver._Y, solver._pred, solver._weight,
                                       solver._gpred, solver._gpredcache,
                                       **solver._lossargs)
        else:
            flocal,gpred = solver.loss(solver._Y, solver._pred, solver._weight,
                                       **solver._lossargs)
        mathutil.dot(solver._X.T, gpred,
                     out = solver._glocal[:K*dim].reshape(dim, K))
        solver._glocal[K*dim:] = gpred.sum(axis=0)
        # add regularization term, but keep in mind that we have multiple nodes
        freg, greg = solver.reg(w, **solver._regargs)
        if mpi.is_root():
            flocal += solver._num_data * solver._gamma * freg
            solver._glocal[:K*dim] += solver._num_data * solver._gamma \
                          * greg.ravel()
        # do mpi reduction
        mpi.barrier()
        f = mpi.COMM.allreduce(flocal)
        mpi.COMM.Allreduce(solver._glocal, solver._g)
        return f, solver._g


class SolverStochastic(Solver):
    """A stochastic solver following existing papers in the literature. The
    method creates minibatches and runs LBFGS (using SolverMC) or Adagrad for
    a few iterations, then moves on to the next minibatch.
    
    The solver should have the following args:
        'mode': the basic solver. Currently 'LBFGS' or 'Adagrad', with LBFGS
            as default.
        'base_lr': the base learning rate (if using Adagrad as the solver).
        'minibatch': the batch size
        'num_iter': the number of iterations to carry out. Note that if you
            use LBFGS, how many iterations to carry out on one minibatch is
            defined in the max_iter parameter defined in fminargs. If you use
            Adagrad, each minibatch will be used once to compute the function
            value and the gradient, and then discarded.
        'fine_tune': if a number larger than 0, we perform the corresponding
            steps of complete LBFGS after the stochastic steps finish.
        'callback': the callback function after each LBFGS iteration. It
            should take the result output by the solver.solve() function and
            return a float number. If callback is a list, then every entry in
            the list is a callback function, and they will be carried out
            sequentially.
    """
    @staticmethod
    def synchronized_shuffle(arrays):
        """Do a synchronized shuffle of a set of arrays along their first axis
        """
        rand_state = np.random.get_state()
        for arr in arrays:
            if arr is None:
                continue
            np.random.set_state(rand_state)
            np.random.shuffle(arr)
    
    def solve(self, X, Y, weight = None, param_init = None):
        """The solve function
        """
        num_data = mpi.COMM.allreduce(X.shape[0])
        # make sure the minibatch size is distributed according to the local
        # data size
        minibatch = int(self._args['minibatch'] * X.shape[0] / num_data)
        if minibatch >= X.shape[0]:
            raise ValueError, "Minibatch size is larger than the data size."
        # deal with minibatch
        SolverStochastic.synchronized_shuffle((X, Y, weight))
        pointer = 0
        mode = self._args.get('mode', 'lbfgs').lower()
        # even when we use Adagrad we create a solver_basic to deal with
        # function value and gradient computation, etc.
        solver_basic = SolverMC(self._gamma, self.loss, self.reg,
                self._args, self._lossargs, self._regargs,
                self._fminargs)
        param = param_init
        localweight = None
        timer = util.Timer()
        for iter in range(self._args['num_iter']):
            logging.info('Solver: running lbfgs round %d, elapsed %s' % \
                    (iter, timer.lap()))
            if (X.shape[0] - pointer < minibatch):
                # reshuffle
                SolverStochastic.synchronized_shuffle((X, Y, weight))
                pointer = 0
            if weight is not None:
                localweight = weight[pointer:pointer + minibatch]
            # carry out the computation
            if mode == 'lbfgs':
                param = solver_basic.solve(X[pointer:pointer + minibatch],
                        Y[pointer:pointer + minibatch], localweight, param)
            else:
                # adagrad: compute gradient and update
                if iter == 0:
                    # we need to build the cache in solver_basic as well as
                    # the accumulated gradients
                    param_flat = solver_basic.presolve(\
                            X[pointer:pointer + minibatch],
                            Y[pointer:pointer + minibatch],
                            localweight, param)
                    accum_grad = np.zeros_like(param_flat) + \
                            np.finfo(np.float64).eps
                f, g = SolverMC.obj(param_flat, solver_basic)
                # update
                if self._fminargs.get('disp', 0) > 0:
                    logging.debug('iter %d f = %f |g| = %f' % (iter, f, 
                            np.sqrt(np.dot(g, g) / g.size)))
                accum_grad += g * g
                # we are MINIMIZING, so go against the gradient direction
                base_lr = self._args['base_lr']
                param_flat -= g / np.sqrt(accum_grad) * base_lr
                param = solver_basic.unflatten_params(param_flat)
            # advance to next minibatch
            pointer += minibatch
            callback = self._args.get('callback', None)
            if callback is None:
                continue
            if type(callback) is not list:
                cb_val = callback(param)
                logging.debug('Round %d callback value: %f.' % (iter, cb_val))
            else:
                for i, cb_func in enumerate(callback):
                    cb_val = cb_func(param)
                    logging.debug('Round %d callback #%d value: %f.' \
                            % (iter, i, cb_val))
        # the stochastic part is done. See if we want to do fine-tuning.
        finetune = self._args.get('fine_tune', 0)
        if finetune > 0:
            solver_basic._fminargs['maxfun'] = int(finetune)
            param = solver_basic.solve(X, Y, weight, param)
        return param


class Loss(object):
    """LOSS defines commonly used loss functions
    For all loss functions:
    Input:
        Y:    a vector or matrix of true labels
        pred: prediction, has the same shape as Y.
    Return:
        f: the loss function value
        g: the gradient w.r.t. pred, has the same shape as pred.
    """
    def __init__(self):
        """All functions in Loss should be static
        """
        raise NotImplementedError, "Loss should not be instantiated!"
     
    @staticmethod
    def loss_l2(Y, pred, weight, **kwargs):
        '''
        The l2 loss: f = ||Y - pred||_{fro}^2
        '''
        diff = pred - Y
        if weight is None:
            return np.dot(diff.flat, diff.flat), 2.*diff 
        else:
            return np.dot((diff**2).sum(1), weight), \
                   2.*diff*weight[:,np.newaxis]
         
    @staticmethod
    def loss_hinge(Y, pred, weight, **kwargs):
        '''The SVM hinge loss. Input vector Y should have values 1 or -1
        '''
        margin = np.maximum(0., 1. - Y * pred)
        if weight is None:
            f = margin.sum()
            g = - Y * (margin>0)
        else:
            f = np.dot(weight, margin).sum()
            g = - Y * weight[:, np.newaxis] * (margin>0)
        return f, g
     
    @staticmethod
    def loss_squared_hinge(Y,pred,weight,**kwargs):
        ''' The squared hinge loss. Input vector Y should have values 1 or -1
        '''
        margin = np.maximum(0., 1. - Y * pred)
        if weight is None:
            return np.dot(margin.flat, margin.flat), -2. * Y * margin
        else:
            wm = weight[:, np.newaxis] * margin
            return np.dot(wm.flat, margin.flat), -2. * Y * wm
 
    @staticmethod
    def loss_bnll(Y,pred,weight,**kwargs):
        '''
        the BNLL loss: f = log(1 + exp(-y * pred))
        '''
        # expnyp is exp(-y * pred)
        expnyp = mathutil.exp(-Y*pred)
        expnyp_plus = 1. + expnyp
        if weight is None:
            return np.sum(np.log(expnyp_plus)), -Y * expnyp / expnyp_plus
        else:
            return np.dot(weight, np.log(expnyp_plus)).sum(), \
                   - Y * weight * expnyp / expnyp_plus
 
    @staticmethod
    def loss_multiclass_logistic(Y, pred, weight, **kwargs):
        """The multiple class logistic regression loss function
         
        The input Y should be a 0-1 matrix 
        """
        # normalized prediction and avoid overflowing
        prob = pred - pred.max(axis=1)[:,np.newaxis]
        mathutil.exp(prob, out=prob)
        prob /= prob.sum(axis=1)[:, np.newaxis]
        g = prob - Y
        # take the log
        mathutil.log(prob, out=prob)
        return -np.dot(prob.flat, Y.flat), g


class Loss2(object):
    """LOSS2 defines commonly used loss functions, rewritten with the gradient
    value cached (provided by the caller) for large-scale problems to save
    memory allocation / deallocation time.
    
    For all loss functions:
    Input:
        Y:    a vector or matrix of true labels
        pred: prediction, has the same shape as Y.
        weight: the weight for each data point.
        gpred: the pre-assigned numpy array to store the gradient. We force
            gpred to be preassigned to save memory allocation time in large
            scales.
        cache: a list (initialized with []) containing any misc cache that
            the loss function computation uses.
    Return:
        f: the loss function value
        gpred: the gradient w.r.t. pred, has the same shape as pred.
    """
    def __init__(self):
        """All functions in Loss should be static
        """
        raise NotImplementedError, "Loss should not be instantiated!"
    
    @staticmethod
    def loss_l2(Y, pred, weight, gpred, cache, **kwargs):
        '''
        The l2 loss: f = ||Y - pred||_{fro}^2
        '''
        if weight is None:
            gpred[:] = pred
            gpred -= Y
            f = np.dot(gpred.flat, gpred.flat)
            gpred *= 2.
        else:
            # we aim to minimize memory usage and avoid re-allocating large 
            # matrices.
            gpred[:] = pred
            gpred -= Y
            gpred **= 2
            f = np.dot(gpred.sum(1), weight)
            gpred[:] = pred
            gpred -= Y
            gpred *= 2. * weight[:, np.newaxis]
        return f, gpred
    
    @staticmethod
    def loss_hinge(Y, pred, weight, gpred, cache, **kwargs):
        '''The SVM hinge loss. Input vector Y should have values 1 or -1
        '''
        gpred[:] = pred
        gpred *= Y
        gpred *= -1
        gpred += 1.
        np.clip(gpred, 0, np.inf, out=gpred)
        if weight is None:
            f = gpred.sum()
            gpred[:] = (gpred > 0)
            gpred *= Y
            gpred *= -1
        else:
            f = np.dot(weight, gpred.sum(axis=1))
            gpred[:] = (gpred > 0)
            gpred *= Y
            gpred *= - weight[:, np.newaxis]
        return f, gpred
    
    @staticmethod
    def loss_squared_hinge(Y, pred, weight, gpred, cache, **kwargs):
        ''' The squared hinge loss. Input vector Y should have values 1 or -1
        '''
        gpred[:] = pred
        gpred *= Y
        gpred *= -1
        gpred += 1.
        np.clip(gpred, 0, np.inf, out=gpred)
        if weight is None:
            f = np.dot(gpred.flat, gpred.flat)
            gpred *= Y
            gpred *= -2
        else:
            gprednorm = inner1d(gpred,gpred)
            f = np.dot(gprednorm, weight)
            gpred *= Y
            gpred *= (-2 * weight[:, np.newaxis])
        return f, gpred

    @staticmethod
    def loss_multiclass_logistic(Y, pred, weight, gpred, cache, **kwargs):
        """The multiple class logistic regression loss function
        
        The input Y should be a 0-1 matrix 
        """
        if len(cache) == 0:
            cache.append(np.empty_like(pred))
        cache[0].resize(pred.shape)
        prob = cache[0]
        # normalize prediction to avoid overflowing
        prob = pred.copy()
        prob -= pred.max(axis=1)[:,np.newaxis]
        mathutil.exp(prob, out=prob)
        prob /= prob.sum(axis=1)[:, np.newaxis]
        gpred[:] = prob
        gpred -= Y
        # take the log
        mathutil.log(prob, out=prob)
        return -np.dot(prob.flat, Y.flat), gpred


class Reg(object):
    '''
    REG defines commonly used regularization functions
    For all regularization functions:
    Input:
        w: the weight vector, or the weight matrix in the case of multiple classes
    Return:
        f: the regularization function value
        g: the gradient w.r.t. w, has the same shape as w.
    '''
    @staticmethod
    def reg_l2(w,**kwargs):
        '''
        l2 regularization: ||w||_2^2
        '''
        return np.dot(w.flat, w.flat), 2.*w

    @staticmethod
    def reg_l1(w,**kwargs):
        '''
        l1 regularization: ||w||_1
        '''
        g = np.sign(w)
        # subgradient
        g[g==0] = 0.5
        return np.abs(w).sum(), g

    @staticmethod
    def reg_elastic(w, **kwargs):
        '''
        elastic net regularization: (1-alpha) * ||w||_2^2 + alpha * ||w||_1
        kwargs['alpha'] is the balancing weight, default 0.5
        '''
        alpha1 = kwargs.get('alpha', 0.5)
        alpha2 = 1. - alpha1
        f1, g1 = Reg.reg_l1(w, **kwargs)
        f2, g2 = Reg.reg_l2(w, **kwargs)
        return f1 * alpha1 + f2 * alpha2, g1 * alpha1 + g2 * alpha2

class Evaluator(object):
    """Evaluator implements some commonly-used criteria for evaluation
    """
    @staticmethod
    def mse(Y, pred, axis=None):
        """Return the mean squared error of the true value and the prediction
        Input:
            Y, pred: the true value and the prediction
            axis: (optional) if Y and pred are matrices, you can specify the
                axis along which the mean is carried out.
        """
        return ((Y - pred) ** 2).mean(axis=axis)
    
    @staticmethod
    def accuracy(Y, pred):
        """Computes the accuracy
        Input: 
            Y, pred: two vectors containing discrete labels. If either is a
            matrix instead of a vector, then argmax is used to get the discrete
            labels.
        """
        if pred.ndim == 2:
            pred = pred.argmax(axis=1)
        if Y.ndim == 2:
            Y = Y.argmax(axis=1)
        correct = mpi.COMM.allreduce((Y==pred).sum())
        num_data = mpi.COMM.allreduce(len(Y))
        return float(correct) / num_data
    
    @staticmethod
    def confusion_table(Y, pred):
        """Computes the confusion table
        Input:
            Y, pred: two vectors containing discrete labels
        Output:
            table: the confusion table. table[i,j] is the number of data points
                that belong to i but predicted as j
        """
        if pred.ndim == 2:
            pred = pred.argmax(axis=1)
        if Y.ndim == 2:
            Y = Y.argmax(axis=1)
        num_classes = Y.max() + 1
        table = np.zeros((num_classes, num_classes))
        for y, p in zip(Y, pred):
            table[y,p] += 1
        return table
    
    @staticmethod
    def accuracy_class_averaged(Y, pred):
        """Computes the accuracy, but averaged over classes instead of averaged
        over data points.
        Input:
            Y: the ground truth vector
            pred: a vector containing the predicted labels. If pred is a matrix
            instead of a vector, then argmax is used to get the discrete label.
        """
        if pred.ndim == 2:
            pred = pred.argmax(axis=1)
        num_classes = Y.max() + 1
        accuracy = 0.0
        correct = (Y == pred).astype(np.float)
        for i in range(num_classes):
            idx = (Y == i)
            accuracy += correct[idx].mean()
        accuracy /= num_classes
        return accuracy

    @staticmethod
    def top_k_accuracy(Y, pred, k):
        """Computes the top k accuracy
        Input:
            Y: a vector containing the discrete labels of each datum
            pred: a matrix of size len(Y) * num_classes, each row containing the
                real value scores for the corresponding label. The classes with
                the highest k scores will be considered.
        """
        if k > pred.shape[1]:
            logging.warning("Warning: k is larger than the number of classes"
                            "so the accuracy would always be one.")
        top_k_id = np.argsort(pred, axis=1)[:, -k:]
        match = (top_k_id == Y[:, np.newaxis])
        correct = mpi.COMM.allreduce(match.sum())
        num_data = mpi.COMM.allreduce(len(Y))
        return float(correct) / num_data
    
    @staticmethod
    def average_precision(Y, pred):
        """Average Precision for binary classification
        """
        # since we need to compute the precision recall curve, we have to
        # compute this on the root node.
        Y = mpi.COMM.gather(Y)
        pred = mpi.COMM.gather(pred)
        if mpi.is_root():
            Y = np.hstack(Y)
            pred = np.hstack(pred)
            precision, recall, _ = metrics.precision_recall_curve(
                    Y == 1, pred)
            ap = metrics.auc(recall, precision)
        else:
            ap = None
        mpi.barrier()
        return mpi.COMM.bcast(ap)
    
    @staticmethod
    def average_precision_multiclass(Y, pred):
        """Average Precision for multiple class classification
        """
        K = pred.shape[1]
        aps = [Evaluator.average_precision(Y==k, pred[:,k]) for k in range(K)]
        return np.asarray(aps).mean()

'''
Utility functions that wraps often-used functions
'''

def svm_onevsall(X, Y, gamma, weight = None, **kwargs):
    if Y.ndim == 1:
        Y = to_one_of_k_coding(Y)
    solver = SolverMC(gamma, Loss.loss_hinge, Reg.reg_l2, **kwargs)
    return solver.solve(X, Y, weight)

def l2svm_onevsall(X, Y, gamma, weight = None, **kwargs):
    if Y.ndim == 1:
        Y = to_one_of_k_coding(Y)
    solver = SolverMC(gamma, Loss.loss_squared_hinge, Reg.reg_l2, **kwargs)
    return solver.solve(X, Y, weight)

def elasticnet_svm_onevsall(X, Y, gamma, weight = None, alpha = 0.5, **kwargs):
    if Y.ndim == 1:
        Y = to_one_of_k_coding(Y)
    solver = SolverMC(gamma, Loss.loss_squared_hinge, Reg.reg_elastic, 
                      lossargs = {'alpha': alpha}, **kwargs)
    return solver.solve(X, Y, weight)
