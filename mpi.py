''' mpi implements common util functions based on mpi4py.
'''

import cPickle as pickle
import glob
import logging
import numpy as np
import os
import random
import socket
import sys
import time

# MPI
try:
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
except Exception, e:
    sys.stderr.write(\
            "Warning: I cannot import mpi4py. Using a dummpy single noded "\
            "implementation instead. The program will run in single node mode "\
            "even if you executed me with mpirun or mpiexec.\n")
    sys.stderr.write("We STRONGLY recommend you to try to install mpi and "\
                     "mpi4py.\n")
    sys.stderr.write("mpi4py exception message is:")
    sys.stderr.write(repr(Exception) + repr(e))
    from _mpi_dummy import COMM

RANK = COMM.Get_rank()
SIZE = COMM.Get_size()
_HOST_RAW = socket.gethostname()
# this is the hack that removes things like ".icsi.berkeley.edu"
if _HOST_RAW.find('.') == -1:
    HOST = _HOST_RAW
else:
    HOST = _HOST_RAW[:_HOST_RAW.find('.')]
_MPI_PRINT_MESSAGE_TAG = 560710
_MPI_BUFFER_LIMIT = 1073741824

# we need to set the random seed different for each mpi instance
random.seed(time.time() * RANK)

def mkdir(dirname):
    '''make a directory. Avoid race conditions.
    '''
    try:
        os.makedirs(dirname)
    except OSError:
        pass
    except:
        raise


def agree(decision):
    """agree() makes the decision consistent by propagating the decision of the
    root to everyone
    """
    return COMM.bcast(decision)


def elect():
    '''elect() randomly chooses a node from all the nodes as the president.
    Input:
        None
    Output:
        the rank of the president
    '''
    president = COMM.bcast(np.random.randint(SIZE))
    return president


def is_president():
    ''' Returns true if I am the president, otherwise return false
    '''
    return (RANK == elect())


def is_root():
    '''returns if the current node is root
    '''
    return RANK == 0


def barrier(tag=0, sleep=0.01):
    ''' A better mpi barrier
    
    The original MPI.comm.barrier() may cause idle processes to still occupy
    the CPU, while this barrier waits.
    '''
    if SIZE == 1: 
        return 
    mask = 1 
    while mask < SIZE: 
        dst = (RANK + mask) % SIZE 
        src = (RANK - mask + SIZE) % SIZE 
        req = COMM.isend(None, dst, tag) 
        while not COMM.Iprobe(src, tag): 
            time.sleep(sleep) 
        COMM.recv(None, src, tag) 
        req.Wait() 
        mask <<= 1


def root_log_level(level, name = None):
    """set the log level on root. 
    Input:
        level: the logging level, such as logging.DEBUG
        name: (optional) the logger name
    """
    if is_root():
        logging.getLogger(name).setLevel(level)

def log_level(level, name = None):
    """set the log level on all nodes. 
    Input:
        level: the logging level, such as logging.DEBUG
        name: (optional) the logger name
    """
    logging.getLogger(name).setLevel(level)


def safe_send_matrix(mat, dest=0, tag=0):
    """A safe send that deals with the mpi4py 2GB limit. should be paired with
    safe_recv_matrix. The input mat should be C_CONTIGUOUS. To be safe, we send
    the matrix in 1GB chunks.
    """
    num_batches = int((mat.nbytes - 1) / _MPI_BUFFER_LIMIT + 1)
    if num_batches == 1:
        COMM.Send(mat, dest, tag)
    else:
        logging.debug("The buffer is larger than 1GB, sending in chunks...")
        batch_size = int(mat.shape[0] / num_batches)
        for i in range(num_batches):
            COMM.Send(mat[batch_size*i:batch_size*(i+1)], dest, tag)
        # send the remaining part
        if mat.shape[0] > batch_size * num_batches:
            COMM.Send(mat[batch_size * num_batches:], dest, tag)


def safe_recv_matrix(mat, source=0, tag=0, status=None):
    """A safe recv that deals with the mpi4py 2GB limit. should be paired with
    safe_send_matrix. The input mat should be C_CONTIGUOUS. To be safe, we recv
    the matrix in 1GB chunks.
    """
    num_batches = int((mat.nbytes - 1) / _MPI_BUFFER_LIMIT + 1)
    if num_batches == 1:
        COMM.Recv(mat, source, tag, status)
    else:
        logging.debug("The buffer is larger than 1GB, sending in chunks...")
        batch_size = int(mat.shape[0] / num_batches)
        for i in range(num_batches):
            COMM.Recv(mat[batch_size*i:batch_size*(i+1)], source, tag, status)
        # send the remaining part
        if mat.shape[0] > batch_size * num_batches:
            COMM.Recv(mat[batch_size * num_batches:], source, tag, status)


def get_segments(total, inverse = False):
    """Get the segments for each local node.
    
    Input:
        inverse: (optional) if set True, also return the inverse index of each
            element in 0:total.
    Output:
        segments: a list of size SIZE+1, where segments[i]:segments[i+1] 
            specifies the range that the local node is responsible for.
        inv: (only if inverse=True) a list of size total, where inv[i] is the
            rank of the node that is responsible for element i.
    """
    if total < SIZE:
        raise ValueError, \
                "The total number %d should be larger than the mpi size %d." % \
                (total, SIZE)
    segments = [int(total * i / float(SIZE)) for i in range(SIZE+1)]
    if inverse:
        inv = sum([[i] * (segments[i+1] - segments[i]) 
                   for i in range(SIZE)], [])
        return segments, inv
    else:
        return segments


def distribute(mat):
    """Distributes the mat from root to individual nodes
    
    The data will be distributed along the first axis, as even as possible.
    You should make sure that the matrix is in C-contiguous format.
    """
    # quick check
    if SIZE == 1:
        return mat
    if is_root():
        shape = mat.shape[1:]
        segments = get_segments(mat.shape[0])
        dtype = mat.dtype
    else:
        shape = None
        segments = None
        dtype = None
    shape = COMM.bcast(shape)
    dtype = COMM.bcast(dtype)
    segments = COMM.bcast(segments)
    if is_root():
        if mat.flags['C_CONTIGUOUS'] != True:
            logging.warning('Warning: mat is not contiguous.')
            mat = np.ascontiguousarray(mat)
        for i in range(1,SIZE):
            safe_send_matrix(mat[segments[i]:segments[i+1]], dest=i)
        data = mat[:segments[1]].copy()
    else:
        data = np.empty((segments[RANK+1] - segments[RANK],) + shape,
                        dtype = dtype)
        safe_recv_matrix(data)
    return data


def distribute_list(source):
    """Distributes the list from root to individual nodes
    """
    # quick check
    if SIZE == 1:
        return source
    if is_root():
        length = len(source)
        if length == 0:
            logging.warning("Warning: List has length 0")
    else:
        length = 0
    length = COMM.bcast(length)
    if length == 0:
        return []
    segments = get_segments(length)
    if is_root():
        for i in range(1,SIZE):
            send_list = source[segments[i]:segments[i+1]]
            COMM.send(send_list, dest=i)
        data = source[:segments[1]]
        del source
    else:
        data = COMM.recv()
    return data


def dump_matrix(mat, filename):
    """Dumps the matrix distributed over machines to one single file.
    
    We do NOT recommend using this - it causes a lot of communications since
    all data need to be transferred to root before writing to disk. Instead,
    use dump_matrix_multi which stores the matrix to multiple chunks.
    """
    if SIZE == 1:
        with open(filename,'w') as fid:
            np.save(fid, mat)
    else:
        mat_sizes = COMM.gather(mat.shape[0])
        if is_root():
            total_size = sum(mat_sizes)
            mat_reduced = np.empty((total_size,) + mat.shape[1:],
                                   dtype = mat.dtype)
            start = mat_sizes[0]
            mat_reduced[:start] = mat
            for i in range(1,SIZE):
                safe_recv_matrix(mat_reduced[start:start+mat_sizes[i]],
                                 source = i)
                start += mat_sizes[i]
            with open(filename,'w') as fid:
                np.save(fid, mat_reduced)
        else:
            safe_send_matrix(mat, dest = 0)
        barrier()


def load_matrix(filename):
    """Load a matrix from a single matrix, and distribute it to each node
    numpy supports memmap so each node will simply load its own part
    """
    if SIZE == 1:
        data = np.load(filename)
        return data
    try:
        raw_data = np.load(filename, mmap_mode = 'r')
    except IOError:
        # we try to load the filename with '.npy' affix. If we fail again,
        # raise IOError.
        raw_data = np.load(filename + '.npy', mmap_mode = 'r')
    total_size = raw_data.shape[0]
    segments = get_segments(total_size)
    data = np.empty((segments[RANK+1] - segments[RANK],) + raw_data.shape[1:])
    data[:] = raw_data[segments[RANK]:segments[RANK+1]]
    barrier()
    return data


def dump_matrix_multi(mat, filename):
    """Dumps the matrix distributed over machines to multiple files, one per
    MPI node.
    """
    if SIZE > 99999:
        # this usually won't happen, but we leave the sanity check here
        raise ValueError, 'I cannot deal with too many MPI instances.'
    logging.debug("Dumping the matrix to %d parts" % SIZE)
    my_filename = '%s-%05d-of-%05d.npy' % (filename, RANK, SIZE)
    mkdir(os.path.dirname(filename))
    np.save(my_filename, mat)


def load_matrix_multi(filename, N = None):
    """Loads the matrix previously dumped by dump_matrix_multi. The MPI size 
    might be different. The stored files are in the format
    filename-xxxxx-of-xxxxx, which we obtain using glob.
    
    Input:
        N: (optional) if given, specify the number of parts the matrix is
            separated too. Otherwise, the number is automatically inferred by
            listing all the files using regexp matching.
    """
    if N is None:
        # figure out the size
        files= glob.glob('%s-?????-of-?????.npy' % (filename))
        N = len(files)
    else:
        logging.debug("Loading the matrix from %d parts" % N)
        # we will load the length of the data, and then try to distribute them
        # as even as possible.
        if RANK == 0:
            # the root will first taste each file
            sizes = np.array([np.load('%s-%05d-of-%05d.npy' % (filename, i, N),
                                      mmap_mode='r').shape[0]
                              for i in range(N)])
            temp = np.load('%s-%05d-of-%05d.npy' % (filename, 0, N),
                                      mmap_mode='r')
            shape = temp.shape[1:]
            dtype = temp.dtype
        else:
            sizes = None
            shape = None
            dtype = None
        barrier()
        sizes = COMM.bcast(sizes)
        shape = COMM.bcast(shape)
        dtype = COMM.bcast(dtype)
        total = sizes.sum()
        segments = get_segments(total)
        # now, each node opens the file that overlaps with its data, and reads
        # the contents.
        my_start = segments[RANK]
        my_end = segments[RANK+1]
        my_size = my_end - my_start
        mat = np.empty((my_size,) + shape, dtype = dtype)
        mat = np.empty((my_size,) + shape)
        f_start = 0
        f_end = 0
        for i, size in enumerate(sizes):
            f_end += size
            if f_start < my_end and f_end > my_start:
                file_mat = np.load('%s-%05d-of-%05d.npy' % (filename, i, N),
                                    mmap_mode='r')
                mat[max(f_start - my_start, 0):\
                    min(f_end - my_start, my_size)] = \
                        file_mat[max(my_start - f_start,0):\
                                 min(my_end - f_start, size)]
            f_start += size
        return mat


def root_pickle(obj, filename):
    if is_root():
        pickle.dump(obj, open(filename, 'w'))

if __name__ == "__main__":
    pass
