''' mpi implements common util functions based on mpi4py.
'''

import glob
import logging
from mpi4py import MPI
import numpy as np
import os
import random
import socket
import time

# MPI
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()
_HOST_RAW = socket.gethostname()
# this is the hack that removes things like ".icsi.berkeley.edu"
if _HOST_RAW.find('.') == -1:
    _HOST = _HOST_RAW
else:
    _HOST = _HOST_RAW[:_HOST_RAW.find('.')]
_MPI_PRINT_MESSAGE_TAG = 560710

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
    '''
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
        

def get_segments(total):
    """Get the segments for each local node
    """
    return [int(total * i / float(SIZE)) for i in range(SIZE+1)]
    
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
            COMM.Send(mat[segments[i]:segments[i+1]], dest=i)
        data = mat[:segments[1]].copy()
    else:
        data = np.empty((segments[RANK+1] - segments[RANK],) + shape,
                        dtype = dtype)
        COMM.Recv(data)
    return data

def distribute_list(source):
    """Distributes the list from root to individual nodes
    """
    # quick check
    if SIZE == 1:
        return source
    if is_root():
        segments = get_segments(len(source))
        for i in range(1,SIZE):
            send_list = source[segments[i]:segments[i+1]]
            COMM.send(send_list, dest=i)
        data = source[:segments[1]]
        del source
    else:
        data = COMM.recv()
    return data
        
def dump_matrix(mat, filename):
    """Dumps the matrix distributed over machines to one single file
    """
    if SIZE == 1:
        with open(filename,'w') as fid:
            np.save(mat, fid)
    else:
        mat_sizes = COMM.gather(mat.shape[0])
        if is_root():
            total_size = sum(mat_sizes)
            mat_reduced = np.empty((total_size,) + mat.shape[1:],
                                   dtype = mat.dtype)
            start = mat_sizes[0]
            mat_reduced[:start] = mat
            for i in range(1,SIZE):
                COMM.Recv(mat_reduced[start:start+mat_sizes[i]], source = i)
                start += mat_sizes[i]
            with open(filename,'w') as fid:
                np.save(fid, mat_reduced)
        else:
            COMM.Send(mat, dest = 0)
        barrier()

def load_matrix(filename):
    """Load a matrix from a single matrix, and distribute it to each node
    numpy supports memmap so each node will simply load its own part
    """
    if SIZE == 1:
        data = np.load(filename)
        return data
    raw_data = np.load(filename, mmap_mode = 'r')
    total_size = raw_data.shape[0]
    segments = get_segments(total_size)
    data = np.array(raw_data[segments[RANK]:segments[RANK+1]])
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
    np.save(my_filename, mat)
    
def load_matrix_multi(filename):
    """Loads the matrix previously dumped by dump_matrix_multi. The MPI size 
    might be different. The stored files are in the format
    filename-xxxxx-of-xxxxx, which we obtain using glob.
    """
    files= glob.glob('%s-?????-of-?????.npy' % (filename))
    N = len(files)
    logging.debug("Loading the matrix from %d parts" % N)
    if N == SIZE:
        # we are lucky
        mat = np.load('%s-%05d-of-%05d.npy' % (filename, RANK, SIZE))
        return mat
    else:
        # we will load the length of the data, and then try to distribute them
        # as even as possible.
        if RANK == 0:
            # the root will first taste each file
            sizes = np.array([np.load('%s-%05d-of-%05d.npy' % (filename, i, N),
                                      mmap_mode='r').shape[0]
                              for i in range(N)])
            shape = np.load('%s-%05d-of-%05d.npy' % (filename, 0, N),
                                      mmap_mode='r').shape[1:]
        else:
            sizes = None
            shape = None
        barrier()
        sizes = COMM.bcast(sizes)
        shape = COMM.bcast(shape)
        total = sizes.sum()
        segments = get_segments(total)
        # now, each node opens the file that overlaps with its data, and reads
        # the contents.
        my_start = segments[RANK]
        my_end = segments[RANK+1]
        my_size = my_end - my_start
        mat = np.empty((my_size,) + shape)
        f_start = -sizes[0]
        f_end = 0
        for i, size in enumerate(sizes):
            f_start += size
            f_end += size
            if f_start < my_end and f_end > my_start:
                file_mat = np.load('%s-%05d-of-%05d.npy' % (filename, i, N),
                                    mmap_mode='r')
                mat[max(f_start - my_start, 0):min(f_end, my_end)-my_start] = \
                        file_mat[max(my_start-f_start,0):\
                                 min(f_end, my_end) - f_start]
        return mat


if __name__ == "__main__":
    pass
