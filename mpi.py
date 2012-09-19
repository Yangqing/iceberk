''' mpi implements common util functions based on mpi4py.
'''

import cPickle as pickle
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
        
def distribute(mat):
    """Distributes the mat from root to individual nodes
    
    The data will be distributed along the first axis, as even as possible.
    """
    # quick check
    if SIZE == 1:
        return mat
    if is_root():
        total = float(mat.shape[0])
        shape = mat.shape[1:]
        segments = [int(total * i / SIZE) for i in range(SIZE+1)]
        dtype = mat.dtype
    else:
        shape = None
        segments = None
        dtype = None
    shape = COMM.bcast(shape)
    dtype = COMM.bcast(dtype)
    segments = COMM.bcast(segments)
    if is_root():
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
        total = float(len(source))
        segments = [int(total * i / SIZE) for i in range(SIZE+1)]
        for i in range(1,SIZE):
            send_list = source[segments[i]:segments[i+1]]
            COMM.send(send_list, dest=i)
        data = source[:segments[1]]
    else:
        data = COMM.recv()
    return data
        
def dump_matrix(mat, filename):
    """Dumps the matrix distributed over machines to one file
    """
    if SIZE == 1:
        with open(filename,'w') as fid:
            pickle.dump(mat, fid)
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
                pickle.dump(mat_reduced, fid)
        else:
            COMM.Send(mat, dest = 0)
        barrier()

def load_matrix(filename):
    """Load a matrix from a single pickle, and distribute it to each node
    """
    if is_root:
        with open(filename,'r') as fid:
            data = pickle.load(fid)
    else:
        data = None
    barrier()
    return distribute(data)

if __name__ == "__main__":
    pass
