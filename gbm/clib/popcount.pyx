"""
Compile with gcc flag -mpopcnt
"""
import numpy as np
cimport numpy as np
cimport cython
from libc.stdint cimport uint32_t, uint8_t

cdef extern int __builtin_popcount(unsigned int) nogil

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _inplace_popcount_32_2d(uint32_t[:, :] arr) nogil:
    """
    Iterates over the elements of an 2D array and replaces them by their popcount
    Parameters:
    -----------
    arr: numpy_array, dtype=np.uint32, shape=(m, n)
       The array for which the popcounts should be computed.
    """
    cdef int i
    cdef int j
    for i in xrange(arr.shape[0]):
        for j in xrange(arr.shape[1]):
            arr[i,j] = __builtin_popcount(arr[i,j])

def inplace_popcount_32(arr):
    """
    Computes the popcount of each element of a numpy array in-place.
    http://en.wikipedia.org/wiki/Hamming_weight
    Parameters:
    -----------
    arr: numpy_array, dtype=np.uint32
         The array of integers for which the popcounts should be computed.
    """
    if len(arr.shape) == 1:
        _inplace_popcount_32_2d(arr.reshape(-1, 1))
    else:
        _inplace_popcount_32_2d(arr)
