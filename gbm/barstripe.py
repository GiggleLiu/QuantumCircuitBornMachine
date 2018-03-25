import numpy as np

def error_distance(valid_samples, error_samples):
    '''calculate error distance for bar and stripes.'''
    dist_list = []
    for sample in error_samples:
        dist_list.append(np.sum(np.abs(valid_samples-sample), axis=tuple(range(1,valid_samples.ndim))).min())
    return np.array(dist_list)

def bs_configs(geometry):
    '''the total configuration space for bar and stripes.'''
    num_bit = np.prod(geometry)
    samples = np.arange(2**num_bit)
    samples = unpacknbits(samples[:, None], num_bit).reshape((-1,)+geometry)
    return samples

def is_bs(samples):
    '''a sample is a bar or a stripe.'''
    return (np.abs(np.diff(samples,axis=-1)).sum(axis=(1,2))==0)|((np.abs(np.diff(samples, axis=1)).sum(axis=(1,2)))==0)

def barstripe_collection(nrow, ncol):
    '''
    Load a predefined database.

    Return:
        list: list of data as a table of database.
    '''
    s0 = np.zeros([nrow, ncol], dtype='int64')
    database = []
    for i in np.arange(2**nrow):
        s = s0.copy()
        mask = np.asarray(unpacknbits(i, nrow), dtype='bool')
        s[mask] = 1
        database.append(s)

    for j in np.arange(2**ncol):
        if j != 0 and j != 2**ncol - 1:
            s = s0.copy()
            mask = np.asarray(unpacknbits(j, ncol), dtype='bool')
            s[:, mask] = 1
            database.append(s)
    return np.asarray(database, dtype='int8')
