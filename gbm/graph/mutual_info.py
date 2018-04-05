import numpy as np
import pdb

def mutual_table(samples):
    '''
    calculate :math:`\sum\limits_{x,y}p(x,y)\log\\frac{p(x,y)}{p(x)p(y)}`
    from ampirical distribution.
    '''
    sl = np.unique(samples)
    d = len(sl)
    num_sample = len(samples)
    num_spin = samples.shape[1]
    pstij = np.zeros([num_spin, num_spin, d, d])
    psi = np.zeros([num_spin, d])

    for s_i in sl:
        mask_i = samples == s_i
        psi[:,s_i] = mask_i.sum(axis=0)
        for s_j in sl:
            mask_j = samples == s_j
            pstij[:,:,s_i,s_j] = (mask_i[:,None,:]&mask_j[:,:,None]).sum(axis=0)
    psi = psi/num_sample
    pstij = pstij/num_sample

    # mutual information
    pratio = pstij/np.maximum(psi[:,None,:,None]*psi[None,:,None,:], 1e-15)
    for i in range(num_spin):
        pratio[i,i] = 1
    Ist = (pstij*np.log(pratio)).sum(axis=(2,3))
    return Ist
