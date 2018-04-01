import numpy as np
from clib.popcount import inplace_popcount_32 as popcount
from flib.fysics import fkernel_expect_bin as fkernel
from flib.fysics import fmix_rbf_kernel_bin, fmix_rbf_kernel_dict
import pdb, time

class RBFMMD2(object):
    def __init__(self, sigma_list, num_bit, is_binary):
        self.sigma_list = sigma_list
        self.num_bit = num_bit
        self.is_binary = is_binary
        self.basis = np.arange(2**num_bit,dtype='uint32')
        self.K = mix_rbf_kernel(self.basis, self.basis, self.sigma_list, is_binary)
        self.use_prob = True

    def __call__(self, px, py):
        '''
        Args:
            px (1darray, default=None): probability for data set x, used only when self.is_exact==True.
            py (1darray, default=None): same as px, but for data set y.

        Returns:
            float, loss.
        '''
        pxy = px-py
        return self.kernel_expect(pxy, pxy)

    def kernel_expect(self, px, py):
        res = px.dot(self.K.dot(py))
        return res

class RBFMMD2MEM(object):
    def __init__(self, sigma_list, num_bit, is_binary, use_prob=False):
        self.sigma_list = sigma_list
        self.is_binary = is_binary
        self.num_bit = num_bit
        self.basis = np.arange(2**num_bit,dtype='int32')
        self.use_prob = use_prob
        if not is_binary:
            vmax = 2**num_bit
            dx = np.arange(vmax)**2
        else:
            # binary
            dx = np.arange(num_bit+1)
        self.KD = _mix_rbf_kernel_d(dx, self.sigma_list)

    def __call__(self, px, py):
        '''
        Args:
            px (1darray, default=None): probability for data set x, used only when self.is_exact==True.
            py (1darray, default=None): same as px, but for data set y.

        Returns:
            float, loss.
        '''
        pxy = px-py
        return self.kernel_expect(pxy, pxy)

    def kernel_expect(self, px, py):
        res = fkernel(self.basis, px, py, self.KD)
        print(res)
        return res

    def kernel_sample(self, x, y):
        res = fmix_rbf_kernel_dict(x,y,self.KD)
        return res

    def _kernel_expect(self, px, py):
        t0=time.time()
        if self.is_binary:
            absd = self.basis[:,None]^self.basis
            popcount(absd)
        else:
            absd = abs(self.basis-self.basis[i])
        k=px.dot(self.KD[absd].dot(py))
        t1=time.time()
        print(k,t1-t0)
        pdb.set_trace()
        return k

def mix_rbf_kernel(x, y, sigma_list, is_binary):
    if is_binary:
        dx2 = x[:,None]^y
        popcount(dx2)
    else:
        dx2 = (x[:, None] - y)**2
    return _mix_rbf_kernel_d(dx2, sigma_list)

def _mix_rbf_kernel_d(dx2, sigma_list):
    K = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma)
        K = K + np.exp(-gamma * dx2)
    return K
