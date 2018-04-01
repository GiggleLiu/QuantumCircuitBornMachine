from __future__ import division
import pdb
import numpy as np

from utils import packnbits, unpacknbits, sample_from_prob
from barstripe import is_bs

def digit_basis(geometry):
    num_bit = np.prod(geometry)
    M = 2**num_bit
    x = np.arange(M)
    return x

def binary_basis(geometry):
    num_bit = np.prod(geometry)
    M = 2**num_bit
    x = np.arange(M)
    return unpacknbits(x[:,None], num_bit).reshape((-1,)+geometry)

def gaussian_pdf(geometry, mu, sigma):
    '''get gaussian distribution function'''
    x = digit_basis(geometry)
    pl = 1. / np.sqrt(2 * np.pi * sigma**2) * \
        np.exp(-(x - mu)**2 / (2. * sigma**2))
    return pl/pl.sum()

def barstripe_pdf(geometry):
    '''get bar and stripes PDF'''
    x = binary_basis(geometry)
    pl = is_bs(x)
    return pl/pl.sum()

def coil_wf(num_bit, winding=2):
    x = digit_basis((num_bit,))
    amp = np.sin(2*np.pi/len(x)*x)
    phase = np.exp(2j*winding*np.pi*x/len(x))
    return amp*phase

def test():
    wf = coil_wf(6)
    import matplotlib.pyplot as plt
    plt.ion()
    plt.plot(wf.real)
    plt.plot(wf.imag)
    plt.legend(['Real', 'Imag'])
    plt.show()
    pdb.set_trace()

if __name__ == '__main__':
    test()
