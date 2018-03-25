'''
Several Models used for testing.
'''

import numpy as np

from blocks import get_demo_circuit, get_nn_pairs
from dataset import gaussian_pdf, barstripe_pdf, digit_basis, binary_basis
from gbm import BornMachine
from mmd import RBFMMD2


def load_gaussian(num_bit, depth):
    '''gaussian distribution.'''
    geometry = (num_bit,)
    hndim = 2**num_bit

    # standard circuit
    pairs = get_nn_pairs(geometry)
    circuit = get_demo_circuit(num_bit, depth, pairs)

    # bar and stripe
    p_bs = gaussian_pdf(geometry, mu=hndim/2., sigma=hndim/4.)

    # mmd loss
    mmd = RBFMMD2(sigma_list=[0.25,0.5,1,2,4], basis=digit_basis(geometry))

    # Born Machine
    bm = BornMachine(circuit, mmd, p_bs)
    return bm

def load_barstripe(geometry, depth):
    '''3 x 3 bar and stripes.'''
    num_bit = np.prod(geometry)

    # standard circuit
    pairs = get_nn_pairs(geometry)
    circuit = get_demo_circuit(num_bit, depth, pairs)

    # bar and stripe
    p_bs = barstripe_pdf(geometry)

    # mmd loss
    mmd = RBFMMD2(sigma_list=[0.5,1,2,4], basis=binary_basis((num_bit,)))

    # Born Machine
    bm = BornMachine(circuit, mmd, p_bs)
    return bm
