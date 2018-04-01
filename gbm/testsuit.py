'''
Several Models used for testing.
'''

import numpy as np

from blocks import get_demo_circuit, get_nn_pairs
from dataset import gaussian_pdf, barstripe_pdf, digit_basis, binary_basis
from gbm import BornMachine
from mmd import RBFMMD2, RBFMMD2MEM


def load_gaussian(num_bit, depth, version='scipy'):
    '''gaussian distribution.'''
    geometry = (num_bit,)
    hndim = 2**num_bit

    # standard circuit
    pairs = get_nn_pairs(geometry)
    circuit = get_demo_circuit(num_bit, depth, pairs)

    # bar and stripe
    p_bs = gaussian_pdf(geometry, mu=hndim/2., sigma=hndim/4.)

    # mmd loss
    mmd = RBFMMD2([0.25,0.5,1,2,4], num_bit, False)

    # Born Machine
    bm = BornMachine(circuit, mmd, p_bs)
    if version == 'projectq':
        bm.set_context('projectq')
    return bm

def load_barstripe(geometry, depth, batch_size=None, version='scipy'):
    '''3 x 3 bar and stripes.'''
    num_bit = np.prod(geometry)

    # standard circuit
    pairs = get_nn_pairs(geometry)
    circuit = get_demo_circuit(num_bit, depth, pairs)

    # bar and stripe
    p_bs = barstripe_pdf(geometry)

    # mmd loss
    mmd = RBFMMD2([0.25,0.5,1,2,4], num_bit, True)

    # Born Machine
    bm = BornMachine(circuit, mmd, p_bs, batch_size=batch_size)
    if version == 'projectq':
        bm.set_context('projectq')
    return bm


def load_complex(geometry, depth, batch_size=None, version='scipy'):
    '''3 x 3 bar and stripes.'''
    num_bit = np.prod(geometry)

    # standard circuit
    pairs = get_nn_pairs(geometry)
    circuit = get_demo_circuit(num_bit, depth, pairs)

    # complex wave function
    wf = coil_wf(num_bit, winding=2)

    # mmd loss
    mmd = RBFMMD2([0.25,1,4], num_bit, False)

    # Born Machine
    bm = CloneBornMachine(circuit, mmd, wf, batch_size=batch_size)
    if version == 'projectq':
        bm.set_context('projectq')
    return bm
