'''
Several Models used for testing.
'''

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

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

def load_barstripe(geometry, depth, version='scipy', structure='nn'):
    '''3 x 3 bar and stripes.'''
    num_bit = np.prod(geometry)

    # standard circuit
    if structure == 'random-tree':
        X = np.random.random([num_bit, num_bit])
        Tcsr = -minimum_spanning_tree(-X)
        Tcoo = Tcsr.tocoo()
        pairs = list(zip(Tcoo.row, Tcoo.col))
        print('Random tree pairs = %s'%pairs)
    elif isinstance(structure, list):
        pairs = structure
    else:
        pairs = get_nn_pairs(geometry)
    circuit = get_demo_circuit(num_bit, depth, pairs)

    # bar and stripe
    p_bs = barstripe_pdf(geometry)

    # mmd loss
    mmd = RBFMMD2([0.5,1,2,4], num_bit, True)

    # Born Machine
    bm = BornMachine(circuit, mmd, p_bs)
    if version == 'projectq':
        bm.set_context('projectq')
    return bm
