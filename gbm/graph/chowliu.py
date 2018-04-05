import pdb
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

from .mutual_info import mutual_table

def chowliu_tree(samples):
    '''
    generate chow-liu tree.
    '''
    X = mutual_table(samples)
    Tcsr = -minimum_spanning_tree(-X)
    Tcoo = Tcsr.tocoo()
    pairs = list(zip(Tcoo.row, Tcoo.col))
    print('Chow-Liu tree pairs = %s'%pairs)
    return pairs

def random_tree(num_bit):
    '''
    generate chow-liu tree.
    '''
    X = np.random.random([num_bit, num_bit])
    Tcsr = -minimum_spanning_tree(-X)
    Tcoo = Tcsr.tocoo()
    pairs = list(zip(Tcoo.row, Tcoo.col))
    print('Random tree pairs = %s'%pairs)
    return pairs

def visualize_tree(pairs, geometry):
    import matplotlib.pyplot as plt

    xs, ys = np.meshgrid(np.arange(geometry[0]), np.arange(geometry[1]), indexing='ij')
    locs = np.concatenate([xs[...,None], ys[...,None]], axis=-1).reshape([-1,2])
    plt.scatter(locs[:,0], locs[:,1], s=80, zorder=101)
    for i, loc in enumerate(locs):
        plt.text(loc[0], loc[1]-0.2, '%d'%i, fontsize=18, va='center', ha='center')
    for i, j in pairs:
        start, end = locs[i], locs[j]
        plt.plot([start[0], end[0]], [start[1], end[1]],color='k')
