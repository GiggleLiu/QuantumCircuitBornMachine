'''
Tests for MMD
'''
import numpy as np
from numpy.testing import dec, assert_, assert_raises,\
    assert_almost_equal, assert_allclose
import matplotlib.pyplot as plt
import pdb
import time

from .chowliu import chowliu_tree, visualize_tree
from .mutual_info import mutual_table
from ..dataset import generate_dataset
from ..utils import unpacknbits, packnbits

def test_barstripe():
    np.random.seed(5)
    geometry = (4,4)
    data = generate_dataset(('bar-stripe', {}), geometry=geometry, num_sample=2000, packbits = False)
    tree = chowliu_tree(data)
    plt.ion()
    visualize_tree(tree, geometry=geometry)
    pdb.set_trace()

def test_ising():
    #np.random.seed(5)
    geometry = (4,2)
    data = generate_dataset(('ising', {'ising_K':1.0}), geometry=geometry, num_sample=5000, packbits = False)
    tree = chowliu_tree(data)
    plt.ion()
    plt.figure(figsize=(5,3))
    plt.axis('equal')
    plt.axis('off')
    visualize_tree(tree, geometry=geometry)
    pdb.set_trace()

def analyse_mutualinfo():
    geometry = (4,2)
    data = generate_dataset(('ising', {'ising_K':1.0}), geometry=geometry, num_sample=10000, packbits = False)
    X = mutual_table(data)
    print('mutual information table for (4,2) ising model (28 independant variables) = \n%s'%np.round(X,decimals=4))

    pairs = [(0,1),(2,3),(4,5),(6,7)]
    print('(0,1) bond:')
    for i,j in pairs:
        print(i,j,'->',X[i,j])


    pairs = [(0,2),(2,4),(4,6),(0,6),(1,3),(3,5),(5,7),(1,7)]
    print('(1,0) bond:')
    for i,j in pairs:
        print(i,j,'->',X[i,j])

    pairs = [(0,4),(2,6),(1,5),(3,7)]
    print('(2,0) bond:')
    for i,j in pairs:
        print(i,j,'->',X[i,j])

    pairs = [(0,3),(1,2),(2,5),(3,4),(4,7),(5,6),(1,6),(0,7)]
    print('(1,1) bond:')
    for i,j in pairs:
        print(i,j,'->',X[i,j])

    pairs = [(0,5),(2,7),(1,4),(3,6)]
    print('(2,1) bond:')
    for i,j in pairs:
        print(i,j,'->',X[i,j])

if __name__ == '__main__':
    #analyse_mutualinfo()
    #test_barstripe()
    test_ising()
