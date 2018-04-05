#!/usr/bin/env python
import numpy as np
import pdb, os, fire
import scipy.sparse as sps

from train import train
from testsuit import load_gaussian, load_barstripe

class UI():
    def rt(self, seed):
        '''random tree test.'''
        np.random.seed(seed)
        depth = 10

        bm = load_barstripe((3, 3), depth, structure='random-tree')
        np.random.seed(2)
        theta_list = np.random.random(bm.circuit.num_param)*2*np.pi
        loss, theta_list = train(bm, theta_list, 'L-BFGS-B', max_iter=1000)
        # save
        np.savetxt('loss-rt-%d.dat'%seed, bm._loss_histo)

    def dumptree(self):
        depth = 10
        data = []
        for seed in range(1,11):
            np.random.seed(seed)
            bm = load_barstripe((3, 3), depth, structure='random-tree')
            data.append(np.array(bm.circuit[1].pairs).ravel())
        np.savetxt('tree.dat', data, fmt='%d')

    def chowliu(self):
        depth = 10
        from graph.chowliu import chowliu_tree
        #bm = load_barstripe((3, 3), depth, structure='')
        #chowliu = chowliu_tree(dataset=sample_prob())

        chowliu = [(1, 7), (2, 8), (4, 7), (5, 3), (6, 0), (6, 3), (6, 7), (8, 5)]
        bm = load_barstripe((3, 3), depth, structure=chowliu)
        np.random.seed(2)
        theta_list = np.random.random(bm.circuit.num_param)*2*np.pi
        loss, theta_list = train(bm, theta_list, 'L-BFGS-B', max_iter=1000)
        # save
        np.savetxt('loss-cl.dat', bm._loss_histo)

if __name__ == '__main__':
    fire.Fire(UI)
