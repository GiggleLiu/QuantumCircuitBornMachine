#!/usr/bin/env python
'''
Learning 3 x 3 bar and stripe using Born Machine.
'''
import numpy as np
import pdb, os, fire
import scipy.sparse as sps

from train import train
from testsuit import load_barstripe

from profilehooks import profile

np.random.seed(2)
# the testcase used in this program.
depth = 10
bm = load_barstripe((3, 3), depth, structure='chowliu-tree')

class UI():
    @profile
    def train(self):
        '''train this circuit.'''
        bm = load_barstripe((3, 3), depth, structure='chowliu')
        theta_list = np.random.random(bm.circuit.num_param)*2*np.pi
        loss, theta_list = train(bm, theta_list, 'L-BFGS-B', max_iter=200)
        # save
        np.savetxt('loss-cl.dat', bm._loss_histo)
        np.savetxt('theta-cl.dat', theta_list)

    def vcircuit(self):
        '''visualize circuit of Born Machine.'''
        from contexts import ProjectQContext
        bm.context = ProjectQContext
        bm.viz()


if __name__ == '__main__':
    fire.Fire(UI)
