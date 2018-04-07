#!/usr/bin/env python
'''
Learning 2 x 3 bar and stripe using Born Machine.
'''
import numpy as np
import pdb, os, fire
import scipy.sparse as sps

from qcbm import train
from qcbm.testsuit import load_barstripe

np.random.seed(2)
# the testcase used in this program.
depth = 7
geometry = (2,3)
bm = load_barstripe(geometry, depth, structure='chowliu-tree')

class UI():
    def checkgrad(self):
        '''check the correctness of our gradient.'''
        theta_list = np.random.random(bm.circuit.num_param)*2*np.pi
        g1 = bm.gradient(theta_list)
        g2 = bm.gradient_numerical(theta_list)
        error_rate = np.abs(g1-g2).sum()/np.abs(g1).sum()
        print('Error Rate = %.4e'%error_rate)

    def train(self):
        '''train this circuit.'''
        theta_list = np.random.random(bm.circuit.num_param)*2*np.pi
        loss, theta_list = train(bm, theta_list, 'L-BFGS-B', max_iter=200)
        # save
        np.savetxt('data/loss-cl.dat', bm._loss_histo)
        np.savetxt('data/theta-cl.dat', theta_list)

    def vcircuit(self):
        '''visualize circuit of Born Machine.'''
        from qcbm import ProjectQContext
        bm.context = ProjectQContext
        bm.viz()

    def vpdf(self):
        '''visualize probability densitys'''
        import matplotlib.pyplot as plt
        from qcbm.dataset import barstripe_pdf
        pl0 = barstripe_pdf(geometry)
        try:
            theta_list = np.loadtxt('data/theta-cl.dat')
            plt.plot(pl0)
            pl = bm.pdf(theta_list)
        except:
            print('No Born Machine Data')
        plt.plot(pl)
        plt.legend(['Data', 'Born Machine'])
        plt.show()

    def generate(self):
        '''show generated samples for bar and stripes'''
        from qcbm.dataset import binary_basis
        from qcbm.utils import sample_from_prob
        import matplotlib.pyplot as plt
        # generate samples
        size = (7,5)
        theta_list = np.loadtxt('data/theta-cl.dat')
        pl = bm.pdf(theta_list)
        indices = np.random.choice(np.arange(len(pl)), np.prod(size), p=pl)
        samples = binary_basis(geometry)[indices]

        # show
        fig = plt.figure(figsize=(5,4))
        gs = plt.GridSpec(*size)
        for i in range(size[0]):
            for j in range(size[1]):
                plt.subplot(gs[i,j]).imshow(samples[i*size[1]+j], vmin=0, vmax=1)
                plt.axis('equal')
                plt.axis('off')
        plt.show()

    def statgrad(self):
        '''layerwise gradient statistics'''
        import matplotlib.pyplot as plt
        nsample = 10

        # calculate
        grad_stat = [[] for i in range(depth+1)]
        for i in range(nsample):
            print('running %s-th random parameter'%i)
            theta_list = np.random.random(bm.circuit.num_param)*2*np.pi
            loss = bm.mmd_loss(theta_list)
            grad = bm.gradient(theta_list)
            loc = 0
            for i, layer in enumerate(bm.circuit[::2]):
                grad_stat[i] = np.append(grad_stat[i], grad[loc:loc+layer.num_param])
                loc += layer.num_param

        # get mean amplitude, expect first and last layer, they have less parameters.
        var_list = []
        for grads in grad_stat[1:-1]:
            var_list.append(np.abs(grads).mean())

        plt.figure(figsize=(5,4))
        plt.plot(range(1,depth), var_list)
        plt.ylabel('Gradient Std. Err.')
        plt.xlabel('Depth')
        plt.ylim(0,np.max(var_list)*1.2)
        plt.show()

if __name__ == '__main__':
    fire.Fire(UI)
