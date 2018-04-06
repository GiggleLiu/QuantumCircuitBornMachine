#!/usr/bin/env python
'''
Learning 3 x 3 bar and stripe using Born Machine.
'''
import numpy as np
import pdb, os, fire
import scipy.sparse as sps

from train import train
from testsuit import load_barstripe

np.random.seed(2)
# the testcase used in this program.
depth = 10
geometry = (3,3)
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
        np.savetxt('loss-cl.dat', bm._loss_histo)
        np.savetxt('theta-cl.dat', theta_list)

    def vcircuit(self):
        '''visualize circuit of Born Machine.'''
        from contexts import ProjectQContext
        bm.context = ProjectQContext
        bm.viz()

    def vpdf(self):
        '''visualize probability densitys'''
        import matplotlib.pyplot as plt
        from dataset import barstripe_pdf
        pl0 = barstripe_pdf(geometry)
        try:
            theta_list = np.loadtxt('theta-cl.dat')
            plt.plot(pl0)
            pl = bm.pdf(theta_list)
        except:
            print('No Born Machine Data')
        plt.plot(pl)
        plt.legend(['Data', 'Born Machine'])
        plt.show()

    def generate(self):
        # show generated samples
        from dataset import binary_basis
        from utils import sample_from_prob
        import matplotlib.pyplot as plt
        size = (7,7)
        theta_list = np.loadtxt('theta-cl.dat')
        pl = bm.pdf(theta_list)
        indices = np.random.choice(np.arange(len(pl)), np.prod(size), p=pl)
        samples = binary_basis((3,3))[indices]

        plt.ion()
        fig = plt.figure(figsize=(5,4))
        width = 0.08
        height = 0.12
        spacex = 0.02
        spacey = 0.02
        row_space = 0.025
        for i in range(size[0]):
            for j in range(size[1]):
                left = 0.17+(width+spacey)*j
                bottom = 0.85-(height+spacex)*i
                ax2 = fig.add_axes([left, bottom, width, height])

                ax2.imshow(samples[i*size[1]+j], vmin=0, vmax=1)
                ax2.axis('off')
        pdb.set_trace()

    def statgrad(self):
        '''layerwise gradient statistics'''
        import matplotlib.pyplot as plt
        nsample = 10

        bm = load_barstripe((3, 3), depth)
        num_bit = bm.circuit.num_bit

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
        plt.plot(range(1,depth), var_list)
        plt.ylabel('Gradient Std. Err.')
        plt.xlabel('Depth')
        plt.ylim(0,1e-3)
        plt.show()

if __name__ == '__main__':
    fire.Fire(UI)
