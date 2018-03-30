import numpy as np
import time
import traceback

from contexts import CircuitContext, ScipyContext
from utils import sample_from_prob

SAMPLE_PREFER_SIZE = 2000

class BornMachine(object):
    '''
    Born Machine,

    Args:
        circuit (BlockQueue): the circuit architechture.
        batch_size (int|None): introducing sampling error, None for no sampling error.
    '''
    def __init__(self, circuit, mmd, p_data, batch_size=None):
        self.circuit = circuit
        self.mmd = mmd
        self.p_data = p_data
        self.batch_size = batch_size
        self.context = ScipyContext
        if not self.mmd.use_prob:
            self.databatch = sample_from_prob(np.arange(len(p_data)), p_data, batch_size)

    @property
    def depth(self):
        return (len(self.circuit)-1)//2

    def set_context(self, context):
        if context == 'scipy':
            self.context = ScipyContext
        elif context == 'projectq':
            self.context = CircuitContext
        else:
            raise

    def viz(self, theta_list=None):
        '''visualize this Born Machine'''
        if theta_list is None:
            theta_list = np.random.random(circuit.num_param)*2*np.pi
        with self.context( self.circuit.num_bit, 'draw') as cc:
            self.circuit(cc.qureg, theta_list)

    def pdf(self, theta_list):
        '''get probability distribution function'''
        t0=time.time()
        with self.context( self.circuit.num_bit, 'simulate') as cc:
            self.circuit(cc.qureg, theta_list)
        t1=time.time()
        #print(t1-t0)
        pl = np.abs(cc.wf)**2
        return pl

    def _sample_or_prob(self, theta_list):
        pl = self.pdf(theta_list)
        if self.batch_size is not None:
            # introducing sampling error
            samples = sample_from_prob(np.arange(len(pl)), pl, self.batch_size)
            if self.mmd.use_prob:
                state = prob_from_sample(samples, len(pl), False)
            else:
                state = samples
        else:
            if self.mmd.use_prob:
                state = pl
            else:
                raise
        return state

    def _data(self):
        return self.p_data if self.mmd.use_prob else self.databatch

    def mmd_loss(self, theta_list):
        '''get the loss'''
        self._state = self._sample_or_prob(theta_list)
        return self.mmd(self._state, self._data)

    def gradient(self, theta_list):
        '''
        cheat and get gradient.
        '''
        state = self._state
        grad = []
        for i in range(len(theta_list)):
            # pi/2 phase
            theta_list[i] += np.pi/2.
            state_pos = self._sample_or_prob(theta_list)
            # -pi/2 phase
            theta_list[i] -= np.pi
            state_neg = self._sample_or_prob(theta_list)
            # recover
            theta_list[i] += np.pi/2.

            grad_pos = self.mmd.kernel_expect(state, state_pos) - self.mmd.kernel_expect(state, state_neg)
            grad_neg = self.mmd.kernel_expect(self.p_data, state_pos) - self.mmd.kernel_expect(self.p_data, state_neg)
            grad.append(grad_pos - grad_neg)
        return np.array(grad)

    def gradient_numerical(self, theta_list, delta=1e-2):
        '''
        numerical differenciation.
        '''
        grad = []
        for i in range(len(theta_list)):
            theta_list[i] += delta/2.
            loss_pos = self.mmd_loss(theta_list)
            theta_list[i] -= delta
            loss_neg = self.mmd_loss(theta_list)
            theta_list[i] += delta/2.

            grad_i = (loss_pos - loss_neg)/delta
            grad.append(grad_i)
        return np.array(grad)
