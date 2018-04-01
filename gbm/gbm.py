import numpy as np
import time, pdb
import traceback

from contexts import CircuitContext, ScipyContext
from utils import sample_from_prob, prob_from_sample

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

    def wf(self, theta_list):
        '''get wf function'''
        with self.context( self.circuit.num_bit, 'simulate') as cc:
            self.circuit(cc.qureg, theta_list)
        return cc.wf

    def pdf(self, theta_list):
        '''get probability distribution function'''
        return pdf_(self.wf(theta_list))

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

    @property
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

class CloneBornMachine(BornMachine):
    def __init__(self, circuit, mmd, wf_data, batch_size=None):
        np.testing.assert_almost_equal((np.abs(wf_data)**2).sum(), 1)
        super(CloneBornMachine, self).__init__(circuit, mmd, np.abs(self.wf_data)**2, batch_size=None)

        self.wf_data = wf_data
        self.p_xdata = xbasis(self.p_data)

        if not self.mmd.use_prob:
            hndim = np.arange(len(self.p_data))
            self.xdatabatch = sample_from_prob(hndim, np.abs(self.p_xdata)**2, batch_size)

    @property
    def _xdata(self):
        return self.p_xdata if self.mmd.use_prob else self.xdatabatch

    def mmd_loss(self, theta_list):
        '''get the loss'''
        self._state = self._sample_or_prob(theta_list)
        return self.mmd(self._state[0], self._data) + self.mmd(self._state[1], self._xdata)

    def gradient(self, theta_list):
        '''
        cheat and get gradient.
        '''
        state2 = self._state
        grad = []
        for i in range(len(theta_list)):
            # pi/2 phase
            theta_list[i] += np.pi/2.
            state_pos2 = self._sample_or_prob(theta_list)
            # -pi/2 phase
            theta_list[i] -= np.pi
            state_neg2 = self._sample_or_prob(theta_list)
            # recover
            theta_list[i] += np.pi/2.

            gi = 0
            for state, state_neg, state_pos in zip(state2, state_neg2, state_pos2):
                grad_pos = self.mmd.kernel_expect(state, state_pos) - self.mmd.kernel_expect(state, state_neg)
                grad_neg = self.mmd.kernel_expect(self.p_data, state_pos) - self.mmd.kernel_expect(self.p_data, state_neg)
                gi += grad_pos - grad_neg
            grad.append(gi)
        return np.array(grad)

    def _sample_or_prob(self, theta_list):
        wf = self.wf(theta_list)
        state = []
        for pl in [pdf_(wf), xpdf_(wf)]:
            if self.batch_size is not None:
                # introducing sampling error
                samples = sample_from_prob(np.arange(len(pl)), pl, self.batch_size)
                if self.mmd.use_prob:
                    state.append(prob_from_sample(samples, len(pl), False))
                else:
                    state.append(samples)
            else:
                if self.mmd.use_prob:
                    state.append(pl)
                else:
                    raise
        return state
 
def xpdf_(self, wf):
    return np.abs(xbasis(wf))**2

def pdf_(self, wf)
    return np.abs(wf)**2

