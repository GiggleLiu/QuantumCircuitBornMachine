from projectq.ops import *
from functools import reduce

import qclibd, qclibs
import pdb

class CircuitBlock(object):
    def __init__(self, num_bit):
        self.num_bit = num_bit

    def __call__(self, qureg, theta_list):
        '''
        build a quantum circuit.

        Args:
            theta_list (1darray<float>, len=3*num*bit*(depth+1)): parameters in this quantum circuit, here, depth equals to the number of entanglement operations.

        Return:
            remaining theta_list
        '''
        pass

class BlockQueue(list):
    @property
    def num_bit(self):
        return self[0].num_bit

    @property
    def num_param(self):
        return sum([b.num_param for b in self])

    def __call__(self, qureg, theta_list):
        for block in self:
            theta_list = block(qureg, theta_list)
        np.testing.assert_(len(theta_list)==0)

    def __str__(self):
        return '\n'.join([str(b) for b in self])


class CleverBlockQueue(BlockQueue):
    '''
    Clever Block Queue that keep track of theta_list changing history, for fast update.
    '''
    def __init__(self, *args):
        list.__init__(self, *args)
        self.theta_last = None
        self.memo = None

    def __call__(self, qureg, theta_list):
        if not isinstance(qureg, np.ndarray):
            return super(CleverBlockQueue, self).__call__(qureg, theta_list)
        mats = []
        theta_last = self.theta_last
        self.theta_last = theta_list.copy()
        qureg_ = qureg
        for iblock, block in enumerate(self):
            # generate or use a block matrix
            num_param = block.num_param
            if self.memo is not None and np.abs(theta_list[:num_param]-theta_last[:num_param]).sum()<1e-12:#np.allclose(theta_list[:num_param], theta_last[:num_param]):
                mat = self.memo[iblock]
                theta_list = theta_list[num_param:]
                theta_last = theta_last[num_param:]
            else:
                #mat, theta_list = block.tomat(theta_list)
                mat, theta_list = block.tocsr(theta_list)
                if theta_last is not None:
                    theta_last = theta_last[num_param:]
            #qureg_ = mat.dot(qureg_)
            for mat_i in mat:
                qureg_ = mat_i.dot(qureg_)
            mats.append(mat)
        np.testing.assert_(len(theta_list)==0)
        self.memo = mats
        qureg[...] = qureg_


class ArbituaryRotation(CircuitBlock):
    def __init__(self, num_bit):
        super(ArbituaryRotation, self).__init__(num_bit)
        self.mask = np.array([True] * (3*num_bit), dtype='bool')

    def __call__(self, qureg, theta_list):
        nvar = sum(self.mask)
        gates = [Rz, Rx, Rz]
        for i, (theta, mask) in enumerate(zip(theta_list[:nvar], self.mask)):
            ibit, igate = i//3, i%3
            if mask:
                gate = gates[igate](theta)
                gate | qureg[ibit]
        return theta_list[nvar:]

    def __str__(self):
        return 'Rotate[%d]'%(self.num_param)

    @property
    def num_param(self):
        return sum(self.mask)

    def tomat(self, theta_list):
        '''transform this block to csr_matrix.'''
        nvar = sum(self.mask)
        theta_list_ = np.zeros(3*self.num_bit)
        theta_list_[self.mask] = theta_list[:nvar]
        rots = [qclibd.rot(*ths) for ths in theta_list_.reshape([self.num_bit,3])]
        res = qclibd._(rots, np.arange(self.num_bit), self.num_bit)
        return res, theta_list[nvar:]

    def tocsr(self, theta_list):
        '''transform this block to csr_matrix.'''
        nvar = sum(self.mask)
        theta_list_ = np.zeros(3*self.num_bit)
        theta_list_[self.mask] = theta_list[:nvar]
        rots = [qclibs.rot(*ths) for ths in theta_list_.reshape([self.num_bit,3])]
        res = [qclibs._([rot], [i], self.num_bit) for i,rot in enumerate(rots)]
        return res, theta_list[nvar:]


class Entangler(CircuitBlock):
    def __init__(self, num_bit, pairs, gate, num_param_per_pair):
        super(Entangler, self).__init__(num_bit)
        self.pairs = pairs
        self.gate = gate
        self.num_param_per_pair = num_param_per_pair
        self.mask = np.array([True]*(num_bit*num_param_per_pair), dtype='bool')

    def __str__(self):
        pair_str = ','.join(['%d-%d'%(i,j) for i,j in self.pairs])
        return '%s(%s)'%(self.gate, pair_str)

    def __call__(self, qureg, theta_list):
        for pair in self.pairs:
            if self.num_param_per_pair == 0:
                gate = self.gate
            else:
                theta_i, theta_list = np.split(theta_list, self.num_param_per_pair)
                gate = self.gate(*theta_i)
            gate | (qureg[pair[0]], qureg[pair[1]])
        return theta_list

    @property
    def num_param(self):
        return sum(self.mask)

    def tomat(self, theta_list):
        '''transform this block to csr_matrix.'''
        i, j = self.pairs[0]
        res = qclibs.CNOT(i, j, self.num_bit)
        for i, j in self.pairs[1:]:
            res = qclibs.CNOT(i,j,self.num_bit).dot(res)
        return res, theta_list

    def tocsr(self, theta_list):
        '''transform this block to csr_matrix.'''
        i, j = self.pairs[0]
        res = qclibs.CNOT(i, j, self.num_bit)
        for i, j in self.pairs[1:]:
            res = qclibs.CNOT(i,j,self.num_bit).dot(res)
        res.eliminate_zeros()
        return [res], theta_list


def const_entangler(num_bit, pairs, gate):
    return Entangler(num_bit, pairs, gate, 0)

def cnot_entangler(num_bit, pairs):
    '''controled-not entangler.'''
    return Entangler(num_bit, pairs, CNOT, 0)

def cz_entangler(num_bit, pairs):
    '''controled-Z entangler.'''
    return Entangler(num_bit, pairs, C(Z), 1)

class BondTimeEvolution(CircuitBlock):
    def __init__(self, num_bit, pairs, hamiltonian):
        super(BondTimeEvolution, self).__init__(num_bit)
        self.pairs = pairs
        self.hamiltonian = hamiltonian
        self.mask = np.array([True]*len(pairs),dtype='bool')

    def __call__(self, qureg, theta_list):
        npar = len(self.pairs)
        for pair, ti, mask_bi in zip(self.pairs, theta_list[:npar], self.mask):
            if mask_bi:
                hamiltonian = self.hamiltonian.replace('i', str(pair[0])).replace('j', str(pair[1]))
                gate = TimeEvolution(ti, QubitOperator(hamiltonian))
                gate | qureg
        return theta_list[npar:]

    def __str__(self):
        pair_str = ','.join(['%d-%d'%(i,j) for i,j in self.pairs])
        return '%s[%s](t)'%(self.hamiltonian, pair_str)

    @property
    def num_param(self):
        return sum(self.mask)

def get_nn_pairs(geometry):
    '''define pairs that cnot gates will apply.'''
    num_bit = np.prod(geometry)
    if len(geometry) == 2:
        nrow, ncol = geometry
        res = []
        for ij in range(num_bit):
            i, j = ij // ncol, ij % ncol
            res.extend([(ij, i_ * ncol + j_)
                        for i_, j_ in [((i + 1) % nrow, j), (i, (j + 1) % ncol)]])
        return res
    elif len(geometry) == 1:
        res = []
        for inth in range(2):
            for i in range(inth, num_bit, 2):
                res = res + [(i, i_ % num_bit) for i_ in range(i + 1, i + 2)]
        return res
    else:
       raise NotImplementedError('')

def get_demo_circuit(num_bit, depth, pairs):
    blocks = []
    # build circuit
    for idepth in range(depth+1):
        blocks.append(ArbituaryRotation(num_bit))
        if idepth!=depth:
            blocks.append(cnot_entangler(num_bit, pairs))

    # set leading and trailing Rz to disabled
    blocks[0].mask[::3] = False
    blocks[-1].mask[2::3] = False
    return CleverBlockQueue(blocks)
