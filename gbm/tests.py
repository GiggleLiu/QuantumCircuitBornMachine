import numpy as np
from numpy.testing import dec, assert_, assert_raises,\
    assert_almost_equal, assert_allclose
import matplotlib.pyplot as plt
import pdb
from profilehooks import profile
import scipy.sparse as sps

from blocks import get_demo_circuit, get_nn_pairs
from dataset import gaussian_pdf, barstripe_pdf, digit_basis, binary_basis
from gbm import BornMachine
from contexts import CircuitContext
from mmd import RBFMMD2
from train import train
from testsuit import load_gaussian, load_barstripe
from qclibs import rot, CNOT, ry, I2
import qclibd

def test_dataset():
    geometry = (3,3)
    pl1 = gaussian_pdf(geometry, mu=0, sigma=255.5)
    pl2 = barstripe_pdf(geometry)
    plt.plot(pl1)
    plt.plot(pl2)
    plt.ylim(0,0.01)
    plt.show()

def test_vcircuit():
    depth = 2
    geometry = (6,)

    num_bit = np.prod(geometry)
    pairs = get_nn_pairs(geometry)
    circuit = get_demo_circuit(num_bit, depth, pairs)
    theta_list = np.zeros(circuit.num_param)

    with CircuitContext('draw', num_bit) as cc:
        circuit(cc.qureg, theta_list)

def test_bm():
    depth = 2
    np.random.seed(2)

    #bm = load_gaussian(6, depth)
    bm = load_barstripe((3,3), depth)
    theta_list = np.random.random(bm.circuit.num_param)*2*np.pi

    assert_(bm.depth == depth)
    print(bm.mmd_loss(theta_list))
    g1 = bm.gradient(theta_list)
    g2 = bm.gradient_numerical(theta_list)
    assert_allclose(g1, g2, atol=1e-5)

def test_wf():
    depth = 2
    geometry = (6,)

    num_bit = np.prod(geometry)
    pairs = get_nn_pairs(geometry)
    circuit = get_demo_circuit(num_bit, depth, pairs)

    with CircuitContext('simulate', np.prod(geometry)) as cc:
        circuit(cc.qureg, theta_list)

    wf = np.zeros(2**num_bit)
    wf[0] = 1
    assert_allclose(cc.wf, wf)

@profile
def test_train_gaussian():
    depth = 6
    np.random.seed(2)

    bm = load_gaussian(6, depth)
    theta_list = np.random.random(bm.circuit.num_param)*2*np.pi
    loss, theta_list = train(bm, theta_list, 'L-BFGS-B', max_iter=20)
    pl = bm.pdf(theta_list)

    # display
    plt.ion()
    plt.plot(bm.p_data)
    plt.plot(pl)
    plt.legend(['Data', 'Gradient Born Machine'])
    pdb.set_trace()

def test_train_gaussian_scipy():
    depth = 6
    np.random.seed(2)

    bm = load_gaussian(6, depth)
    theta_list = np.random.random(bm.circuit.num_param)*2*np.pi
    loss, theta_list = train(bm, theta_list, 'L-BFGS-B', max_iter=20)
    pl = bm.pdf(theta_list)

    # display
    plt.ion()
    plt.plot(bm.p_data)
    plt.plot(pl)
    plt.legend(['Data', 'Gradient Born Machine'])
    pdb.set_trace()

def test_train_bs22():
    np.random.seed(2)
    depth = 4

    bm = load_barstripe((2, 2), depth)
    theta_list = np.random.random(bm.circuit.num_param)*2*np.pi
    loss, theta_list = train(bm, theta_list, 'L-BFGS-B', max_iter=20)
    pl = bm.pdf(theta_list)

    # display
    plt.ion()
    plt.plot(bm.p_data)
    plt.plot(pl)
    plt.legend(['Data', 'Gradient Born Machine'])
    pdb.set_trace()

def test_qclib():
    cnot = CNOT(1,0,2)
    assert_(cnot.nnz==4)
    assert_allclose(cnot.toarray(), sps.coo_matrix(([1,1,1,1],([0,1,2,3],[0,1,3,2]))).toarray())
    assert_allclose(rot(-np.pi/2.,np.pi/4.,np.pi/2.).toarray(),ry(np.pi/4.).toarray())

def test_qclibd():
    cnot = qclibd.CNOT(1,0,2)
    assert_allclose(cnot, sps.coo_matrix(([1,1,1,1],([0,1,2,3],[0,1,3,2]))).toarray())
    assert_allclose(qclibd.rot(-np.pi/2.,np.pi/4.,np.pi/2.),qclibd.ry(np.pi/4.))


if __name__ == '__main__':
    #test_dataset()
    #test_vcircuit()
    #test_wf()
    #test_bm()
    #test_qclibd()
    #test_qclib()
    test_train_gaussian()
    #test_train_bs22()
