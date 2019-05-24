#!/usr/bin/env python
'''
Learning 2 x 3 bar and stripe using Born Machine.
'''
import numpy as np
import pdb, os, fire
import time

from qcbm import train
from qcbm.testsuit import load_barstripe

import mkl
mkl.set_num_threads(1)

np.random.seed(2)
try:
    os.mkdir('data')
except:
    pass

# the testcase used in this program.
depth = 10
geometry = (3,3)
bm = load_barstripe(geometry, depth, structure='ring', context='projectq')

theta_list = np.random.rand(bm.circuit.num_param)*2*np.pi

nrepeat = 100
t0 = time.time()
for i in range(nrepeat):
    with bm.context( bm.circuit.num_bit, 'simulate') as cc:
        bm.circuit(cc.qureg, theta_list)
t1 = time.time()
print("Time/Loop = %s seconds"%((t1-t0)/nrepeat))

pdb.set_trace()
