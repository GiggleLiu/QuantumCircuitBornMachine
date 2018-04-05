#!/usr/bin/python
from __future__ import division
from numpy import *
import pdb

# MPI setting
try:
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
    SIZE = COMM.Get_size()
    RANK = COMM.Get_rank()
    if SIZE != 1:
        try:
            import mkl
            mkl.set_num_threads(1)
        except:
            print('Can not set MP-thread to 1')
            pass
except:
    print('WARNING, NOT USING MULTITHREADING.')
    COMM = None
    SIZE = 1
    RANK = 0

__all__ = ['mpido', 'SIZE', 'COMM', 'RANK']


def mpido(func, inputlist, bcastoutputmesh=True, interweave=True):
    '''
    MPI for list input.

    Parameters:
        :func: function, the function defined on inputlist.
        :inputlist: list/array, the input list.
        :bcastoutputmesh: bool, broadcast output mesh if True.
        :interweave: bool, mpi with interweave job distribution to boost performance.
    '''
    N = len(inputlist)
    datas = []
    if not interweave:
        # non-interweaving mode
        ntask = (N - 1) // SIZE + 1
        for i in range(N):
            if i // ntask == RANK:
                datas.append(func(inputlist[i]))
        if SIZE != 1:
            datal = COMM.gather(datas, root=0)
            if RANK == 0:
                datas = []
                for datai in datal:
                    datas += datai
            # broadcast mesh
            if bcastoutputmesh:
                datas = COMM.bcast(datas, root=0)
    else:
        # interweave mode
        for i in range(N):
            if i % SIZE == RANK:
                datas.append(func(inputlist[i]))
        if SIZE != 1:
            datal = COMM.gather(datas, root=0)
            if RANK == 0:
                datas = []
                for i in range(N):
                    datas.append(datal[i % SIZE][i // SIZE])
            # broadcast mesh
            if bcastoutputmesh:
                datas = COMM.bcast(datas, root=0)
    return datas


def test_mpido():
    x = linspace(0, 1, 100)

    def f(x): return x**2
    y_true = f(x)

    for interweave in [True, False]:
        y = mpido(func=f, inputlist=x, interweave=interweave)
        assert(allclose(y, y_true))


if __name__ == '__main__':
    test_mpido()
