import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

def run_sample_sparse():
    X = csr_matrix([[0, 8, 0, 3],
                     [0, 0, 2, 5],
                     [0, 0, 0, 6],
                     [0, 0, 0, 0]])
    Tcsr = -minimum_spanning_tree(-X)
    print(Tcsr)
    Tcsr.toarray().astype(int)

def run_sample_dense():
    X = np.array([[0.1, 8, 0.2, 3],
                    [0, 0.01, 2, 5],
                    [0, 0, 0.1, 6],
                    [0, 0, 0, 0.7]])
    Tcsr = -minimum_spanning_tree(-X)
    print(Tcsr)
    Tcsr.toarray().astype(int)

if __name__ == '__main__':
    run_sample_sparse()
    run_sample_dense()
