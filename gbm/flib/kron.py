'''
Utilities for hexpand:

    * kronecheck product for sparse matrix.
'''

from scipy import sparse as sps
from numpy import asarray,diff
import pdb,time

from .fysics import fkron_coo,fkron_csr_takerow,fkron_csr

__all__=['kron_coo','kron_csr']

def kron_coo(A,B):
    '''
    sparse kronecker product, the version eliminate zeros.

    Parameters:
        :A,B: matrix, the two sparse matrices.

    Return:
        coo_matrix, the kronecker product of A and B, without zeros.
    '''
    A=A.asformat('coo')
    B=B.asformat('coo')
    if len(A.data)==0 or len(B.data)==0:
        return sps.coo_matrix((A.shape[0]*B.shape[0],A.shape[1]*B.shape[1]))
    rown,coln,datn=fkron_coo(col1=A.col,row1=A.row,dat1=A.data,col2=B.col,row2=B.row,dat2=B.data,ncol2=B.shape[1],nrow2=B.shape[0])
    mat=sps.coo_matrix((datn,(rown,coln)),shape=(A.shape[0]*B.shape[0],A.shape[1]*B.shape[1]))
    return mat

def kron_csr(A,B,takerows=None):
    '''
    sparse kronecker product, the csr version.

    Parameters:
        :A,B: matrix, the two sparse matrices.
        :takerows: 1darray, the row desired.

    Return:
        csr_matrix, the kronecker product of A and B.
    '''
    A=A.asformat('csr')
    B=B.asformat('csr')
    rowdim=len(takerows) if takerows is not None else A.shape[0]*B.shape[0]
    if len(A.data)==0 or len(B.data)==0: return sps.csr_matrix((rowdim,A.shape[1]*B.shape[1]))
    if takerows is None:
        indptr,indices,data=fkron_csr(indptr1=A.indptr,indices1=A.indices,dat1=A.data,indptr2=B.indptr,indices2=B.indices,dat2=B.data,ncol2=B.shape[1])
    else:
        #calculate non-zero elements desired
        nrow2=B.shape[0]
        i1s=asarray(takerows)/nrow2
        i2s=takerows-nrow2*i1s
        nnz=sum(diff(A.indptr)[i1s]*diff(B.indptr)[i2s])
        if nnz==0: return sps.csr_matrix((rowdim,A.shape[1]*B.shape[1]))

        #calculate
        indptr,indices,data=fkron_csr_takerow(indptr1=A.indptr,indices1=A.indices,dat1=A.data,indptr2=B.indptr,indices2=B.indices,dat2=B.data,ncol2=B.shape[1],takerows=takerows,nnz=nnz)
    mat=sps.csr_matrix((data,indices,indptr),shape=(rowdim,A.shape[1]*B.shape[1]))
    return mat
