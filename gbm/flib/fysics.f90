!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!Fortran version kronercker product for coo_matrix
!Author: Leo
!Data: Mar. 11. 2016
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!Kronercker product
!
!Parameters:
!   row1, col1, dat1, row2, col2, dat2, the rows, columns and datas defining the 1st and 2nd coo_matrices.
!   n1, n2, the number of non zero items in 1st and 2nd matrices.
!   nrow2, ncol2, the number of rows and columns of the 2nd matrix.
!
!Return:
!   rown, coln, datn, the row, column and data of the new coo_matrix.
subroutine fkron_coo(row1,col1,dat1,row2,col2,dat2,n1,n2,nrow2,ncol2,rown,coln,datn)
    implicit none
    integer,intent(in) :: n1,n2,nrow2,ncol2
    integer,intent(in) :: row1(n1),col1(n1),row2(n2),col2(n2)
    complex*16,intent(in) :: dat1(n1),dat2(n2)
    integer,intent(out) :: rown(n1*n2),coln(n1*n2)
    complex*16,intent(out) :: datn(n1*n2)
    integer :: op,ed,i
    
    !f2py intent(in) :: n1,n2,nrow2,ncol2,row1,col1,dat1,row2,col2,dat2
    !f2py intent(out) :: rown,coln,datn

    do i=1,n1
        op=(i-1)*n2+1
        ed=i*n2
        rown(op:ed)=row1(i)*nrow2+row2
        coln(op:ed)=col1(i)*ncol2+col2
        datn(op:ed)=dat1(i)*dat2
    enddo
end subroutine fkron_coo

!Kronercker product for scr matrices.
!
!Parameters:
!   indptr1, indices1, dat1, indptr2, indices2, dat2, the row pointers, columns and datas defining the 1st and 2nd coo_matrices.
!   nrow1, nrow2, the number of non zero items in 1st and 2nd matrices.
!   nnz1, nnz2, the number of non zero elements in 1st and 1nd matrices.
!
!Return:
!   indptrn, indicesn, datn, the row pointers, columns and datas of the new coo_matrix.
subroutine fkron_csr(indptr1,indices1,dat1,indptr2,indices2,dat2,nrow1,nnz1,nrow2,nnz2,indptrn,indicesn,datn,ncol2)
    implicit none
    integer,intent(in) :: nrow1,nnz1,nrow2,nnz2,ncol2
    integer,intent(in) :: indptr1(nrow1+1),indptr2(nrow2+1)
    integer,intent(in) :: indices1(nnz1),indices2(nnz2)
    complex*16,intent(in) :: dat1(nnz1),dat2(nnz2)
    integer,intent(out) :: indptrn(nrow1*nrow2+1),indicesn(nnz1*nnz2)
    complex*16,intent(out) :: datn(nnz1*nnz2)
    integer :: op,ed,i1,i2,jn1,j1,start1,start2,stop1,stop2
    
    !f2py intent(in) :: nnz1,nnz2,nrow1,nrow2,ncol2,indptr1,indices1,dat1,indptr2,indices2,dat2
    !f2py intent(out) :: indptrn,indicesn,datn

    ed=0
    do i1=1,nrow1
        start1=indptr1(i1)
        stop1=indptr1(i1+1)
        do i2=1,nrow2
            indptrn((i1-1)*nrow2+i2)=ed  !setup indptr
            start2=indptr2(i2)
            stop2=indptr2(i2+1)
            do jn1=1,stop1-start1
                j1=indices1(start1+jn1)
                op=ed+1
                ed=ed+(stop2-start2) !calculate nonzero elements in this row.
                datn(op:ed)=dat1(start1+jn1)*dat2(start2+1:stop2)
                indicesn(op:ed)=j1*ncol2+indices2(start2+1:stop2)
            enddo
        enddo
    enddo
    indptrn(nrow1*nrow2+1)=ed
end subroutine fkron_csr


!Kronercker product for scr matrices, taking rows.
!
!Parameters:
!   indptr1, indices1, dat1, indptr2, indices2, dat2, the row pointers, columns and datas defining the 1st and 2nd coo_matrices.
!   nrow1, nrow2, the number of non zero items in 1st and 2nd matrices.
!   nnz1, nnz2, the number of non zero elements in 1st and 1nd matrices.
!   takerows, 1darray the rows desired.
!   ntake, the number of rows desired
!
!Return:
!   indptrn, indicesn, datn, the row pointers, columns and datas of the new coo_matrix.
subroutine fkron_csr_takerow(indptr1,indices1,dat1,indptr2,indices2,dat2,&
        nrow1,nnz1,nrow2,nnz2,indptrn,indicesn,datn,ncol2,takerows,ntake,nnz)
    implicit none
    integer,intent(in) :: nrow1,nnz1,nrow2,nnz2,ncol2,ntake,nnz
    integer,intent(in) :: indptr1(nrow1+1),indptr2(nrow2+1),takerows(ntake)
    integer,intent(in) :: indices1(nnz1),indices2(nnz2)
    complex*16,intent(in) :: dat1(nnz1),dat2(nnz2)
    integer,intent(out) :: indptrn(ntake+1),indicesn(nnz)
    complex*16,intent(out) :: datn(nnz)
    integer :: op,ed,i1,i2,jn1,j1,start1,start2,stop1,stop2,ii
    
    !f2py intent(in) :: nnz1,nnz2,nrow1,nrow2,ncol2,indptr1,indices1,dat1,indptr2,indices2,dat2,takerows,ntake,nnz
    !f2py intent(out) :: indptrn,indicesn,datn

    ed=0
    do ii=1,ntake  !iterate over rows.
        !get i1,i2
        i2=takerows(ii)
        i1=i2/nrow2
        i2=i2-nrow2*i1+1
        i1=i1+1

        start1=indptr1(i1)
        stop1=indptr1(i1+1)
        start2=indptr2(i2)
        stop2=indptr2(i2+1)
        indptrn(ii)=ed  !setup indptr

        do jn1=1,stop1-start1
            j1=indices1(start1+jn1)
            op=ed+1
            ed=ed+(stop2-start2) !calculate nonzero elements in this row.
            datn(op:ed)=dat1(start1+jn1)*dat2(start2+1:stop2)
            indicesn(op:ed)=j1*ncol2+indices2(start2+1:stop2)
        enddo
    enddo
    indptrn(ntake+1)=ed
end subroutine fkron_csr_takerow
