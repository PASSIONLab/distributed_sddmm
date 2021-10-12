#include <vector>
#include <utility>
#include <immintrin.h>
#include <omp.h>
#include <iostream>
#include <mkl_spblas.h>
#include "sparse_kernels.h"
#include "SpmatLocal.hpp"
#include "common.h"

using namespace std;

size_t StandardKernel::sddmm_local(
    SpmatLocal &S, 
    DenseMatrix &A,
    DenseMatrix &B,
    int block,
    int offset
    ) {

    assert(A.cols() == B.cols());

    size_t processed = 0;

    if(S.csr_blocks[block] == nullptr) {
        return processed;
    }

    double *Aptr, *Bptr;
    if(S.csr_blocks[block]->transpose) {
        Aptr = B.data();
        Bptr = A.data();
    }
    else {
        Aptr = A.data();
        Bptr = B.data();
    }

    int num_coords = S.csr_blocks[block]->num_coords;
    CSRHandle* active = S.csr_blocks[block]->getActive();

    int r = A.cols();

    #pragma omp parallel for
    for(int i = 0; i < num_coords; i++) {
        double* Arow = Aptr + r * active->row_idx[i];
        double* Brow = Bptr + r * active->col_idx[i];

        double value = 0.0;
        #pragma ivdep
        for(int k = 0; k < r; k++) {
            value += Arow[k] * Brow[k];
        }
        active->values[i] += value;
    }
    return processed;
}

size_t StandardKernel::spmm_local(
    SpmatLocal &S,
    DenseMatrix &A,
    DenseMatrix &B,
    MatMode mode,
    int block) {

    size_t processed = 0;

	struct matrix_descr descr;
	descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    if(S.csr_blocks[block] == nullptr) {
        return processed;
    }

    if(mode == Amat && S.csr_blocks[block]->transpose) {
        cout << "Error, local matrix is transposed, can't perform SpmmA" << endl;
        exit(1);
    }

    else if(mode == Bmat && ! S.csr_blocks[block]->transpose) {
        cout << "Error, local matrix is not transposed, can't perform SpmmB" << endl;
        exit(1);
    }

    if(S.csr_blocks[block]->num_coords == 0) {
        return processed;
    }

    double* Aptr = A.data();
    double* Bptr = B.data();

    MKL_INT R = A.cols();

    if(mode == Amat) {
        mkl_sparse_d_mm (
                SPARSE_OPERATION_NON_TRANSPOSE,	
                1.0, 
                S.csr_blocks[block]->getActive()->mkl_handle,
                descr,	
                SPARSE_LAYOUT_ROW_MAJOR,	
                Bptr, 
                R, 
                R,  // ldb
                1.0, 
                Aptr, 
                R); // ldc	
    }
    else if(mode == Bmat) {
        mkl_sparse_d_mm (
                SPARSE_OPERATION_NON_TRANSPOSE,	
                1.0, 
                S.csr_blocks[block]->getActive()->mkl_handle,
                descr,	
                SPARSE_LAYOUT_ROW_MAJOR,	
                Aptr, 
                R, 
                R,  // ldb
                1.0, 
                Bptr, 
                R); // ldc	
    }
    else {
        assert(false);
    }

    return processed;
}