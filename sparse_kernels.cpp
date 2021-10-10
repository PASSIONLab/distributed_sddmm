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

inline double vectorized_dot_product(double* A, double* B, size_t r) {
    /*__m512d lane1 = _mm512_setzero_pd();

    #pragma GCC unroll 20
    for(int j = 0; j < r; j+=8) {
        __m512d Avec1 = _mm512_loadu_pd(A + j);
        __m512d Bvec1 = _mm512_loadu_pd(B + j);

        lane1 = _mm512_fmadd_pd(Avec1, Bvec1, lane1);
    }
    return (_mm512_reduce_add_pd(lane1));*/

    double sum = 0;
    for(int j = 0; j < r; j++) {
        sum += A[j] * B[j];
    }
    return sum;
}


// Parameters: output row, input row, coefficient
inline void row_fmadd(double* A, double* B, double coeff, size_t r) {
    /*for(int j = 0; j < r; j+=8) {
        __m512d Avec1 = _mm512_loadu_pd(A + j);
        __m512d Bvec1 = _mm512_loadu_pd(B + j);

        Avec1 = _mm512_fmadd_pd(_mm512_set1_pd(coeff), Bvec1, Avec1);

        _mm512_storeu_pd(A + j, Avec1);
    }*/

    for(int j = 0; j < r; j++) {
        A[j] += coeff * B[j];
    }
}

// Just for consistency, we're going to code this with a CSR
// representation 
size_t StandardKernel::sddmm_local(
    SpmatLocal &S, 
    DenseMatrix &A,
    DenseMatrix &B,
    int block,
    int offset
    ) {

    assert(A.cols() == B.cols());

    size_t processed = 0;

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

    //#pragma omp parallel for
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

    /*if(mode == Amat) {
        for(int i = S.blockStarts[block]; i < S.blockStarts[block + 1]; i++) {
            double* Arow = Aptr + R * S.coords[i].r;
            double* Brow = Bptr + R * S.coords[i].c; 
            
            for(int t = 0; t < R; t++) {
                Arow[t] += Brow[t] * S.coords[i].value;
            }
        }
    }
    else {
        for(int i = S.blockStarts[block]; i < S.blockStarts[block + 1]; i++) {
            double* Arow = Aptr + R * S.coords[i].r;
            double* Brow = Bptr + R * S.coords[i].c;

            for(int t = 0; t < R; t++) {
                Brow[t] += Arow[t] * S.coords[i].value;
            }
        }
    }*/

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