#include <vector>
#include <utility>
#include <immintrin.h>
#include <omp.h>
#include <iostream>

#include "sparse_kernels.h"
#include "SpmatLocal.hpp"
#include "common.h"

using namespace std;

inline double vectorized_dot_product(double* A, double* B, size_t r) {
        __m512d lane1 = _mm512_setzero_pd();

        #pragma GCC unroll 20
        for(int j = 0; j < r; j+=8) {
            __m512d Avec1 = _mm512_loadu_pd(A + j);
            __m512d Bvec1 = _mm512_loadu_pd(B + j);

            lane1 = _mm512_fmadd_pd(Avec1, Bvec1, lane1);
        }
        return (_mm512_reduce_add_pd(lane1));
}


// Parameters: output row, input row, coefficient
inline void row_fmadd(double* A, double* B, double coeff, size_t r) {
    for(int j = 0; j < r; j+=8) {
        __m512d Avec1 = _mm512_loadu_pd(A + j);
        __m512d Bvec1 = _mm512_loadu_pd(B + j);

        Avec1 = _mm512_fmadd_pd(_mm512_set1_pd(coeff), Bvec1, Avec1);

        _mm512_storeu_pd(A + j, Avec1);
    }
}

// TODO: Add assertions making sure all of the sizes match 
size_t StandardKernel::sddmm_local(
    SpmatLocal &S, 
    VectorXd &SValues,
    DenseMatrix &A,
    DenseMatrix &B,
    VectorXd& result,
    uint64_t start,
    uint64_t end) {

    size_t processed = 0;

    double* Aptr = A.data();
    double* Bptr = B.data();
    double* Sptr = SValues.data();
    double* res = result.data();
    int r = A.cols();

    //#pragma omp parallel for reduction(+:processed)

    #pragma omp parallel for
    for(int i = start; i < end; i++) {
        //processed++;
        double* Arow = Aptr + r * S.coords[i].r;
        double* Brow = Bptr + r * S.coords[i].c; 
        res[i] += Sptr[i] * vectorized_dot_product(Arow, Brow, r); 
    }
    return processed;
}

size_t StandardKernel::spmm_local(
    SpmatLocal &S,
    VectorXd &SValues,
    DenseMatrix &A,
    DenseMatrix &B,
    int mode,
    uint64_t start,
    uint64_t end) {

    size_t processed = 0;

    double* Aptr = A.data();
    double* Bptr = B.data();
    double* Sptr = SValues.data();
    int r = A.cols();

    //#pragma omp parallel for reduction(+:processed)
    
    #pragma omp parallel for
    for(int i = start; i < end; i++) {
        //processed++;

        if(mode == 0) {
            double* Arow = Aptr + r * S.coords[i].r;
            double* Brow = Bptr + r * S.coords[i].c; 
            row_fmadd(Arow, Brow, Sptr[i], r); 
        }
        else if(mode == 1) {
            double* Arow = Aptr + r * S.coords[i].r;
            double* Brow = Bptr + r * S.coords[i].c; 
            row_fmadd(Brow, Arow, Sptr[i], r); 
        }
        else {
            assert(false);
        }
    }
    return processed;
}

size_t FusedStandardKernel::sddmm_local(
    SpmatLocal &S, 
    VectorXd &SValues,
    DenseMatrix &A,
    DenseMatrix &B,
    VectorXd& result,
    uint64_t start,
    uint64_t end) {

    // Does the same operation as the standard kernel
    size_t processed = 0;

    double* Aptr = A.data();
    double* Bptr = B.data();
    double* Sptr = SValues.data();
    double* res = result.data();
    int r = A.cols();

    #pragma omp parallel for
    for(int i = start; i < end; i++) {
        processed++;
        double* Arow = Aptr + r * S.coords[i].r;
        double* Brow = Bptr + r * S.coords[i].c; 
        res[i] += Sptr[i] * vectorized_dot_product(Arow, Brow, r); 
    }
    return processed;
}

size_t FusedStandardKernel::spmm_local(
    SpmatLocal &S,
    VectorXd &SValues,
    DenseMatrix &A,
    DenseMatrix &B,
    int mode,
    uint64_t start,
    uint64_t end) {

    size_t processed = 0;

    double* Aptr = A.data();
    double* Bptr = B.data();
    double* Sptr = SValues.data();
    int r = A.cols();

    // #pragma omp parallel for
    for(int i = start; i < end; i++) {
        processed++;

        if(mode == 0) {
            double* Arow = Aptr + r * S.coords[i].r;
            double* Brow = Bptr + r * S.coords[i].c;
            
            double coeff = Sptr[i] * vectorized_dot_product(Arow, Brow, r);
            //double coeff = Sptr[i];

            row_fmadd(Arow, Brow, coeff, r); 
        }
        else if(mode == 1) {
            double* Arow = Aptr + r * S.coords[i].r;
            double* Brow = Bptr + r * S.coords[i].c; 
     
            double coeff = Sptr[i] * vectorized_dot_product(Arow, Brow, r);

            row_fmadd(Brow, Arow, coeff, r); 
        }
        else {
            assert(false);
        }
    }
    return processed;
}

