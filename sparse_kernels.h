#ifndef SPARSE_KERNELS_H
#define SPARSE_KERNELS_H

#include <vector>
#include <utility>

#include "common.h"
#include "SpmatLocal.hpp"
#include <Eigen/Dense>

using namespace std;

typedef enum {k_sddmm, k_spmmA, k_spmmB} KernelMode;

class KernelImplementation {
public:
    // Performs an operation that looks like a local SDDMM
    // and returns the number of nonzeros processed 
    virtual size_t sddmm_local(
        SpmatLocal &S,
        DenseMatrix &A,
        DenseMatrix &B,
        uint64_t start,
        uint64_t end) = 0;

    /*
    * S is m x n
    * A is m x r
    * B is n x r 
    * When mode is 0, A = S B (output is A)
    * When mode is 1, B = S^T A (output is B)
    *
    */
    virtual size_t spmm_local(
        SpmatLocal &S,
        DenseMatrix &A,
        DenseMatrix &B,
        int mode,
        uint64_t start,
        uint64_t end) = 0;

    size_t triple_function(
            KernelMode mode,
            SpmatLocal &S,
            DenseMatrix &localA,
            DenseMatrix &localB,
            uint64_t start,
            uint64_t end) {
        
        size_t nnz_processed = 0;
        if(mode == k_sddmm) {
            nnz_processed += sddmm_local(
                S,
                localA,
                localB,
                start,
                end);
        }
        else if(mode == k_spmmA) { 
            nnz_processed += spmm_local(
                S,
                localA,
                localB,
                Amat,
                start,
                end);
        }
        else if(mode == k_spmmB) {
            nnz_processed += spmm_local(
                S,
                localA,
                localB,
                Bmat,
                start,
                end);
        }
        return nnz_processed;
    }
};

/*
 * Exactly the algebra on the box, no funny business. 
 */
class StandardKernel : public KernelImplementation {
public:
    size_t sddmm_local(
        SpmatLocal &S,
        DenseMatrix &A,
        DenseMatrix &B,
        uint64_t start,
        uint64_t end);

    size_t spmm_local(
        SpmatLocal &S,
        DenseMatrix &A,
        DenseMatrix &B,
        int mode,
        uint64_t start,
        uint64_t end);
};

/*
 * Allows a single SpMM operation to do both an SpMM and an SDDMM
 * in one shot. 
 */
/*class FusedStandardKernel : public KernelImplementation {
public:
    size_t sddmm_local(
        SpmatLocal &S,
        DenseMatrix &A,
        DenseMatrix &B, 
        uint64_t start,
        uint64_t end);

    size_t spmm_local(
        SpmatLocal &S,
        DenseMatrix &A,
        DenseMatrix &B,
        int mode,
        uint64_t start,
        uint64_t end);
};*/

#endif
