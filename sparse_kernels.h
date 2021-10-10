#ifndef SPARSE_KERNELS_H
#define SPARSE_KERNELS_H

#include <vector>
#include <utility>

#include "common.h"
#include "SpmatLocal.hpp"
#include <Eigen/Dense>

using namespace std;

typedef enum {k_sddmmA, k_spmmA, k_spmmB, k_sddmmB} KernelMode;

class KernelImplementation {
public:
    // Performs an operation that looks like a local SDDMM
    // and returns the number of nonzeros processed 
    virtual size_t sddmm_local(
        SpmatLocal &S,
        DenseMatrix &A,
        DenseMatrix &B,
        int block,
        int offset 
        ) = 0;

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
        MatMode mode,
        int block) = 0;

    size_t triple_function(
            KernelMode mode,
            SpmatLocal &S,
            DenseMatrix &localA,
            DenseMatrix &localB,
            int block,
            int offset 
            ) {
        
        size_t nnz_processed = 0;
        if(mode == k_sddmmA || mode == k_sddmmB) {
            nnz_processed += sddmm_local(
                S,
                localA,
                localB,
                block,
                offset 
                );
        }
        else if(mode == k_spmmA) { 
            nnz_processed += spmm_local(
                S,
                localA,
                localB,
                Amat,
                block);
        }
        else if(mode == k_spmmB) {
            nnz_processed += spmm_local(
                S,
                localA,
                localB,
                Bmat,
                block);
        }
        return nnz_processed;
    }
};

/*
 * Exactly the algebra on the box, no funny business. 
 */
class StandardKernel : public KernelImplementation {
public:
    virtual size_t sddmm_local(
        SpmatLocal &S,
        DenseMatrix &A,
        DenseMatrix &B,
        int block,
        int offset);

    size_t spmm_local(
        SpmatLocal &S,
        DenseMatrix &A,
        DenseMatrix &B,
        MatMode mode,
        int block);
};

#endif
