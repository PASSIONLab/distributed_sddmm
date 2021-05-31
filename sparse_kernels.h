#ifndef SPARSE_KERNELS_H
#define SPARSE_KERNELS_H

#include <vector>
#include <utility>

#include "common.h"
#include <Eigen/Dense>

using namespace std;

size_t sddmm_local(int64_t* rCoords,
    int64_t* cCoords,
    VectorXd Svalues,
    DenseMatrix &A,
    DenseMatrix &B,
    VectorXd result,
    int start,
    int end);

/*
 * S is m x n
 * A is m x r
 * B is n x r 
 * When mode is 0, A = S B
 * When mode is 1, B = S^T A
 *
 */
size_t spmm_local(int64_t* rCoords,
    int64_t* cCoords,
    VectorXd Svalues,
    DenseMatrix &A,
    DenseMatrix &B,
    int mode,
    int start,
    int end);

#endif
