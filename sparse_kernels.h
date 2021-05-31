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


size_t spmm_local(int64_t* rCoords,
    int64_t* cCoords,
    VectorXd Svalues,
    DenseMatrix &A,
    DenseMatrix &B,
    int start,
    int end);

#endif
