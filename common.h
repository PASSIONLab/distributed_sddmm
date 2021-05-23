#ifndef COMMON_HEADER
#define COMMON_HEADER

#include <chrono>
#include "CombBLAS/CombBLAS.h"
#include <Eigen/Dense>

using namespace std;
using namespace combblas;
using namespace Eigen;


typedef Matrix<double, Dynamic, Dynamic, RowMajor> DenseMatrix;
typedef SpParMat < int64_t, int, SpDCCols<int32_t,int> > PSpMat_s32p64_Int;

chrono::time_point<std::chrono::steady_clock> start_clock() {
    return std::chrono::steady_clock::now();
}

void stop_clock_and_add(chrono::time_point<std::chrono::steady_clock> &start, double* timer) {
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;
    *timer += diff.count();
}


#endif