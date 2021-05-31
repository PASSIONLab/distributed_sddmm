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

chrono::time_point<std::chrono::steady_clock> start_clock();
void stop_clock_and_add(chrono::time_point<std::chrono::steady_clock> &start, double* timer);


#endif