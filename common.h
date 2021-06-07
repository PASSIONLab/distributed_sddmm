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


/*
 * I am aware that CombBLAS defines its own type for this, so I think this
 * is redundant. Might redefine / eliminate this type later...
 */ 
typedef struct {
    vector<int64_t> rCoords;
    vector<int64_t> cCoords;
    VectorXd Svalues;

    /*
     * This is the local portion of a larger sparse matrix. How many nonzeros
     * are there across the entire sparse matrix? vvv This variable gives that
     * quantity
     */
    int local_nnz;
    int dist_nnz;

    int nrows;
    int ncols;

} spmat_local_t;

#endif