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

typedef chrono::time_point<std::chrono::steady_clock> my_timer_t; 

my_timer_t start_clock();
double stop_clock_get_elapsed(my_timer_t &start);

/*
 * I am aware that CombBLAS defines its own type for this, so I think this
 * is redundant. Might redefine / eliminate this type later...
 */ 
typedef struct {
    vector<int64_t> rCoords;
    vector<int64_t> cCoords;

    int local_nnz;
    int dist_nnz;

    int nrows;
    int ncols;

    int distrows;
    int distcols;
} spmat_local_t;

typedef enum {Amat, Bmat} MatMode;

int pMod(int num, int denom);

int divideAndRoundUp(int num, int denom);

#endif