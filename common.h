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

    int local_nnz;
    int dist_nnz;

    int nrows;
    int ncols;

    int distrows;
    int distcols;
} spmat_local_t;

class DistributedDenseMatrix {
public:
    MPI_Comm row_world; // Same block row
    MPI_Comm col_world; // Same block column
    MPI_Comm replication_world; 
    DenseMatrix localMatrix;
};

class DistributedVector {
    MPI_Comm dist_world;
    MPI_Comm replication_world;
    VectorXd localVector;
};

typedef enum {Amat, Bmat} MatMode;

int pMod(int num, int denom);

#endif