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

typedef enum {Amat, Bmat} MatMode;

int pMod(int num, int denom);

int divideAndRoundUp(int num, int denom);

struct spcoord_t {
	uint64_t r;
	uint64_t c;
	double value;
};

bool sortbycolumns(spcoord_t &a, spcoord_t &b);

extern MPI_Datatype SPCOORD;

void initialize_mpi_datatypes();

#endif