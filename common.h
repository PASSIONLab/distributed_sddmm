#ifndef COMMON_HEADER
#define COMMON_HEADER

#include <chrono>
#include <string>
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

	string string_rep();
};

bool column_major(spcoord_t a, spcoord_t b);

bool row_major(spcoord_t &a, spcoord_t &b);

extern MPI_Datatype SPCOORD;

void initialize_mpi_datatypes();

/*
 * Creates an array of roughly equal segments. Returns an array
 * of length num_segments + 1.
 */
void divideIntoSegments(int total, int num_segments, vector<int> &segment_starts, vector<int> &segment_sizes);

class BufferPair {
public:
	DenseMatrix* original;
	DenseMatrix* extra;

	int switchVal;

	BufferPair(DenseMatrix* buf) {
		switchVal = 0;
		original = buf;
		extra = new DenseMatrix(buf->rows(), buf->cols());
	}

	~BufferPair() {
		delete extra;
	}

	DenseMatrix* getActive() {
		if(switchVal == 0) {
			return original;	
		}
		else {
			return extra;
		}
	}

	DenseMatrix* getPassive() {
		if(switchVal == 0) {
			return extra;	
		}
		else {
			return original;
		}
	}

	void swapActive() {
		switchVal = 1 - switchVal;
	}

	void sync_active() {
		if(switchVal == 1) {
			*original = *extra;
		}
	}
};


#endif