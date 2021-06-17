#ifndef DISTRIBUTED_SPARSE_H
#define DISTRIBUTED_SPARSE_H

#include <Eigen/Dense>
#include "sparse_kernels.h"
#include "common.h"

using namespace std;
using namespace Eigen;


class Distributed_Sparse {
public:
    // Pointer to object implementing the local SDDMM / SPMM Operations 
    KernelImplementation *kernel;

    MPI_Comm A_R_split_world, B_R_split_world;

    bool verbose;

    void setVerbose(bool value) {
        verbose = value;
    }

    virtual VectorXd like_S_values(double value) = 0;

    virtual DenseMatrix like_A_matrix(double value) = 0;

    virtual DenseMatrix like_B_matrix(double value) = 0;

    virtual void reset_performance_timers() = 0;

    virtual void print_algorithm_info() = 0;

    virtual void initial_synchronize(DenseMatrix *localA, DenseMatrix *localB, VectorXd *localS) = 0;

    virtual void spmmA(DenseMatrix &localA, DenseMatrix &localB, VectorXd &SValues) = 0;

    virtual void spmmB(DenseMatrix &localA, DenseMatrix &localB, VectorXd &SValues) = 0;

    virtual void sddmm(DenseMatrix &localA, DenseMatrix &localB, VectorXd &SValues, VectorXd &sddmm_result) = 0;

};

/*
void benchmark() {
VectorXd Svals        = like_S_values(1.0);
VectorXd sddmm_result = like_S_values(1.0);

DenseMatrix A = like_A_matrix(1.0); 
DenseMatrix B = like_B_matrix(1.0); 

spmmB(A, B, Svals);
reset_performance_timers();

int nruns = 10;
for(int i = 0; i < nruns; i++) {
	spmmB(A, B, Svals);
}
print_statistics();
}
*/

#endif