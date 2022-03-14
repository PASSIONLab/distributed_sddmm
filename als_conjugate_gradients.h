#pragma once

#include "common.h"
#include "mpi.h"
#include "distributed_sparse.h"
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

/*
 * Subclass this and implement its methods to create a distributed memory
 * (or single-noded shared memory) alternating least-squares factorization algorithm
 * for a sparse matrix.
 *
 * The only two data members needed for ALS are the dense matrices A and B,
 * along with two MPI communicators in case that an individual embedding in either
 * matrix is sharded along an axis. This also applies to matrices that are distributed
 * in the same manner as either A or B. Note that A and B may be replicated, etc.,
 * and the subclasser is responsible for keeping those matrices in sync, etc. The
 * user takes all responsibility for managing the sparse matrix. 
 */

class ALS_CG {
public:
    Distributed_Sparse* d_ops;
    DenseMatrix A;
    DenseMatrix B;

    MPI_Comm A_R_split_world;
    MPI_Comm B_R_split_world;

    MPI_Comm residual_reduction_world;

    int proc_rank;

    double application_communication_time;

    virtual void computeRHS(MatMode matrix_to_optimize,
                            DenseMatrix &rhs) = 0;

    // TODO: I know these variable names shadow, I need to fix that... 
    virtual void computeQueries(DenseMatrix &A,
                                DenseMatrix &B, 
                                MatMode matrix_to_optimize,
                                DenseMatrix &result) = 0;

    virtual double computeResidual() = 0;

    virtual void initializeEmbeddings() = 0;

    void allreduceVector(VectorXd &vec, MPI_Comm comm);

    void cg_optimizer(  MatMode matrix_to_optimize,
                        int cg_max_iter
                        );

    void run_cg(int n_alternating_steps);
 
    virtual ~ALS_CG() { }
};


class Distributed_ALS : public ALS_CG {
public:
    VectorXd ground_truth;
    VectorXd ground_truth_transpose;

    Distributed_ALS(Distributed_Sparse* d_ops, bool artifical_groundtruth);

    void computeRHS(MatMode matrix_to_optimize,
                            DenseMatrix &rhs);

    // TODO: I know these variable names shadow, I need to fix that... 
    void computeQueries(DenseMatrix &A,
                                DenseMatrix &B, 
                                MatMode matrix_to_optimize,
                                DenseMatrix &result);

    double computeResidual();

    void initializeEmbeddings();

};
