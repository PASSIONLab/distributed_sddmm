#include "als_conjugate_gradients.h"
#include "common.h"
#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

VectorXd batch_dot_product(DenseMatrix &A, DenseMatrix &B) {
    return A.cwiseProduct(B).rowwise().sum();
} 

DenseMatrix scale_matrix_rows(VectorXd &scale_vector, DenseMatrix &mat) {
    return scale_vector.asDiagonal() * mat;
}

void ALS_CG::cg_optimizer(MatMode matrix_to_optimize, int cg_max_iter) { 
    double cg_residual_tol = 1e-3;
    double nan_avoidance_constant = 1e-14;

    int nrows, ncols;
    ncols = A.cols();                // A and B have the same # of columns 
    if(matrix_to_optimize == Amat) {
        nrows = A.rows();
    }
    else {
        nrows = B.rows(); 
    }

    DenseMatrix rhs(nrows, ncols);
    DenseMatrix Mx(nrows, ncols);
    DenseMatrix Mp(nrows, ncols);

    rhs.setZero();

    computeRHS(matrix_to_optimize, rhs);
    computeQueries(A, B, matrix_to_optimize, Mx);

    DenseMatrix r = rhs - Mx;
    DenseMatrix p = r;
    VectorXd rsold = batch_dot_product(r, r); 

    // TODO: restabilize the residual to avoid numerical error
    // after a certain number of iterations

    int cg_iter;
    for(cg_iter = 0; cg_iter < cg_max_iter; cg_iter++) {

        if(matrix_to_optimize == Amat) {
            computeQueries(p, B, Amat, Mp);
        }
        else {
            computeQueries(A, p, Bmat, Mp);
        }
        VectorXd bdot = batch_dot_product(p, Mp);
        bdot.array() += nan_avoidance_constant; 
        VectorXd alpha = rsold.cwiseQuotient(bdot);

        if(matrix_to_optimize == Amat) {
            A += scale_matrix_rows(alpha, p);
        }
        else {
            B += scale_matrix_rows(alpha, p);
        }
        r -= scale_matrix_rows(alpha, Mp);

        VectorXd rsnew = batch_dot_product(r, r); 
 
        double rsnew_norm_sqrt = sqrt(rsnew.sum());

        MPI_Allreduce(MPI_IN_PLACE, &rsnew_norm_sqrt, 1, MPI_DOUBLE, MPI_SUM, residual_reduction_world);

        rsnew_norm_sqrt = sqrt(rsnew_norm_sqrt);

        if(rsnew_norm_sqrt < cg_residual_tol) {
            break;
        }

        VectorXd coeffs = rsnew.cwiseQuotient(rsold);
        p = r + scale_matrix_rows(coeffs, p);
        rsold = rsnew;
    }
    //if (cg_iter == cg_max_iter) {
    //    cout << "WARNING: Conjugate gradients did not converge to specified tolerance "
    //        << "in max iteration count." << endl;
    //}
}

void ALS_CG::run_cg(int n_alternating_steps) {
    initializeEmbeddings();

    double residual = computeResidual();
    if(proc_rank == 0) {
        cout << "Embeddings initialized" << endl;
        cout << "Initial Residual: " << residual << endl;
    }

    for(int i = 0; i < n_alternating_steps; i++) {
        cg_optimizer(Amat, 30);
        cg_optimizer(Bmat, 30);

        residual = computeResidual();
        if(proc_rank == 0) {
            cout << "Residual after step " << i << " : " << residual << endl;
        }
    }
}


void initialize_dense_matrix(DenseMatrix &X) {
    X.setRandom();
    X /= X.cols();
}

Distributed_ALS::Distributed_ALS(Distributed_Sparse* d_ops, MPI_Comm residual_reduction_world) {
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

    this->residual_reduction_world = residual_reduction_world;
    this->d_ops = d_ops;

    DenseMatrix Agt = d_ops->like_A_matrix(0.0);
    DenseMatrix Bgt = d_ops->like_B_matrix(0.0);

    initialize_dense_matrix(Agt);
    initialize_dense_matrix(Bgt);

    // Compute a ground truth using an SDDMM, setting all sparse values to 1 
    VectorXd ones = d_ops->like_S_values(1.0);
    ground_truth = d_ops->like_S_values(0.0); 

    d_ops->initial_synchronize(&Agt, &Bgt, nullptr);
    d_ops->sddmm(Agt, Bgt, ones, ground_truth);
}

void Distributed_ALS::computeRHS(MatMode matrix_to_optimize, DenseMatrix &rhs) {
    if(matrix_to_optimize == Amat) {
        d_ops->spmmA(rhs, B, ground_truth);
    }
    else if(matrix_to_optimize == Bmat) {
        d_ops->spmmB(A, rhs, ground_truth);
    }
} 

double Distributed_ALS::computeResidual() {
    VectorXd ones = d_ops->like_S_values(1.0);
    VectorXd sddmm_result = d_ops->like_S_values(0.0); 
    d_ops->sddmm(A, B, ones, sddmm_result);

    double sqnorm = (sddmm_result - ground_truth).squaredNorm();

    MPI_Allreduce(MPI_IN_PLACE, &sqnorm, 1, MPI_DOUBLE, MPI_SUM, residual_reduction_world);
    
    return sqrt(sqnorm);
}

void Distributed_ALS::initializeEmbeddings() {
    A = d_ops->like_A_matrix(0.0);
    B = d_ops->like_B_matrix(0.0);

    initialize_dense_matrix(A);
    initialize_dense_matrix(B);
    d_ops->initial_synchronize(&A, &B, nullptr);
}

void Distributed_ALS::computeQueries(
                    DenseMatrix &A,
                    DenseMatrix &B,
                    MatMode matrix_to_optimize,
                    DenseMatrix &result) {

    double lambda = 1e-8;

    result.setZero();
    VectorXd sddmm_result = d_ops->like_S_values(0.0); 
    VectorXd ones = d_ops->like_S_values(1.0);

    d_ops->sddmm(A, B, ones, sddmm_result);

    if(matrix_to_optimize == Amat) {
        d_ops->spmmA(result, B, sddmm_result);
        result += lambda * A;
    }
    else if(matrix_to_optimize == Bmat) {
        d_ops->spmmB(A, result, sddmm_result);
        result += lambda * B;
    }
}

