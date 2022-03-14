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
    DenseMatrix res = mat;
    double* resData = res.data();
    double* matData = res.data();
    int resRows = res.rows();
    int resCols = res.cols();

    #pragma omp parallel for collapse(2)
    for(int i = 0; i < resRows; i++) {
        for(int j = 0; j < resCols; j++) {
            double coeff = scale_vector[i];
            resData[i * resCols + j] = matData[i * resCols + j] * coeff;
        }
    }
    //return scale_vector.asDiagonal() * mat;
    return res;
}

void ALS_CG::allreduceVector(VectorXd &vec, MPI_Comm comm) {
    double start_time = MPI_Wtime();
    MPI_Allreduce(MPI_IN_PLACE, vec.data(), vec.size(), MPI_DOUBLE, MPI_SUM, comm);
    double end_time = MPI_Wtime();
    application_communication_time += end_time - start_time; 
}

void ALS_CG::cg_optimizer(MatMode matrix_to_optimize, int cg_max_iter) { 
    double cg_residual_tol = 1e-8;
    double nan_avoidance_constant = 1e-8;

    MPI_Comm reduction_world;
    if(matrix_to_optimize == Amat) {
        reduction_world = A_R_split_world; 
    }
    else if(matrix_to_optimize == Bmat){
        reduction_world = B_R_split_world; 
    }

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
    VectorXd alpha = rsold; // This just initializes the shape of alpha
    VectorXd coeffs = rsold;

    if(d_ops->r_split) {
        allreduceVector(rsold, reduction_world);
    }

    int cg_iter;

    for(cg_iter = 0; cg_iter < cg_max_iter; cg_iter++) {
        double start = MPI_Wtime();
        if(matrix_to_optimize == Amat) {
            computeQueries(p, B, Amat, Mp);
        }
        else {
            computeQueries(A, p, Bmat, Mp);
        }
        double end = MPI_Wtime();

        start = MPI_Wtime();
        VectorXd bdot = batch_dot_product(p, Mp);

        end = MPI_Wtime();

        if(d_ops->r_split) {
            allreduceVector(bdot, reduction_world);
        }

        bdot.array() += nan_avoidance_constant;
        rsold.array() += nan_avoidance_constant;

        alpha = rsold.cwiseQuotient(bdot);
        /*double* bdotData = bdot.data();
        double* rsoldData = rsold.data();
        double* alphaData = alpha.data();
        #pragma ivdep
        #pragma omp parallel for
        for(int i = 0; i < rsold.size(); i++) {
            alphaData[i] = rsoldData[i] / bdotData[i];
        }*/

        if(matrix_to_optimize == Amat) {
            A += scale_matrix_rows(alpha, p);
        }
        else {
            B += scale_matrix_rows(alpha, p);
        }
        r -= scale_matrix_rows(alpha, Mp);

        VectorXd rsnew = batch_dot_product(r, r);

        if(d_ops->r_split) {
            allreduceVector(rsnew, reduction_world);
        }

        //double rsnew_norm_sqrt = rsnew.sum();
        //MPI_Allreduce(MPI_IN_PLACE, &rsnew_norm_sqrt, 1, MPI_DOUBLE, MPI_SUM, residual_reduction_world);
        //rsnew_norm_sqrt = sqrt(rsnew_norm_sqrt);

        /* Uncomment this to re-enable early stopping for conjugate gradients
        if(rsnew_norm_sqrt < cg_residual_tol) {
            break;
        }
        */

        coeffs = rsnew.cwiseQuotient(rsold);
        p = r + scale_matrix_rows(coeffs, p);
        rsold = rsnew;
        end = MPI_Wtime();
    }
}

void initialize_dense_matrix(DenseMatrix &X, int R) {
    X.setRandom();
    X /= R;
}

Distributed_ALS::Distributed_ALS(Distributed_Sparse* d_ops, bool artificial_groundtruth) {
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

    this->residual_reduction_world = MPI_COMM_WORLD;
    this->A_R_split_world = d_ops->A_R_split_world;
    this->B_R_split_world = d_ops->B_R_split_world;

    this->d_ops = d_ops;

    if(artificial_groundtruth) {
        DenseMatrix Agt = d_ops->like_A_matrix(0.0);
        DenseMatrix Bgt = d_ops->like_B_matrix(0.0);

        initialize_dense_matrix(Agt, d_ops->R);
        initialize_dense_matrix(Bgt, d_ops->R);

        //d_ops->dummyInitialize(Agt, Amat);
        //d_ops->dummyInitialize(Bgt, Bmat);
        Agt /= d_ops->M * d_ops->R;
        Bgt /= d_ops->N * d_ops->R;

        // Compute a ground truth using an SDDMM, setting all sparse values to 1 
        VectorXd ones = d_ops->like_S_values(1.0);
        ground_truth = d_ops->like_S_values(0.0);

        // Initialization is random, but we should still do initial and final shifts 
        d_ops->initial_shift(&Agt, &Bgt, k_sddmmA);
        d_ops->sddmmA(Agt, Bgt, ones, ground_truth); 
        d_ops->de_shift(&Agt, &Bgt, k_sddmmA);

        ones = d_ops->like_ST_values(1.0);
        ground_truth_transpose = d_ops->like_ST_values(0.0); 

        d_ops->initial_shift(&Agt, &Bgt, k_sddmmB);
        d_ops->sddmmB(Agt, Bgt, ones, ground_truth_transpose);
        d_ops->de_shift(&Agt, &Bgt, k_sddmmB);
    }
    else {
        // TODO: This is broken!! Need a better way to initialize
        // the ground truth 
        //ground_truth = d_ops->input_Svalues; 
    }
}

void Distributed_ALS::computeRHS(MatMode matrix_to_optimize, DenseMatrix &rhs) {
    if(matrix_to_optimize == Amat) {
        // Can potentially optimize away the initial shift here! 
        d_ops->initial_shift(&rhs, &B, k_spmmA); 
        d_ops->spmmA(rhs, B, ground_truth);
        d_ops->de_shift(&rhs, &B, k_spmmA); 

    }
    else if(matrix_to_optimize == Bmat) {
        d_ops->initial_shift(&A, &rhs, k_spmmB); 
        d_ops->spmmB(A, rhs, ground_truth_transpose);
        d_ops->de_shift(&A, &rhs, k_spmmB);
    }
} 

double Distributed_ALS::computeResidual() {
    VectorXd ones = d_ops->like_S_values(1.0);
    VectorXd sddmm_result = d_ops->like_S_values(0.0); 

    d_ops->initial_shift(&A, &B, k_sddmmA); 
    d_ops->sddmmA(A, B, ones, sddmm_result);
    d_ops->de_shift(&A, &B, k_sddmmA); 

    double sqnorm = (sddmm_result - ground_truth).squaredNorm();
    MPI_Allreduce(MPI_IN_PLACE, &sqnorm, 1, MPI_DOUBLE, MPI_SUM, residual_reduction_world);
    
    return sqrt(sqnorm);
}

void Distributed_ALS::initializeEmbeddings() {
    A = d_ops->like_A_matrix(1.0);
    B = d_ops->like_B_matrix(1.0);

    initialize_dense_matrix(A, d_ops->R);
    initialize_dense_matrix(B, d_ops->R);

    //d_ops->dummyInitialize(A, Amat);
    //d_ops->dummyInitialize(B, Bmat);

    A *= 1.4;
    B /= 1.3;
}

void ALS_CG::run_cg(int n_alternating_steps) {
    initializeEmbeddings();

    //double residual = computeResidual();
    if(proc_rank == 0) {
        cout << "Embeddings initialized +" << endl;
        //cout << "Initial Residual: " << residual << endl;
    }

    for(int i = 0; i < n_alternating_steps; i++) {
        cg_optimizer(Amat, 10);
        cg_optimizer(Bmat, 10);

        //residual = computeResidual();

        /*if(i == n_alternating_steps - 1) {
            residual = computeResidual();
        }*/

        if(proc_rank == 0) {
            if(i < n_alternating_steps - 1) {
                cout << "Completed step " << i << endl;
            }
            else {
                //cout << "Residual after step " << i << " : " << residual << endl;
            }
        }
    }
}

void Distributed_ALS::computeQueries(
        DenseMatrix &A,
        DenseMatrix &B,
        MatMode matrix_to_optimize,
        DenseMatrix &result) {

    double lambda = 1e-13;

    result.setZero();

    VectorXd sddmm_result;
    VectorXd ones; 

    KernelMode mode;
    if(matrix_to_optimize == Amat) {
        ones = d_ops->like_S_values(1.0);
        sddmm_result = d_ops->like_S_values(0.0);
        mode = k_sddmmA;
        result = A;

        d_ops->initial_shift(&result, &B, mode);
        d_ops->fusedSpMM(result, B, ones, sddmm_result, matrix_to_optimize); 
        d_ops->de_shift(&result, &B, mode);
        result += lambda * A;
    }
    else {
        ones = d_ops->like_ST_values(1.0);
        sddmm_result = d_ops->like_ST_values(0.0);
        mode = k_sddmmB;
        result = B;
        d_ops->initial_shift(&A, &result, mode);
        d_ops->fusedSpMM(A, result, ones, sddmm_result, matrix_to_optimize); 
        d_ops->de_shift(&A, &result, mode);

        result += lambda * B;
    }
}

