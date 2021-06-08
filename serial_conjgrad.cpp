#include <iostream>
#include <memory>
#include <Eigen/Dense>

#include "mpi.h"
#include "io_utils.h"
#include "sparse_kernels.h"
#include "CombBLAS/CombBLAS.h"

using namespace std;
using namespace Eigen;

// The simplest possible program: solves a random dense linear system using
// conjugate gradients, and compares it to an exact solution. 

// Demmel, Applied Numerical Linear Algebra: Conjugate Gradient Algorithm
// is on page 312 

void batch_conjugate_gradient_step(MatrixXd vectors, MatrixXd queries) {
    cout << "Starting serial conjugate gradient algorithm!" << endl; 

    int rows = 50;
    int cols = 20;

    DenseMatrix X(rows, cols);
    VectorXd b(rows);
    VectorXd z(cols);

    X.setRandom(rows, cols);
    b.setRandom(rows);

    VectorXd x = VectorXd::Zero(rows);
    DenseMatrix A = 0.01 * MatrixXd::Identity(rows, rows) + (X * X.transpose());

    double tol = 1e-10;
    int max_iter = 10000;
    int k = 0;

    // Conjugate gradient algorithm taken directly from Wikipedia 
    VectorXd r = b;
    VectorXd p = r;
    double rsold = r.dot(r);

    bool max_iterations_exceeded = true; 
    while(k < b.size()) {
        k++;
        VectorXd Ap = A * p;
        
        double alpha = rsold / p.dot(Ap);
        x += alpha * p;
        r -= alpha * Ap;
        double rsnew = r.dot(r);

        if(sqrt(rsnew) < tol) {
            max_iterations_exceeded = false;
            break;
        }

        p = r + (rsnew / rsold) * p;
        rsold = rsnew;
    }
}

void initialize_dense_matrix(DenseMatrix &X) {
    X.setRandom();
    //X /= X.cols();
}

double computeResidual(
    DenseMatrix &A, 
    DenseMatrix &B, 
    spmat_local_t &S,
    VectorXd &gt
) {
    VectorXd ones = VectorXd::Constant(S.local_nnz, 1.0);
    VectorXd sddmm_result(S.local_nnz);

    sddmm_local(S,
                ones,
                A,
                B,
                sddmm_result,
                0, 
                S.local_nnz);
    
    return (sddmm_result - gt).norm();
}

double computeResidualDense(DenseMatrix &A, DenseMatrix &B, DenseMatrix &mask, DenseMatrix &C_mat) {
    DenseMatrix sddmm = (A * B.transpose()).cwiseProduct(mask);
    return (C_mat - sddmm).norm();
}

DenseMatrix makeDense(spmat_local_t &S, VectorXd &values) {
    DenseMatrix res(S.nrows, S.ncols);
    res.setZero();
    for(int i = 0; i < S.rCoords.size(); i++) {
        res(S.rCoords[i], S.cCoords[i]) = values(i); 
    }
    return res;
}

void computeQueriesDense(
                    DenseMatrix &S, 
                    DenseMatrix &B, 
                    DenseMatrix &x, 
                    DenseMatrix &result) {

    DenseMatrix sddmm = (x * B.transpose()).cwiseProduct(S);
    result = sddmm * B + 0.1 * x;
}

void computeQueries(spmat_local_t &S, 
                    DenseMatrix &B, 
                    DenseMatrix &x, 
                    DenseMatrix &result) {
    VectorXd ones = VectorXd::Constant(S.local_nnz, 1.0);

    VectorXd sddmm_result(S.local_nnz);

    sddmm_local(S, 
                ones,
                x,
                B,
                sddmm_result,
                0, 
                S.local_nnz);
    spmm_local(S, sddmm_result, result, B, 0, 0, S.local_nnz);
    result += 0.01 * x;
}

VectorXd batch_dot_product(DenseMatrix &A, DenseMatrix &B) {
    return A.cwiseProduct(B).rowwise().sum();
} 

DenseMatrix scale_matrix_rows(VectorXd &scale_vector, DenseMatrix &mat) {
    return scale_vector.asDiagonal() * mat;
}

void test_single_process_factorization(int logM, int nnz_per_row, int R) {
    // Generate latent factor Matrices
    int n = 1 << logM;

    // Generate two random sets of latent factors
    DenseMatrix Agt(n, R);
    DenseMatrix Bgt(n, R);

    initialize_dense_matrix(Agt);
    initialize_dense_matrix(Bgt);

    // Generate a random sparse matrix using CombBLAS
    shared_ptr<CommGrid> grid;
    grid.reset(new CommGrid(MPI_COMM_WORLD, 1, 1));

    spmat_local_t S;
    VectorXd SValues;

    generateRandomMatrix(logM, 
        nnz_per_row,
        grid,
        S,
        SValues
    );

    // Compute a ground truth using an SDDMM, setting all sparse values to 1 
    VectorXd ones = VectorXd::Constant(S.local_nnz, 1.0);
    VectorXd ground_truth(S.local_nnz);

    DenseMatrix mask = makeDense(S, ones);
    DenseMatrix C_mat = (Agt * Bgt.transpose()).cwiseProduct(mask);

    /*sddmm_local(S,
                ones,
                Agt,
                Bgt,
                ground_truth,
                0, 
                S.local_nnz);
    */


    // For now, all weights are uniform due to the Erdos Renyi Random matrix,
    // so just test for convergence of the uniformly weighted configuration. 

    double lambda = 0.1;
    
    // Initialize our guesses for A, B
    DenseMatrix A(n, R);
    DenseMatrix B(n, R);
    initialize_dense_matrix(A);
    initialize_dense_matrix(B);

    cout << "Algorithm Initialized!" << endl;
    cout << "Testing Residual Computation: " << computeResidualDense(Agt, Bgt, mask, C_mat) << endl;

    DenseMatrix rhs(A.rows(), A.cols());
    DenseMatrix Ax(A.rows(), A.cols());
    DenseMatrix Ap(A.rows(), A.cols());

    //spmm_local(S, ground_truth, rhs, B, 0, 0, S.local_nnz);
    rhs = C_mat * B;

    cout << mask << endl;

    // computeQueries(S, B, A, Ax);
    computeQueriesDense(mask, B, A, Ax);
    DenseMatrix r = rhs - Ax;
    DenseMatrix p = r;
    VectorXd rsold = batch_dot_product(r, r); 

    double tol = 1e-8;

    // First optimize for A
    for(int cg_iter = 0; cg_iter < 2; cg_iter++) {
        //cout << "True Residual: " << computeResidual(A, B, S, ground_truth) << endl;
        cout << "True Residual: " << computeResidualDense(A, B, mask, C_mat) << endl;


        // computeQueries(S, B, p, Ap);
        computeQueriesDense(mask, B, p, Ap);
        VectorXd bdot = batch_dot_product(p, Ap);
        bdot.array() += 1e-14; // For numerical stability 
        VectorXd alpha = rsold.cwiseQuotient(bdot);

        A += scale_matrix_rows(alpha, p);
        r -= scale_matrix_rows(alpha, Ap);

        VectorXd rsnew = batch_dot_product(r, r); 
        
        double rsnew_norm_sqrt = sqrt(rsnew.sum());

        cout << "Stopping Condition Residual: " << rsnew_norm_sqrt << endl;

        if(rsnew_norm_sqrt < tol) {
            break;
        }

        VectorXd coeffs = rsnew.cwiseQuotient(rsold);

        p = r + scale_matrix_rows(coeffs, p);
        rsold = rsnew;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    //conjugate_gradients();
    test_single_process_factorization(4, 4, 8);

    MPI_Finalize();
}