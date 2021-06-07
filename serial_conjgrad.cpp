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
    X /= X.cols();
}

/*DenseMatrix computeQueries(DenseMatrix &v) {

}*/


void test_single_process_factorization(int logM, int nnz_per_row, int r) {
    // Generate latent factor Matrices
    int n = 1 << logM;

    // Generate two random sets of latent factors
    DenseMatrix Agt(n, r);
    DenseMatrix Bgt(n, r);

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
    VectorXd initial_sparse_contents = VectorXd::Constant(S.local_nnz, 1.0);
    VectorXd ground_truth(S.local_nnz);

    sddmm_local(S,
                initial_sparse_contents,
                Agt,
                Bgt,
                ground_truth,
                0, 
                S.local_nnz);

    // For now, all weights are uniform due to the Erdos Renyi Random matrix,
    // so just test for convergence of the uniformly weighted configuration. 

    double lambda = 0.1;
    
    // Initialize our guesses for A, B
    DenseMatrix A(n, r);
    DenseMatrix B(n, r);
    initialize_dense_matrix(A);
    initialize_dense_matrix(B);

    VectorXd sddmm_result(S.local_nnz);

    DenseMatrix spmmA(n, r);
    DenseMatrix spmmB(n, r);

    for(int cg_iter = 0; cg_iter < 1; cg_iter++) {
        // First optimize for A
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    //conjugate_gradients();
    //test_single_process_factorization(8, 8, 128);

    MPI_Finalize();
}
