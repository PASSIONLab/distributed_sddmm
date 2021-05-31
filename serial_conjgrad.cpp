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

void conjugate_gradients() {
    cout << "Starting serial conjugate gradient algorithm!" << endl; 

    int rows = 50;
    int cols = 20;

    DenseMatrix X(rows, cols);
    VectorXd b(rows);
    VectorXd z(cols);

    X.setRandom(rows, cols);
    b.setRandom(rows);

    VectorXd x = VectorXd::Zero(rows);
    DenseMatrix A = 0.01 * MatrixXd::Identity(rows, rows) + (X * X.transpose());// .selfadjointView<Eigen::Upper>();

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

    if(max_iterations_exceeded) {
        cout << "CG algorithm exceeded maximum number of iterations!" << endl;
    }
    else {
        cout << "CG algorithm converged in " << k << " iterations!" << endl;
    }

    VectorXd solution = A.colPivHouseholderQr().solve(b);

    cout << A * x - b << endl;
    // cout << A * solution - b << endl;
}

void test_single_process_factorization(int logM, int nnz_per_row, int r) {
    // Generate latent factor Matrices

    int n = 1 << logM;

    // Generate two random sets of latent factors
    DenseMatrix A(n, r);
    DenseMatrix B(n, r);
    A.setRandom();
    B.setRandom();
    A /= r;
    B /= r;

    // Generate a random sparse matrix using CombBLAS
    shared_ptr<CommGrid> grid;
    grid.reset(new CommGrid(MPI_COMM_WORLD, 1, 1));
    vector<int64_t> rCoords;
    vector<int64_t> cCoords;
    VectorXd Svalues;
    int total_nnz;

    generateRandomMatrix(logM, 
        nnz_per_row,
        grid,
        &total_nnz,
        &rCoords,
        &cCoords,
        &Svalues 
    );

    // Compute a ground truth using an SDDMM, setting all sparse values to 1 
    VectorXd initial_sparse_contents = VectorXd::Constant(total_nnz, 1.0);
    VectorXd ground_truth(total_nnz);

    sddmm_local(rCoords.data(),
                cCoords.data(),
                initial_sparse_contents,
                A,
                B,
                ground_truth,
                0, 
                total_nnz);

    // For now, all weights are uniform due to the Erdos Renyi Random matrix,
    // so just test for convergence.


    


}


int main(int argc, char** argv) {
    //conjugate_gradients();
    MPI_Init(&argc, &argv);

    test_single_process_factorization(8, 8, 128);

    MPI_Finalize();
}
