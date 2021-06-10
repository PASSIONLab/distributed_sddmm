#include <iostream>
#include <memory>
#include <Eigen/Dense>
#include <cassert>
#include "mpi.h"
#include "io_utils.h"
#include "sparse_kernels.h"
#include "als_conjugate_gradients.h"
#include "CombBLAS/CombBLAS.h"

using namespace std;
using namespace Eigen;

class SingleNodeALS : public ALS_CG {
    spmat_local_t S;
    VectorXd ground_truth;
    int R;
public:
    // This constructor tests with a random matrix. 

    void initialize_dense_matrix(DenseMatrix &X) {
        X.setRandom();
        X /= X.cols();
    }

    SingleNodeALS(int logM, int nnz_per_row, int R) {
        // Generate latent factor Matrices
        this->R = R;
        int n = 1 << logM;

        // Generate two random sets of latent factors
        DenseMatrix Agt(n, R);
        DenseMatrix Bgt(n, R);

        initialize_dense_matrix(Agt);
        initialize_dense_matrix(Bgt);

        // Generate a random sparse matrix using CombBLAS
        shared_ptr<CommGrid> grid;
        grid.reset(new CommGrid(MPI_COMM_WORLD, 1, 1));

        VectorXd SValues; // We discard and generate ourselves 

        generateRandomMatrix(logM, 
            nnz_per_row,
            grid,
            S,
            SValues
        );

        // Compute a ground truth using an SDDMM, setting all sparse values to 1 
        VectorXd ones = VectorXd::Constant(S.local_nnz, 1.0);
        new (&ground_truth) VectorXd(S.local_nnz);

        sddmm_local(S,
                    ones,
                    Agt,
                    Bgt,
                    ground_truth,
                    0, 
                    S.local_nnz);

        // TODO: Maybe I should corrupt the entries with some random noise?

        A_R_split_world = MPI_COMM_WORLD;
        B_R_split_world = MPI_COMM_WORLD;
    }
  
    void computeRHS(MatMode matrix_to_optimize,
                            DenseMatrix &rhs) {
        if(matrix_to_optimize == Amat) {
            spmm_local(S, ground_truth, rhs, B, Amat, 0, S.local_nnz);
        }
        else {
            spmm_local(S, ground_truth, A, rhs, Bmat, 0, S.local_nnz);
        }
    }

    void initializeEmbeddings() {
        new (&A) DenseMatrix(S.nrows, R);
        new (&B) DenseMatrix(S.ncols, R);
    }

    double computeResidual() {
        VectorXd ones = VectorXd::Constant(S.local_nnz, 1.0);
        VectorXd sddmm_result = VectorXd::Zero(S.local_nnz);

        sddmm_local(S,
                    ones,
                    A,
                    B,
                    sddmm_result,
                    0, 
                    S.local_nnz);
        
        return (sddmm_result - ground_truth).norm();
    }

    void computeQueries(MatMode matrix_to_optimize,
                        DenseMatrix &result) {

        double lambda = 1e-6;

        VectorXd ones = VectorXd::Constant(S.local_nnz, 1.0);
        result.setZero();
        VectorXd sddmm_result = VectorXd::Zero(S.local_nnz);

        sddmm_local(S, 
                    ones,
                    A,
                    B,
                    sddmm_result,
                    0, 
                    S.local_nnz);

        if(matrix_to_optimize == Amat) {
            spmm_local(S, sddmm_result, result, B, Amat, 0, S.local_nnz);
            result += lambda * A;
        }
        else if(matrix_to_optimize == Bmat) {
            spmm_local(S, sddmm_result, A, result, Bmat, 0, S.local_nnz);
            result += lambda * B;
        }
    }

    ~SingleNodeALS() {
        // Empty destructor
    }
};

/*
void cg_optimizer(  spmat_local_t &S, 
                    VectorXd &ground_truth, 
                    DenseMatrix &A,
                    DenseMatrix &B,
                    MatMode matrix_to_optimize,
                    int cg_max_iter
                    ) {
    double cg_residual_tol = 1e-5;
    double nan_avoidance_constant = 1e-14;
    VectorXd ones = VectorXd::Constant(S.local_nnz, 1.0);

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

    if(matrix_to_optimize == Amat) {
        spmm_local(S, ground_truth, rhs, B, Amat, 0, S.local_nnz);
    }
    else {
        spmm_local(S, ground_truth, A, rhs, Bmat, 0, S.local_nnz);
    }
    computeQueries(S, A, B, matrix_to_optimize, Mx);

    DenseMatrix r = rhs - Mx;
    DenseMatrix p = r;
    VectorXd rsold = batch_dot_product(r, r); 

    // TODO: restabilize the residual to avoid numerical error
    // after a certain number of iterations

    int cg_iter;
    for(cg_iter = 0; cg_iter < cg_max_iter; cg_iter++) {

        if(matrix_to_optimize == Amat) {
            computeQueries(S, p, B, Amat, Mp);
        }
        else {
            computeQueries(S, A, p, Bmat, Mp);
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
        if(rsnew_norm_sqrt < cg_residual_tol) {
            break;
        }

        VectorXd coeffs = rsnew.cwiseQuotient(rsold);
        p = r + scale_matrix_rows(coeffs, p);
        rsold = rsnew;
    }
    if (cg_iter == cg_max_iter) {
        cout << "WARNING: Conjugate gradients did not converge to specified tolerance "
             << "in max iteration count." << endl;
    }
}
*/

/*void test_single_process_factorization(int logM, int nnz_per_row, int R) {
    // For now, all weights are uniform due to the Erdos Renyi Random matrix,
    // so just test for convergence of the uniformly weighted configuration. 

    spmat_local_t S;
    VectorXd ground_truth;

    gen_synthetic_factorization_matrix(
        logM, 
        nnz_per_row, 
        R,
        S,
        ground_truth
        );

    double lambda = 0.1;
    
    // Initialize our guesses for A, B
    DenseMatrix A(S.nrows, R);
    DenseMatrix B(S.ncols, R);
    initialize_dense_matrix(A);
    initialize_dense_matrix(B);

    cout << "Algorithm Initialized!" << endl;
    cout << "Initial Residual: " << computeResidual(A, B, S, ground_truth) << endl;

    int num_alternating_steps = 5;
    for(int i = 0; i < num_alternating_steps; i++) {
        cg_optimizer(S, ground_truth, A, B, Amat, 40);
        cout << "Residual: " << computeResidual(A, B, S, ground_truth) << endl;
        cg_optimizer(S, ground_truth, A, B, Bmat, 40);
        cout << "Residual: " << computeResidual(A, B, S, ground_truth) << endl;
    }
}*/

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    SingleNodeALS test(4, 4, 16);
    test.run_cg(5);

    //test_single_process_factorization(4, 4, 16);

    MPI_Finalize();
}