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

VectorXd b_dot(DenseMatrix &A, DenseMatrix &B) {
    return A.cwiseProduct(B).rowwise().sum();
} 

DenseMatrix scale_rows(VectorXd &scale_vector, DenseMatrix &mat) {
    return scale_vector.asDiagonal() * mat;
}

class SingleNodeALS : public ALS_CG {
public:
    spmat_local_t S;
    VectorXd ground_truth;
    int R;
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
        initialize_dense_matrix(A);
        initialize_dense_matrix(B);
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
                        DenseMatrix &queries,
                        DenseMatrix &result) {

        double lambda = 1e-6;

        VectorXd ones = VectorXd::Constant(S.local_nnz, 1.0);
        result.setZero();
        VectorXd sddmm_result = VectorXd::Zero(S.local_nnz);

        if(matrix_to_optimize == Amat) {
            sddmm_local(S, 
                        ones,
                        queries,
                        B,
                        sddmm_result,
                        0, 
                        S.local_nnz);

            spmm_local(S, sddmm_result, result, B, Amat, 0, S.local_nnz);
            result += lambda * A;
        }
        else if(matrix_to_optimize == Bmat) {
            sddmm_local(S, 
                        ones,
                        A,
                        queries,
                        sddmm_result,
                        0, 
                        S.local_nnz);

            spmm_local(S, sddmm_result, A, result, Bmat, 0, S.local_nnz);
            result += lambda * B;
        }
    }



    ~SingleNodeALS() {
        // Empty destructor
    }


void computeQueriesControl(
                    spmat_local_t &S,
                    DenseMatrix &A,
                    DenseMatrix &B,
                    MatMode matrix_to_optimize,
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


void cg_optimizer_control(
                    spmat_local_t &S,
                    VectorXd ground_truth,
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

    computeQueriesControl(S, A, B, matrix_to_optimize, Mx);

    DenseMatrix r = rhs - Mx;
    DenseMatrix p = r;
    VectorXd rsold = b_dot(r, r); 

    // TODO: restabilize the residual to avoid numerical error
    // after a certain number of iterations

    int cg_iter;
    for(cg_iter = 0; cg_iter < cg_max_iter; cg_iter++) {

        if(matrix_to_optimize == Amat) {
            computeQueriesControl(S, p, B, Amat, Mp);
        }
        else {
            computeQueriesControl(S, A, p, Bmat, Mp);
        }
        VectorXd bdot = b_dot(p, Mp);
        bdot.array() += nan_avoidance_constant; 
        VectorXd alpha = rsold.cwiseQuotient(bdot);

        if(matrix_to_optimize == Amat) {
            A += scale_rows(alpha, p);
        }
        else {
            B += scale_rows(alpha, p);
        }
        r -= scale_rows(alpha, Mp);

        VectorXd rsnew = b_dot(r, r); 
        double rsnew_norm_sqrt = sqrt(rsnew.sum());
        if(rsnew_norm_sqrt < cg_residual_tol) {
            break;
        }

        VectorXd coeffs = rsnew.cwiseQuotient(rsold);
        p = r + scale_rows(coeffs, p);
        rsold = rsnew;
    }
    if (cg_iter == cg_max_iter) {
        cout << "WARNING: Conjugate gradients did not converge to specified tolerance "
            << "in max iteration count." << endl;
    }
}
};




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
    test.initializeEmbeddings();
    //test.run_cg(5);


    int num_alternating_steps = 5;
    for(int i = 0; i < num_alternating_steps; i++) {
        test.cg_optimizer_control(test.S, test.ground_truth, test.A, test.B, Amat, 40);
        cout << "Residual: " << test.computeResidual() << endl;
        test.cg_optimizer_control(test.S, test.ground_truth, test.A, test.B, Bmat, 40);
        cout << "Residual: " << test.computeResidual() << endl;
    }

    //test_single_process_factorization(4, 4, 16);

    MPI_Finalize();
}