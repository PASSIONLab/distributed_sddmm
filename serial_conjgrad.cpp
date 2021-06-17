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
public:
    // This constructor tests with a random matrix. 
    spmat_local_t S;
    VectorXd ground_truth;
    int R;

    StandardKernel spOps;

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

        spOps.sddmm_local(S,
                    ones,
                    Agt,
                    Bgt,
                    ground_truth,
                    0, 
                    S.local_nnz);
    }
  
    void computeRHS(MatMode matrix_to_optimize,
                            DenseMatrix &rhs) {
        if(matrix_to_optimize == Amat) {
            spOps.spmm_local(S, ground_truth, rhs, B, Amat, 0, S.local_nnz);
        }
        else {
            spOps.spmm_local(S, ground_truth, A, rhs, Bmat, 0, S.local_nnz);
        }
    }

    void initializeEmbeddings() {
        new (&A) DenseMatrix(S.nrows, R);
        new (&B) DenseMatrix(S.ncols, R);
        initialize_dense_matrix(A);
        initialize_dense_matrix(B);
    }

    ~SingleNodeALS() {
        // Empty destructor
    }

    double computeResidual() {
        VectorXd ones = VectorXd::Constant(S.local_nnz, 1.0);
        VectorXd sddmm_result = VectorXd::Zero(S.local_nnz);

        spOps.sddmm_local(S,
                    ones,
                    A,
                    B,
                    sddmm_result,
                    0, 
                    S.local_nnz);
        
        return (sddmm_result - ground_truth).norm();
    }

    void computeQueries(
                        DenseMatrix &A,
                        DenseMatrix &B,
                        MatMode matrix_to_optimize,
                        DenseMatrix &result) {

        double lambda = 1e-7;

        VectorXd ones = VectorXd::Constant(S.local_nnz, 1.0);
        result.setZero();
        VectorXd sddmm_result = VectorXd::Zero(S.local_nnz);

        spOps.sddmm_local(S, 
                    ones,
                    A,
                    B,
                    sddmm_result,
                    0, 
                    S.local_nnz);

        if(matrix_to_optimize == Amat) {
            spOps.spmm_local(S, sddmm_result, result, B, Amat, 0, S.local_nnz);
            result += lambda * A;
        }
        else if(matrix_to_optimize == Bmat) {
            spOps.spmm_local(S, sddmm_result, A, result, Bmat, 0, S.local_nnz);
            result += lambda * B;
        }
    }
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    SingleNodeALS test(12, 8, 16);
    test.run_cg(20);

    MPI_Finalize();
}