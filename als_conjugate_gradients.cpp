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
    double cg_residual_tol = 1e-5;
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
    computeQueries(matrix_to_optimize, Mx);

    DenseMatrix r = rhs - Mx;
    DenseMatrix p = r;
    VectorXd rsold = batch_dot_product(r, r); 

    MPI_Comm reduction_world;
    if(matrix_to_optimize == Amat) {
	reduction_world = A_R_split_world;
    }
    else {
	reduction_world = B_R_split_world;	
    }

    MPI_Allreduce(MPI_IN_PLACE, rsold.data(), rsold.size(),
	MPI_DOUBLE, MPI_SUM, reduction_world);

    // TODO: restabilize the residual to avoid numerical error
    // after a certain number of iterations
    int cg_iter;
    for(cg_iter = 0; cg_iter < cg_max_iter; cg_iter++) {
	computeQueries(matrix_to_optimize, Mp);

        VectorXd bdot = batch_dot_product(p, Mp);
	MPI_Allreduce(MPI_IN_PLACE, bdot.data(), bdot.size(),
                MPI_DOUBLE, MPI_SUM, reduction_world);

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
	MPI_Allreduce(MPI_IN_PLACE, rsnew.data(), rsnew.size(),
		MPI_DOUBLE, MPI_SUM, reduction_world);

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

void ALS_CG::run_cg(int n_alternating_steps) {
    initializeEmbeddings();
    cout << "Embeddings Initialized!" << endl;
    cout << "Initial Residual: " << computeResidual() << endl;

    for(int i = 0; i < n_alternating_steps; i++) {
        cg_optimizer(Amat, 40);
        cout << "Residual: " << computeResidual() << endl;
        cg_optimizer(Bmat, 40);
        cout << "Residual: " << computeResidual() << endl;
    }
}
