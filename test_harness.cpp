#include "15D_sparse.hpp"
#include "25D_sparse.hpp"
#include "sparse_kernels.h"
#include "common.h"
#include "als_conjugate_gradients.h"

using namespace std;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    string fname(argv[1]);

    StandardKernel local_ops;
    Sparse25D* d_ops = new Sparse25D(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), &local_ops);

    srand((unsigned int) time(0) + d_ops->proc_rank);

    DenseMatrix A = d_ops->like_A_matrix(1.0);
    DenseMatrix B = d_ops->like_B_matrix(1.0);

    VectorXd ones = d_ops->like_S_values(1.0);
    VectorXd sddmm_result = d_ops->like_S_values(0.0);

    d_ops->sddmm(A, B, ones, sddmm_result);


    Distributed_ALS* x = new Distributed_ALS(d_ops, d_ops->grid->GetLayerWorld(), true);
    d_ops->reset_performance_timers();
    x->run_cg(5);
    d_ops->print_performance_statistics();

    MPI_Finalize();
}