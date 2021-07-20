//#include "15D_mdense_bcast.hpp"
//#include "15D_mdense_shift.hpp"
#include "15D_mdense_shift_striped.hpp"
#include "SpmatLocal.hpp"
//#include "25D_mdense_nostage.hpp"

#include "sparse_kernels.h"
#include "common.h"
#include "als_conjugate_gradients.h"

using namespace std;

void deterministic_initialize(DenseMatrix &X) {
    for(int i = 0; i < X.rows(); i++) {
        for(int j = 0; j < X.cols(); j++) {
            X(i, j) = ((float) i + j) / X.size();
        }
    }
}

void test_fusion(Sparse15D_MDense_Shift_Striped* d_ops) {
    int proc_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

    if(proc_rank == 0) {
        cout << "Testing fused SDDMM / SpMM Correctness" << endl;
    }

    DenseMatrix A = d_ops->like_A_matrix(0.0);    
    DenseMatrix B = d_ops->like_B_matrix(0.0);    

    deterministic_initialize(A);
    deterministic_initialize(B);

    DenseMatrix dummy_resultA = d_ops->like_A_matrix(0.0);    
    DenseMatrix dummy_resultB = d_ops->like_B_matrix(0.0);    

    VectorXd Svalues = d_ops->like_S_values(1.0);
    VectorXd STvalues = d_ops->like_ST_values(1.0);

    VectorXd Sbuffer  = d_ops->like_S_values(0.0);
    VectorXd STbuffer = d_ops->like_ST_values(0.0);
    VectorXd standard_result = d_ops->like_S_values(0.0);

    d_ops->fusedSpMM(A, B, Svalues, Sbuffer, dummy_resultA, Amat);
    d_ops->fusedSpMM(A, B, STvalues, STbuffer, dummy_resultB, Bmat);
    d_ops->sddmm(A, B, Svalues, standard_result); 

    double sqnorm; 
    sqnorm = Sbuffer.norm(); 
    MPI_Allreduce(MPI_IN_PLACE, &sqnorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if(proc_rank == 0) {
        cout << "S Result Buffer: " << sqnorm << endl;
    }

    sqnorm = STbuffer.norm(); 
    MPI_Allreduce(MPI_IN_PLACE, &sqnorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if(proc_rank == 0) {
        cout << "ST Result Buffer: " << sqnorm << endl;
    }

    sqnorm = standard_result.norm(); 
    MPI_Allreduce(MPI_IN_PLACE, &sqnorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if(proc_rank == 0) {
        cout << "Standard Result Buffer: " << sqnorm << endl;
    }

    if(proc_rank == 0) {
        cout << "Fusion testing complete!" << endl;
    }
}

void test_sparse_transpose() {
    SpmatLocal x;
	x.loadMatrixAndRedistribute("../data/testmat.mtx", nullptr);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    initialize_mpi_datatypes();

    test_sparse_transpose();

    //string fname(argv[1]);

    //StandardKernel local_ops;
    //FusedStandardKernel fused_local_ops;

    //Sparse15D_MDense_Bcast* d_ops = new Sparse15D_MDense_Bcast(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), &local_ops);
    //Sparse15D_MDense_Shift* d_ops = new Sparse15D_MDense_Shift(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), &local_ops);

    /*Sparse15D_MDense_Shift_Striped* d_ops 
        = new Sparse15D_MDense_Shift_Striped(
                atoi(argv[1]), 
                atoi(argv[2]), 
                atoi(argv[3]), 
                atoi(argv[4]), 
                &local_ops, 
                true,  // Whether we should support fusing SDDMM / SpMM
                false  // Whether we should auto-fuse the provided operation, or rely on
                );     // the backend local operation to do it for us
    */

    /*
    Sparse15D_MDense_Shift_Striped* d_ops 
        = new Sparse15D_MDense_Shift_Striped(
                fname, 
                atoi(argv[2]), 
                atoi(argv[3]), 
                &local_ops, 
                false, // Whether we should support fusing SDDMM / SpMM
                false  // Whether we should auto-fuse the provided operation, or rely on
                );     // the backend local operation to do it for us 

    */

    //Sparse25D_MDense_Nostage* d_ops = new Sparse25D_MDense_Nostage(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), &local_ops);

    //srand((unsigned int) time(0) + d_ops->proc_rank + 2);
    //test_fusion(d_ops);

    //Distributed_ALS* x = new Distributed_ALS(d_ops, d_ops->grid->GetLayerWorld(), true);

    //d_ops->reset_performance_timers();
    //x->run_cg(5);
    //d_ops->print_performance_statistics();

    MPI_Finalize();
}