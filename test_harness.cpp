//#include "15D_mdense_bcast.hpp"
//#include "15D_mdense_shift.hpp"

//#include "15D_mdense_shift_striped.hpp"
//#include "2D_cannon.hpp"
//#include "25D_cannon_dense.hpp"
#include "25D_cannon_sparse.hpp"

#include "SpmatLocal.hpp"
#include "FlexibleGrid.hpp"

#include "sparse_kernels.h"
#include "common.h"
#include "als_conjugate_gradients.h"

using namespace std;

/*void test_fusion(Sparse15D_MDense_Shift_Striped* d_ops) {
    int proc_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

    if(proc_rank == 0) {
        cout << "Testing fused SDDMM / SpMM Correctness" << endl;
    }

    DenseMatrix A = d_ops->like_A_matrix(0.0);    
    DenseMatrix B = d_ops->like_B_matrix(0.0);    

    //deterministic_initialize(A);
    //deterministic_initialize(B);

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
    sqnorm = Sbuffer.squaredNorm(); 
    MPI_Allreduce(MPI_IN_PLACE, &sqnorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if(proc_rank == 0) {
        cout << "S Result Buffer: " << sqnorm << endl;
    }

    sqnorm = STbuffer.squaredNorm(); 
    MPI_Allreduce(MPI_IN_PLACE, &sqnorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if(proc_rank == 0) {
        cout << "ST Result Buffer: " << sqnorm << endl;
    } 

    sqnorm = standard_result.squaredNorm(); 
    MPI_Allreduce(MPI_IN_PLACE, &sqnorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if(proc_rank == 0) {
        cout << "Standard Result Buffer: " << sqnorm << endl;
    }

    if(proc_rank == 0) {
        cout << "Fusion testing complete!" << endl;
    }
}*/

class ZeroProcess : public NonzeroDistribution {
public:
    ZeroProcess(int M, int N) { 
        rows_in_block = M; 
        cols_in_block = N;
    }

	int blockOwner(int row_block, int col_block) {
        return 0; 
    }
};

void verify_operation(SpmatLocal &spmat, Distributed_Sparse* d_ops) {
    int proc_rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    DenseMatrix A = d_ops->like_A_matrix(0.0);    
    DenseMatrix B = d_ops->like_B_matrix(0.0);

    VectorXd S = d_ops->like_S_values(1.0);
    VectorXd ST = d_ops->like_ST_values(1.0);

    d_ops->dummyInitialize(A, Amat);
    d_ops->dummyInitialize(B, Bmat);

    //d_ops->print_nonzero_distribution(A, B);

    VectorXd result = d_ops->like_S_values(0.0);

    d_ops->initial_synchronize(&A, &B, nullptr);

    d_ops->sddmm(A, B, S, result);

    double sddmm_fingerprint = result.squaredNorm(); 
    MPI_Allreduce(MPI_IN_PLACE, &sddmm_fingerprint, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    d_ops->spmmA(A, B, S);

    double spmmA_fingerprint = A.squaredNorm();
    MPI_Allreduce(MPI_IN_PLACE, &spmmA_fingerprint, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    d_ops->spmmB(A, B, ST);

    double spmmB_fingerprint = B.squaredNorm(); 
    MPI_Allreduce(MPI_IN_PLACE, &spmmB_fingerprint, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    //d_ops->print_nonzero_distribution(A, B); 

    if(proc_rank == 0) {
        cout << "SDDMM Fingerprint: " << sddmm_fingerprint << endl;
        cout << "SpMMA Fingerprint: " << spmmA_fingerprint << endl;
        cout << "SpMMB Fingerprint: " << spmmB_fingerprint << endl; 
    }

    // The most reliable strategy: Every processor does the computation
    // locally and compare the dense matrix results (ideally, the SDDMM
    // is folded in and should also be correct). 

    /*ZeroProcess dist(S.M, S.N);
	SpmatLocal* gathered = redistribute_nonzeros(&dist, false, false) {

    DenseMatrix sp_dense = new DenseMatrix::Constant(S.M, S.N, 0.0);

    for(int i = 0; i < gathered->coords.size(); i++) {
        sp_dense(coords[i].r, coords[i].c) = 1.0;
    }

    delete gathered;
    */
}

/*void test_sparse_transpose() {
    int proc_rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    string fname = "../data/testmat.mtx";

    SpmatLocal x;

    StandardKernel local_ops;
    Sparse15D_MDense_Shift_Striped d_ops(fname, 8, 1, &local_ops, true, false);

    for(int i = 0; i < num_procs; i++) {
        if(proc_rank == i) {
            cout << "Process " << i << ":" << endl;
            cout << "Rank in Fiber: " << d_ops.rankInFiber << endl;
            cout << "Rank in Layer: " << d_ops.rankInLayer << endl;

            for(int j = 0; j < d_ops.ST->coords.size(); j++) {
                cout << d_ops.ST->coords[j].string_rep() << endl;
            }
            cout << "==================" << endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}*/

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    initialize_mpi_datatypes();

    /*{
        FlexibleGrid grid(4, 3, 2, 3);
        grid.self_test();
    }*/

    string fname(argv[1]);

    StandardKernel local_ops;
    SpmatLocal S;
    S.loadTuples(true, -1, -1, fname);
 
    /*Sparse15D_MDense_Shift_Striped* d_ops 
        = new Sparse15D_MDense_Shift_Striped(
                atoi(argv[1]), 
                atoi(argv[2]), 
                atoi(argv[3]), 
                atoi(argv[4]), 
                &local_ops, 
                true,  // Whether we should support fusing SDDMM / SpMM
                true   // Whether we should auto-fuse the provided operation, or rely on
                );     // the backend local operation to do it for us  
    */


    Sparse25D_Cannon_Sparse* d_ops
        = new Sparse25D_Cannon_Sparse(
            &S,
            atoi(argv[2]),
            atoi(argv[3]),
            &local_ops
        );

    verify_operation(S, d_ops);

    //Sparse25D_MDense_Nostage* d_ops = new Sparse25D_MDense_Nostage(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), &local_ops);

    //srand((unsigned int) time(0) + d_ops->proc_rank + 2);
    //test_fusion(d_ops);

    //test_15D(d_ops);

    /*
    Distributed_ALS* x = new Distributed_ALS(d_ops, MPI_COMM_WORLD, true);
    d_ops->reset_performance_timers();
    x->run_cg(5);
    d_ops->print_performance_statistics(); 
    */

    //delete d_ops;

    MPI_Finalize();
}