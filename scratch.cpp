#include "15D_mdense_shift_striped.hpp"
#include "25D_cannon_dense.hpp"
#include "25D_cannon_sparse.hpp"
#include "SpmatLocal.hpp"
#include <string>
#include "benchmark_dist.h"

using namespace std;

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
    VectorXd result = d_ops->like_S_values(0.0);

    d_ops->dummyInitialize(A, Amat);
    d_ops->dummyInitialize(B, Bmat);
    //d_ops->initial_synchronize(&A, nullptr, nullptr);

    d_ops->initial_synchronize(&A, &B, nullptr);

    //d_ops->print_nonzero_distribution(A, B);
    d_ops->sddmmA(A, B, S, result);

    double A_fingerprint = A.squaredNorm();
    MPI_Allreduce(MPI_IN_PLACE, &A_fingerprint, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double sddmm_fingerprint = result.squaredNorm(); 
    MPI_Allreduce(MPI_IN_PLACE, &sddmm_fingerprint, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    d_ops->dummyInitialize(A, Amat);
    d_ops->dummyInitialize(B, Bmat);
    d_ops->initial_synchronize(&A, nullptr, nullptr);
    //d_ops->spmmA(A, B, S);

    double spmmA_fingerprint = A.squaredNorm();
    MPI_Allreduce(MPI_IN_PLACE, &spmmA_fingerprint, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    d_ops->dummyInitialize(A, Amat);
    d_ops->dummyInitialize(B, Bmat);

    d_ops->initial_synchronize(nullptr, &B, nullptr);
    //d_ops->spmmB(A, B, ST);

    double spmmB_fingerprint = B.squaredNorm(); 
    MPI_Allreduce(MPI_IN_PLACE, &spmmB_fingerprint, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    DenseMatrix fused_result = d_ops->like_A_matrix(0.0);

    //d_ops->print_nonzero_distribution(A, B); 

    if(proc_rank == 0) {
        cout << "A Fingerprint: " << A_fingerprint << endl;
        cout << "SDDMM Fingerprint: " << sddmm_fingerprint << endl;
        cout << "SpMMA Fingerprint: " << spmmA_fingerprint << endl;
        cout << "SpMMB Fingerprint: " << spmmB_fingerprint << endl; 
    } 
}

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
    //S.loadTuples(false, 18, 30, fname);
    S.loadTuples(true, -1, -1, fname);

    /*Sparse25D_Cannon_Dense* d_ops
        = new Sparse25D_Cannon_Dense(
            &S,
            atoi(argv[2]),
            atoi(argv[3]),
            &local_ops
        );*/

    /*Sparse15D_MDense_Shift_Striped* d_ops =
            new Sparse15D_MDense_Shift_Striped(&S, 
                atoi(argv[2]), 
                atoi(argv[3]), 
                1, 
                &local_ops);*/

    Sparse25D_Cannon_Sparse* d_ops
        = new Sparse25D_Cannon_Sparse(
            &S,
            atoi(argv[2]),
            atoi(argv[3]),
            &local_ops
        );

    //cout << "Initialization complete from " << d_ops->proc_rank << endl;

    verify_operation(S, d_ops);

    //Sparse25D_MDense_Nostage* d_ops = new Sparse25D_MDense_Nostage(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), &local_ops);

    //srand((unsigned int) time(0) + d_ops->proc_rank + 2);
    //test_fusion(d_ops);

    //test_15D(d_ops);

    /*for(int R = 32; R < 256; R += 32) {
        Sparse25D_Cannon_Sparse* d_ops
            = new Sparse25D_Cannon_Sparse(
                &S,
                R,
                atoi(argv[3]),
                &local_ops
            );
        /*Sparse25D_Cannon_Dense* d_ops
            = new Sparse25D_Cannon_Dense(
                &S,
                R, 
                atoi(argv[3]),
                &local_ops
            );*/

    /*
        if(d_ops->proc_rank == 0) {
            cout << "Created d_ops..." << endl;
        }
        Distributed_ALS* x = new Distributed_ALS(d_ops, MPI_COMM_WORLD, true);
        d_ops->reset_performance_timers();
        x->run_cg(5);
        d_ops->print_performance_statistics(); 
        delete x;
        delete d_ops;
    }
    */

    MPI_Finalize();
}