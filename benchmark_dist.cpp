#include <iostream>
#include <string>
#include "benchmark_dist.h"
#include "distributed_sparse.h"
#include "15D_mdense_shift_striped.hpp"
#include "25D_cannon_dense.hpp"
#include "25D_cannon_sparse.hpp"

#include "SpmatLocal.hpp"
#include "FlexibleGrid.hpp"

#include "sparse_kernels.h"
#include "common.h"
#include "als_conjugate_gradients.h"
#include <mpi.h>

using namespace std;

void benchmark_algorithm(SpmatLocal* spmat, 
        string algorithm_name,
        bool fused,
        int R,
        int c 
        ) {

    StandardKernel local_ops;
    Distributed_Sparse* d_ops;

    if(algorithm_name=="15d_fusion1") {
        d_ops = new Sparse15D_MDense_Shift_Striped(
            spmat, 
            R, 
            c, 
            1, 
            &local_ops);
    }
    else if(algorithm_name=="15d_fusion2") {
        d_ops = new Sparse15D_MDense_Shift_Striped(
            spmat, 
            R,
            c, 
            2, 
            &local_ops); 
    }
    else if(algorithm_name=="25d_dense_replicate") {
        d_ops = new Sparse25D_Cannon_Dense(
            spmat,
            R, 
            c, 
            &local_ops);
    }
    else if(algorithm_name=="25d_sparse_replicate") {
        d_ops = new Sparse25D_Cannon_Sparse(
            spmat,
            R, 
            c, 
            &local_ops);
    }

    DenseMatrix A = d_ops->like_A_matrix(0.0);    
    DenseMatrix B = d_ops->like_B_matrix(0.0);

    VectorXd S = d_ops->like_S_values(1.0); 
    VectorXd sddmm_result = d_ops->like_S_values(0.0);
    DenseMatrix fused_result = d_ops->like_A_matrix(0.0); 

    d_ops->reset_performance_timers();
    my_timer_t t = start_clock();
    int num_trials = 0;
    do {
        num_trials++;

        if(fused) {
            d_ops->fusedSpMM(A, 
                    B, 
                    S, 
                    sddmm_result, 
                    fused_result, 
                    Amat);
        }
        else {
            d_ops->sddmmA(A, B, S, sddmm_result);
            d_ops->spmmA(A, B, S);
        } 

    } while(num_trials < 5);
    MPI_Barrier(MPI_COMM_WORLD);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(rank == 0) {
        double elapsed = stop_clock_get_elapsed(t);

        double ops = 2 * spmat->dist_nnz * 2 * R * num_trials;
        double throughput = ops / elapsed;
        throughput /= 1e9;

        cout << "*************************************" << endl;

        if(fused) {
            cout << "Fused Overall Throughput: " 
                    << throughput << " GFLOPs" << endl;
        }
        else {
            cout << "Unfused Overall Throughput: " 
                    << throughput << " GFLOPs" << endl; 
        }
        cout << "Total Time: " << elapsed << "s" << endl;
        cout << "# of Trials: " << num_trials << endl;

        cout << "Breakdown: " << endl;
 
    }
    d_ops->print_performance_statistics();

    if(rank == 0) {
        cout << "*************************************" << endl;
    }

    delete d_ops;
}

