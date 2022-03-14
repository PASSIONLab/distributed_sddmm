#include "benchmark_dist.hpp"

#include <iostream>
#include <fstream>
#include <string>
#include "distributed_sparse.h"
#include "15D_dense_shift.hpp"
#include "15D_sparse_shift.hpp"
#include "25D_cannon_dense.hpp"
#include "25D_cannon_sparse.hpp"

#include "SpmatLocal.hpp"
#include "FlexibleGrid.hpp"

#include "sparse_kernels.h"
#include "common.h"
#include "als_conjugate_gradients.h"
#include "gat.hpp"
#include <mpi.h>
#include "json.hpp"

using json = nlohmann::json;

using namespace std;

void benchmark_algorithm(SpmatLocal* spmat, 
        string algorithm_name,
        string output_file,
        bool fused,
        int R,
        int c,
        string app 
        ) {

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    ofstream fout;
        fout.open(output_file, std::ios_base::app 
    );

    StandardKernel local_ops;
    Distributed_Sparse* d_ops;

    if(algorithm_name=="15d_fusion1") {
        d_ops = new Sparse15D_Dense_Shift(
            spmat, 
            R, 
            c, 
            1, 
            &local_ops);
    }
    else if(algorithm_name=="15d_sparse") {
        d_ops = new Sparse15D_Sparse_Shift(
            spmat, 
            R, 
            c, 
            &local_ops);
    }
    else if(algorithm_name=="15d_fusion2") {
        d_ops = new 
        Sparse15D_Dense_Shift(
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

    unique_ptr<GAT> gnn;
    unique_ptr<Distributed_ALS> d_als;
    vector<GATLayer> layers;

    if(app=="gat") {
        // Input features, features per head, output features
        layers.emplace_back(256, 256, 4);
        layers.emplace_back(1024, 256, 4);
        layers.emplace_back(1024, 256, 6);
        gnn.reset(new GAT(layers, d_ops)); 
    }
    else if(app=="als") {
        d_als.reset(new Distributed_ALS(d_ops, true));
    }
    else {
        assert(app=="vanilla");
    }

    DenseMatrix A = d_ops->like_A_matrix(0.001);    
    DenseMatrix B = d_ops->like_B_matrix(0.001);

    VectorXd S = d_ops->like_S_values(1.0); 
    VectorXd sddmm_result = d_ops->like_S_values(0.0);

    if(rank == 0) {
        std::cout << "Starting benchmark " << app << endl;
    }

    d_ops->reset_performance_timers();
    my_timer_t t = start_clock();
    int num_trials = 0;

    double application_communication_time = 0.0;
    do {
        num_trials++;

        if(app == "vanilla") {
            if(fused) {
                d_ops->fusedSpMM(A, 
                        B, 
                        S, 
                        sddmm_result, 
                        Amat);
            }
            else {
                d_ops->sddmmA(A, B, S, sddmm_result);
                d_ops->spmmA(A, B, S);
            } 
        }
        else if(app=="gat") {
            gnn->forwardPass();
        }
        else if(app=="als") {
            d_als->application_communication_time = 0.0;
            d_als->run_cg(1);
            application_communication_time = d_als->application_communication_time;
        }
    } while(num_trials < 5);
    MPI_Barrier(MPI_COMM_WORLD);

    json j_obj; 

    double elapsed = stop_clock_get_elapsed(t);
    double ops = 2 * spmat->dist_nnz * 2 * R * num_trials;
    double throughput = ops / elapsed;
    throughput /= 1e9;

    j_obj["elapsed"] = elapsed;
    j_obj["overall_throughput"] = throughput;
    j_obj["fused"] = fused;
    j_obj["num_trials"] = num_trials;
    j_obj["alg_name"] = algorithm_name;
    j_obj["alg_info"] = d_ops->json_algorithm_info();
    j_obj["application_communication_time"] = application_communication_time; 
    j_obj["perf_stats"] = d_ops->json_perf_statistics();

    if(rank == 0) {
        fout << j_obj.dump(4) << "," << endl;
    } 

    fout.close();

    delete d_ops;
}
