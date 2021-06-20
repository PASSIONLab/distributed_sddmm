#ifndef DISTRIBUTED_SPARSE_H
#define DISTRIBUTED_SPARSE_H

#include <Eigen/Dense>
#include <vector>
#include <map>
#include <cassert>

#include "sparse_kernels.h"
#include "common.h"
#include <mpi.h>

using namespace std;
using namespace Eigen;

class Distributed_Sparse {
public:
    int proc_rank;     // Global process rank
    int p;             // Total # of processes

    // General algorithm information
    string algorithm_name;
    vector<string> proc_grid_names;
    vector<int> proc_grid_dimensions;

    // Related to performance counting
    vector<string>      perf_counter_keys;
    map<string, int>    call_count; 
    map<string, double> total_time; 

    // Matrix Dimensions, R is the short inner dimension
    int64_t M, N, R;

    // Local dimensions of the dense matrices
    int localArows, localAcols, localBrows, localBcols;

    // Related to the sparse matrix
    spmat_local_t S;
    VectorXd input_Svalues; // The values that sparse matrix S came with;
                            // when reading a file, this value is filled.
    int nnz_per_row;

    // Pointer to object implementing the local SDDMM / SPMM Operations 
    KernelImplementation *kernel;
 
    //MPI_Comm A_row_world, B_row_world;
    //MPI_Comm A_col_world, B_col_world;
    //MPI_Comm A_replication_world, B_replication_world;

    MPI_Comm A_R_split_world, B_R_split_world;

    bool verbose;

    /*
     * Some boilerplate, but also forces subclasses to initialize what they need to 
     */
    Distributed_Sparse(KernelImplementation* k) {
        MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &p);
        verbose = false;

        kernel = k;

        // Dummy initializations
        algorithm_name = "";
        M = -1;
        N = -1;
        R = -1;
        localArows = -1;
        localAcols = -1;
        localBrows = -1;
        localBcols = -1;

        // TODO: Need to dummy-initialize the MPI constructors. 
    }

    void check_initialized() {
        assert(algorithm_name != "");

        assert(proc_grid_names.size() > 0);
        assert(proc_grid_dimensions.size() > 0);
        assert(perf_counter_keys.size() > 0);

        assert(M != -1 && N != -1 && R != -1);
        assert(localAcols != -1 && localBcols != -1);
        assert(localArows != -1 && localBrows != -1);

        // TODO: Remove this statement
        cout << "Algorith initialized successfully!" << endl;
    }

    void print_algorithm_info() { 
        cout << algorithm_name << endl;
        cout << "Matrix Dimensions: " 
        << this->M << " x " << this->N << endl;
        cout << "R-Value: " << this->R << endl;
        cout << "Grid Dimensions : ";
        for(int i = 0; i < proc_grid_dimensions.size(); i++) {
            cout << proc_grid_dimensions[i];
            if (i != proc_grid_dimensions.size()) {
                cout << " x ";
            }
        }
        cout << endl;

        cout << "Grid Dimension Interpretations: ";        
        for(int i = 0; i < proc_grid_names.size(); i++) {
            cout << proc_grid_names[i];
            if (i != proc_grid_names.size()) {
                cout << " x ";
            }
        }
        cout << endl;
    }

    void setVerbose(bool value) {
        verbose = value;
    }

    VectorXd like_S_values(double value) {
        return VectorXd::Constant(S.local_nnz, value); 
    }

    DenseMatrix like_A_matrix(double value) {
        return DenseMatrix::Constant(localArows, localAcols, value);  
    }

    DenseMatrix like_B_matrix(double value) {
        return DenseMatrix::Constant(localBrows, localBcols, value);  
    }

    void reset_performance_timers() {
        for(auto it = perf_counter_keys.begin(); it != perf_counter_keys.end(); it++) {
            call_count[*it] = 0;
            total_time[*it] = 0.0;
        }
    }

    void stop_clock_and_add(my_timer_t &start, string counter_name) {
        if(find(perf_counter_keys.begin(), perf_counter_keys.end(), counter_name) 
            != perf_counter_keys.end()) {
            call_count[counter_name]++;
            total_time[counter_name] += stop_clock_get_elapsed(start); 
        }
        else {
            cout    << "Error, performance counter " 
                    << counter_name << " not registered." << endl;
            exit(1);
        }
    } 

    void print_performance_statistics() {
        // This is going to assume that all timing starts and ends with a barrier, so that
        // all processors enter and leave the call at the same time. Also, I'm taking an
        // average over several calls by all processors; might want to compute the
        // variance as well. 
        if(proc_rank == 0) {
            cout << endl;
            cout << "================================" << endl;
            cout << "==== Performance Statistics ====" << endl;
            cout << "================================" << endl;
        } 

        for(auto it = perf_counter_keys.begin(); it != perf_counter_keys.end(); it++) {
            double avg_time = total_time[*it]; 

            MPI_Allreduce(MPI_IN_PLACE, &avg_time, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            avg_time /= call_count[*it] * p;

            if(proc_rank == 0) {
                cout << "Avg. " << *it << ":\t" << avg_time << "s" << endl;
            }
        }
        if(proc_rank == 0) {
            cout << "=================================" << endl;
        } 
    }

    virtual void initial_synchronize(DenseMatrix *localA, DenseMatrix *localB, VectorXd *localS) = 0;
    virtual void spmmA(DenseMatrix &localA, DenseMatrix &localB, VectorXd &SValues) = 0;
    virtual void spmmB(DenseMatrix &localA, DenseMatrix &localB, VectorXd &SValues) = 0;
    virtual void sddmm(DenseMatrix &localA, DenseMatrix &localB, VectorXd &SValues, VectorXd &sddmm_result) = 0;
};

#endif