#ifndef DISTRIBUTED_SPARSE_H
#define DISTRIBUTED_SPARSE_H

#include <Eigen/Dense>
#include <vector>
#include <map>
#include <cassert>

#include <mpi.h>
#include "sparse_kernels.h"
#include "common.h"
#include "SpmatLocal.hpp"
#include "FlexibleGrid.hpp"

using namespace std;
using namespace Eigen;

class DenseSubmatrix {
public:
    int topRow, leftCol, rowCount, colCount;

    DenseSubmatrix(int tR, int lC, int rC, int cC) {
        topRow = tR;
        leftCol = lC;
        rowCount = rC;
        colCount = cC;
    }
};

class Distributed_Sparse {
public:
    int proc_rank;     // Global process rank
    int p;             // Total # of processes

    // General algorithm information
    string algorithm_name;
    vector<string> proc_grid_names;

    // Related to performance counting
    vector<string>      perf_counter_keys;
    map<string, int>    call_count; 
    map<string, double> total_time; 

    // Matrix Dimensions, R is the short inner dimension
    int64_t M, N, R;

    // Local dimensions of the dense matrix buffers (which may contain multiple submatrices) 
    int localArows, localAcols, localBrows, localBcols;

    // Gives the exact row and column indices of dense matrix submatrices
    // stored by this implementation.
    vector<DenseSubmatrix> aSubmatrices;
    vector<DenseSubmatrix> bSubmatrices;

    // Related to the sparse matrix
    unique_ptr<SpmatLocal> S;
    unique_ptr<SpmatLocal> ST;
    shared_ptr<FlexibleGrid> grid;

    int superclass_constructor_sentinel;

    // Pointer to object implementing the local SDDMM / SPMM Operations 
    KernelImplementation *kernel;
 
    bool r_split;
    MPI_Comm A_R_split_world, B_R_split_world;
    //MPI_Comm A_row_world, B_row_world;
    //MPI_Comm A_col_world, B_col_world;
    //MPI_Comm A_replication_world, B_replication_world;

    bool verbose;

    string debug_msg;

    /*
     * Some boilerplate, but also forces subclasses to initialize what they need to 
     */
    Distributed_Sparse(KernelImplementation* k, int R) {
        MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &p);
        verbose = false; 

        kernel = k;
        this->R = R;

        // Dummy initializations
        algorithm_name = "";
        M = -1;
        N = -1;
        localArows = -1;
        localAcols = -1;
        localBrows = -1;
        localBcols = -1;

        superclass_constructor_sentinel = 3;
        // TODO: Need to dummy-initialize the MPI constructors. 
    }

    void check_initialized() {
        assert(algorithm_name != "");

        assert(proc_grid_names.size() > 0);
        assert(perf_counter_keys.size() > 0);

        assert(M != -1 && N != -1 && R != -1);
        assert(localAcols != -1 && localBcols != -1);
        assert(localArows != -1 && localBrows != -1);


        assert(superclass_constructor_sentinel == 3);
        assert(aSubmatrices.size() > 0);
        assert(bSubmatrices.size() > 0);
        assert(S->initialized);
        assert(ST->initialized);

        assert(S->coordinate_ownership_initialized);
        assert(ST->coordinate_ownership_initialized);

        assert(S->blockStarts.size() > 0);
        assert(ST->blockStarts.size() > 0);

        assert(S->csr_initialized);
        assert(ST->csr_initialized);

    }

    void print_algorithm_info() {
        cout << algorithm_name << endl;
        cout << "Sparse Matrix Dimensions: " 
                << this->M << " x " << this->N << endl;
        cout << "R-Value: " << this->R << endl;

        cout << "Grid Adjacency Mode: " << grid->adjacency << endl; 
        cout << "Grid Dimension Interpretations: ";        
        for(int i = 0; i < proc_grid_names.size(); i++) {
            cout << proc_grid_names[i];
            if (i != proc_grid_names.size() - 1) {
                cout << " x ";
            }
        } 
        cout << endl;

        cout << "Grid Dimensions : ";
        for(int i = 0; i < proc_grid_names.size(); i++) {
            cout << grid->dim_list[i];
            if (i != proc_grid_names.size() - 1) {
                cout << " x ";
            }
        }
        cout << endl;
        cout << "================================" << endl;
    }

    void setVerbose(bool value) {
        verbose = value;
    }

    virtual VectorXd like_S_values(double value) {
        return VectorXd::Constant(S->owned_coords_end - S->owned_coords_start, value); 
    }

    virtual VectorXd like_ST_values(double value) {
        return VectorXd::Constant(ST->owned_coords_end - ST->owned_coords_start, value); 
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
            print_algorithm_info();
        }

        for(auto it = perf_counter_keys.begin(); it != perf_counter_keys.end(); it++) {
            double val = total_time[*it]; 

            MPI_Allreduce(MPI_IN_PLACE, &val, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            // We also have the call count for each statistic timed
            val /= p;

            if(proc_rank == 0) {
                cout << "Total " << *it << ":\t" << val << "s" << endl;
            }
        }
        if(proc_rank == 0) {
            cout << "=================================" << endl;
        } 
    }

    /*
     * If the input buffers need to be shifted / preprocessed 
     */
    virtual void initial_synchronize(DenseMatrix *localA, DenseMatrix *localB, VectorXd *localS) = 0;

    /*
     * The five functions below are just convenience functions. 
     */

    void spmmA(DenseMatrix &localA, DenseMatrix &localB, VectorXd &SValues) {
        algorithm(localA, localB, SValues, nullptr, k_spmmA, true);
    }

    void spmmB(DenseMatrix &localA, DenseMatrix &localB, VectorXd &SValues) {
        algorithm(localA, localB, SValues, nullptr, k_spmmB, true);
    }

    void sddmmA(DenseMatrix &localA, DenseMatrix &localB, VectorXd &SValues, VectorXd &sddmm_result) { 
        algorithm(localA, localB, SValues, &sddmm_result, k_sddmmA, true);
    }

    void sddmmB(DenseMatrix &localA, DenseMatrix &localB, VectorXd &SValues, VectorXd &sddmm_result) { 
        algorithm(localA, localB, SValues, &sddmm_result, k_sddmmB, true);
    }

    /*
     * The user is responsible for any initial and
     * final shifts under this strategy (fairness in benchmarking) 
     */
    virtual void fusedSpMM(DenseMatrix &localA, 
            DenseMatrix &localB, 
            VectorXd &Svalues, 
            VectorXd &sddmm_buffer, 
            DenseMatrix &result, 
            MatMode mode) {

        if(mode == Amat) {
            algorithm(localA, localB, Svalues, &sddmm_buffer, k_sddmmA, true);
            algorithm(localA, localB, sddmm_buffer, nullptr, k_spmmA, false);
        }
        else if(mode == Bmat) {
            algorithm(localA, localB, Svalues, &sddmm_buffer, k_sddmmB, true);
            algorithm(localA, localB, sddmm_buffer, nullptr, k_spmmB, false);
        }
    }

    virtual void algorithm( DenseMatrix &localA, 
                            DenseMatrix &localB, 
                            VectorXd &SValues, 
                            VectorXd *sddmm_result_ptr, 
                            KernelMode mode,
                            bool initial_replicate 
                            ) = 0;

    void dummyInitialize(DenseMatrix &loc, MatMode mode) {
        // Assume that the dense matrices will always be stored in row major format. 

        vector<DenseSubmatrix> &submatrices = (mode == Amat) ? aSubmatrices : bSubmatrices;

        double* ptr = loc.data();

        for(int t = 0; t < submatrices.size(); t++) {
            int topRow = submatrices[t].topRow;
            int leftCol = submatrices[t].leftCol;

            for(int i = 0; i < loc.rows(); i++) {
                for(int j = 0; j < loc.cols(); j++) {
                    *ptr = (topRow + i) * R + leftCol + j;
                    ptr++;
                }
            }
        } 
    }

    /*
     * Convenience functions. 
     */

    void shiftDenseMatrix(DenseMatrix &mat, MPI_Comm world, int send_dst, int tag) {
        MPI_Status stat;
        DenseMatrix recvBuffer(mat.rows(), mat.cols());

        MPI_Sendrecv(mat.data(), mat.size(), MPI_DOUBLE,
                send_dst, 0,
                recvBuffer.data(), recvBuffer.size(), MPI_DOUBLE,
                MPI_ANY_SOURCE, 0,
                world, &stat); 

        MPI_Barrier(MPI_COMM_WORLD);

        mat = recvBuffer;
    }

    void print_nonzero_distribution(DenseMatrix &localA, DenseMatrix &localB) {
        for(int i = 0; i < p; i++) {
            if(proc_rank == i) {
                cout << "==================================" << endl;
                cout << "Process " << i << ":" << endl;
                cout << "Rank in Row: " << grid->rankInRow << endl;
                cout << "Rank in Column: " << grid->rankInCol << endl;
                cout << "Rank in Fiber: " << grid->rankInFiber << endl;

                for(int j = 0; j < S->coords.size(); j++) {
                    cout << S->coords[j].string_rep() << endl;
                }

                cout << "==================" << endl;
                cout << "A matrix: " << endl;
                cout << localA << endl; 
                cout << "==================" << endl;
                cout << "B matrix: " << endl;
                cout << localB << endl; 
                cout << "==================================" << endl;
            }

            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
};

#endif