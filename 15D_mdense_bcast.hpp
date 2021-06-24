#pragma once

#include <cmath>
#include <iostream>
#include <utility>
#include <vector>
#include <memory>
#include <Eigen/Dense>
#include <string.h>
#include <mpi.h>

#include "CombBLAS/CombBLAS.h"
#include "sparse_kernels.h"
#include "common.h"
#include "io_utils.h"
#include "als_conjugate_gradients.h"
#include "distributed_sparse.h"

using namespace std;
using namespace combblas;
using namespace Eigen;

class Sparse15D_MDense_Bcast : public Distributed_Sparse {
public:
    int c; // # of layers 

    unique_ptr<CommGrid3D> grid;
    vector<int64_t> blockStarts;

    int rankInFiber, rankInLayer, shift; 

    // We can either read from a file or use the R-mat generator for testing purposes
    void constructor_helper(bool readFromFile, int logM, int nnz_per_row, string filename, int R, int c) {
        // STEP 0: Fill information about this algorithm so that the printout functions work correctly. 

        if(p % (c * c) != 0) {
            if(proc_rank == 0) {
                cout << "Error, for 1.5D algorithm, must have c^2 divide num_procs!" << endl;
                exit(1);
            }
        }

        algorithm_name = "1.5D Block Row Replicated ABS Broadcast with Allreduce";
        proc_grid_names = {"# Block Rows per Layer", "Layers"};
        proc_grid_dimensions = {p/c, c};

        perf_counter_keys = 
                {"Dense Allreduction Time", 
                "Sparse Allreduction Time", 
                "Multiplication Broadcast Time",
                "Computation Time" 
                };

        grid.reset(new CommGrid3D(MPI_COMM_WORLD, c, p / c, 1));

        this->R = R;
        this->c = c;
        this->nnz_per_row = nnz_per_row;

        localAcols = R;
        localBcols = R; 

        if(grid->GetRankInFiber() == 0) {
            if(! readFromFile) {
                generateRandomMatrix(logM, nnz_per_row,
                    grid->GetCommGridLayer(),
                    S,
                    input_Svalues 
                );

                if(proc_rank == 0) {
                    cout << "R-mat generator created " << S.dist_nnz << " nonzeros." << endl;
                }
            }
            else {
                loadMatrixFromFile(filename, grid->GetCommGridLayer(), S, input_Svalues);
                if(proc_rank == 0) {
                    cout << "File reader read " << S.dist_nnz << " nonzeros." << endl;
                }
            }
            this->M = S.distrows;
            this->N = S.distcols;
            localArows = S.nrows;
        }

        MPI_Bcast(&(this->M), 1, MPI_UINT64_T, 0, grid->GetFiberWorld());
        MPI_Bcast(&(this->N), 1, MPI_UINT64_T, 0, grid->GetFiberWorld());

        MPI_Bcast(&(S.local_nnz), 1, MPI_INT, 0, grid->GetFiberWorld());
        MPI_Bcast(&localArows, 1, MPI_UINT64_T, 0, grid->GetFiberWorld());

        localBrows = (int) ceil((float) this->N  * c / p);

        S.rCoords.resize(S.local_nnz);
        S.cCoords.resize(S.local_nnz);
        input_Svalues.resize(S.local_nnz);

        // Broadcast the sparse matrices (the coordinates, not the values)
        MPI_Bcast(S.rCoords.data(), S.local_nnz, MPI_UINT64_T, 0, grid->GetFiberWorld());
        MPI_Bcast(S.cCoords.data(), S.local_nnz, MPI_UINT64_T, 0, grid->GetFiberWorld());

        // Locate block starts within the local sparse matrix (i.e. divide a long
        // block row into subtiles) 
        int currentStart = 0;
        for(int i = 0; i < S.local_nnz; i++) {
            while(S.cCoords[i] >= currentStart) {
                blockStarts.push_back(i);
                currentStart += localBrows;
            }

            // This modding step helps indexing. 
            S.cCoords[i] %= localBrows;
        }
        while(blockStarts.size() < p / c + 1) {
            blockStarts.push_back(S.local_nnz);
        }

        rankInFiber = grid->GetRankInFiber();
        rankInLayer = grid->GetRankInLayer();
        shift = rankInFiber * p / (c * c);

        A_R_split_world = grid->GetCommGridLayer()->GetRowWorld();
        B_R_split_world = grid->GetCommGridLayer()->GetRowWorld();

        check_initialized();
    }

    // Initiates the algorithm for a Graph500 benchmark 
    Sparse15D_MDense_Bcast(int logM, int nnz_per_row, int R, int c, KernelImplementation* k)
        : Distributed_Sparse(k) {
        constructor_helper(false, logM, nnz_per_row, "", R, c);
    }

    // Reads the underlying sparse matrix from a file
    Sparse15D_MDense_Bcast(string &filename, int R, int c, KernelImplementation* k) 
        : Distributed_Sparse(k) {
        constructor_helper(true, 0, -1, filename, R, c);
    }

    // Synchronizes data across three levels of the processor grid
    void initial_synchronize(DenseMatrix *localA, DenseMatrix *localB, VectorXd *SValues) {
        if(localA != nullptr) {
            MPI_Bcast((void*) localA->data(), localA->size(), MPI_DOUBLE, 0, grid->GetFiberWorld());
        }
        if(localB != nullptr) {
            MPI_Bcast((void*) localB->data(), localB->size(), MPI_DOUBLE, 0, grid->GetFiberWorld());
        }
        if(SValues != nullptr) {
            MPI_Bcast((void*) SValues->data(), SValues->size(), MPI_DOUBLE,     0, grid->GetFiberWorld());
        }
    }

    void spmmA(DenseMatrix &localA, DenseMatrix &localB, VectorXd &SValues) {
        algorithm(localA, localB, SValues, nullptr, k_spmmA);
        auto t = start_clock();
        MPI_Allreduce(MPI_IN_PLACE, localA.data(), localA.size(), MPI_DOUBLE, MPI_SUM, grid->GetFiberWorld());
        stop_clock_and_add(t, "Dense Allreduction Time");
    }

    void spmmB(DenseMatrix &localA, DenseMatrix &localB, VectorXd &SValues) {
        algorithm(localA, localB, SValues, nullptr, k_spmmB);
        auto t = start_clock();
        MPI_Allreduce(MPI_IN_PLACE, localB.data(), localB.size(), MPI_DOUBLE, MPI_SUM, grid->GetFiberWorld());
        stop_clock_and_add(t, "Dense Allreduction Time");
    }

    void sddmm(DenseMatrix &localA, DenseMatrix &localB, VectorXd &SValues, VectorXd &sddmm_result) { 
        algorithm(localA, localB, SValues, &sddmm_result, k_sddmm);
        auto t = start_clock();
        MPI_Allreduce(MPI_IN_PLACE, sddmm_result.data(), sddmm_result.size(), MPI_DOUBLE, MPI_SUM, grid->GetFiberWorld()); 
        stop_clock_and_add(t, "Sparse Allreduction Time");
    }

    /*
     * Set the mode to take an SDDMM, SpMM with A as the output matrix, or 
     * SpMM with B as the output matrix. 
     */
    void algorithm(         DenseMatrix &localA, 
                            DenseMatrix &localB, 
                            VectorXd &SValues, 
                            VectorXd *sddmm_result_ptr,
                            DenseMatrix* aux_messages, 
                            KernelMode mode
                            ) {
        MPI_Status stat;
        MPI_Request send_request;
        MPI_Request recv_request;
    
        int nnz_processed = 0;

        // Temporary buffer to hold the received portion of matrix B.

        DenseMatrix recvColSlice(localA.rows(), localA.cols());
        DenseMatrix recvRowSlice(localB.rows(), localB.cols());

        for(int i = 0; i < p / (c * c); i++) {

            int block_id = p / (c * c) * rankInFiber + i;

            assert(blockStarts[block_id] < S.local_nnz);
            assert(blockStarts[block_id + 1] <= S.local_nnz);

            recvRowSlice.setZero();
            if(mode == k_sddmm || mode == k_spmmA) {
                auto t = start_clock();
                if(grid->rankInLayer == block_id) {
                    recvRowSlice = localB;
                } 
                MPI_Bcast(recvRowSlice.data(), recvRowSlice.size(), MPI_DOUBLE, block_id, grid->GetLayerWorld()); 
                stop_clock_and_add(t, "Multiplication Broadcast Time");
            }

            auto t = start_clock();
            if(mode == k_sddmm) {
                nnz_processed += kernel->sddmm_local(
                    S,
                    SValues,
                    localA,
                    recvRowSlice,
                    *sddmm_result_ptr,
                    blockStarts[block_id],
                    blockStarts[block_id + 1]);
            }
            else if(mode == k_spmmA) {
                nnz_processed += kernel->spmm_local(
                    S,
                    SValues,
                    localA,
                    recvRowSlice,
                    nullptr,
                    Amat,
                    blockStarts[block_id],
                    blockStarts[block_id + 1]);
            }
            else if(mode == k_spmmB) {
                nnz_processed += kernel->spmm_local(
                    S,
                    SValues,
                    localA,
                    recvRowSlice,
                    nullptr,
                    Bmat,
                    blockStarts[block_id],
                    blockStarts[block_id + 1]);
            }

            stop_clock_and_add(t, "Computation Time");

            if(mode == k_spmmB) {
                t = start_clock();
                MPI_Reduce(recvRowSlice.data(), localB.data(), localB.size(), MPI_DOUBLE, MPI_SUM, block_id, grid->GetLayerWorld()); 
                stop_clock_and_add(t, "Multiplication Broadcast Time");
            }

        }

        int total_processed;
        MPI_Reduce(&nnz_processed, &total_processed, 1, MPI_INT,
                MPI_SUM, 0, MPI_COMM_WORLD);

        if(proc_rank == 0 && verbose) {
            cout << "Total Nonzeros Processed: " << total_processed << endl;
        }
    }
};

