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

/*
 * Unlike its non-striped counterpart, this algorithm uses reductions of smaller
 * messages instead of one large AllReduce 
 */

class Sparse15D_MDense_Shift_Striped : public Distributed_Sparse {
public:
    int c; // # of layers 

    unique_ptr<CommGrid3D> grid;
    vector<uint64_t> blockStarts;

    int rankInFiber, rankInLayer;

    bool fused;

    // This is only used for the fused algorithm
    SpmatLocal ST;
    vector<uint64_t> transposedBlockStarts; 

    // For the reduce-scatter operation
    vector<int> recvCounts;

    // We can either read from a file or use the R-mat generator for testing purposes
    void constructor_helper(bool readFromFile, int logM, int nnz_per_row, string filename, int R, int c, bool fused) {
        // STEP 0: Fill information about this algorithm so that the printout functions work correctly. 

        this->fused = fused;

        if(p % (c * c) != 0) {
            if(proc_rank == 0) {
                cout << "Error, for 1.5D algorithm, must have c^2 divide num_procs!" << endl;
                exit(1);
            }
        }

        algorithm_name = "1.5D Block Row Replicated S Striped AB Cyclic Shift with Reduce";
        proc_grid_names = {"# Block Rows per Layer", "Layers"};
        proc_grid_dimensions = {p/c, c};

        perf_counter_keys = 
                {"Dense Broadcast Time",
                "Dense Reduction Time", 
                "Sparse Allreduction Time", 
                "Cyclic Shift Time",
                "Computation Time" 
                };

        grid.reset(new CommGrid3D(MPI_COMM_WORLD, c, p / c, 1));

        this->R = R;
        this->c = c;
        this->nnz_per_row = nnz_per_row;

        localAcols = R;
        localBcols = R; 

        SpmatLocal * tptr;
        if(fused) {
            tptr = &ST;
        }
        else {
            tptr = nullptr;
        }

        if(grid->GetRankInFiber() == 0) {
            SpmatLocal::loadMatrix(readFromFile,
               logM,
               nnz_per_row,
               filename,
               grid->GetCommGridLayer(),
               &S,
               tptr 
            );
        }

        S.broadcast_synchronize(grid->GetRankInFiber(), 0, grid->GetFiberWorld());

        if(fused) {
            ST.broadcast_synchronize(grid->GetRankInFiber(), 0, grid->GetFiberWorld());
        }

        this->M = S.M;
        this->N = S.N;

        // TODO: Check the calculation of localArows! 
        localArows = divideAndRoundUp(this->M, p);
        localBrows = divideAndRoundUp(this->N, p);

	    S.divideIntoBlockCols(blockStarts, localBrows, p); 

        if(fused) {
            ST.divideIntoBlockCols(transposedBlockStarts, localArows, p); 
        }

        rankInFiber = grid->GetRankInFiber();
        rankInLayer = grid->GetRankInLayer();

        A_R_split_world = grid->GetCommGridLayer()->GetRowWorld();
        B_R_split_world = grid->GetCommGridLayer()->GetRowWorld();

        for(int i = 0; i < c; i++) {
            recvCounts.push_back(localArows * R);
        }

        check_initialized();
    }

    // Initiates the algorithm for a Graph500 benchmark 
    Sparse15D_MDense_Shift_Striped (int logM, int nnz_per_row, int R, int c, KernelImplementation* k, bool fused)
        : Distributed_Sparse(k) {
        constructor_helper(false, logM, nnz_per_row, "", R, c, fused);
    }

    // Reads the underlying sparse matrix from a file
    Sparse15D_MDense_Shift_Striped (string &filename, int R, int c, KernelImplementation* k, bool fused) 
        : Distributed_Sparse(k) {
        constructor_helper(true, 0, -1, filename, R, c, fused);
    }

    // Synchronizes data across three levels of the processor grid
    void initial_synchronize(DenseMatrix *localA, DenseMatrix *localB, VectorXd *SValues) {
		// A and B are distributed across the entire process grid, so no initial synchronization needed. 
        if(SValues != nullptr) {
            MPI_Bcast((void*) SValues->data(), SValues->size(), MPI_DOUBLE,     0, grid->GetFiberWorld());
        }
    }

    void spmmA(DenseMatrix &localA, DenseMatrix &localB, VectorXd &SValues) {
        algorithm(localA, localB, SValues, nullptr, k_spmmA);
    }

    void spmmB(DenseMatrix &localA, DenseMatrix &localB, VectorXd &SValues) {
        algorithm(localA, localB, SValues, nullptr, k_spmmB);
    }

    void sddmm(DenseMatrix &localA, DenseMatrix &localB, VectorXd &SValues, VectorXd &sddmm_result) { 
        algorithm(localA, localB, SValues, &sddmm_result, k_sddmm);
        auto t = start_clock();
        MPI_Allreduce(MPI_IN_PLACE, sddmm_result.data(), sddmm_result.size(), MPI_DOUBLE, MPI_SUM, grid->GetFiberWorld()); 
        stop_clock_and_add(t, "Sparse Allreduction Time");
    }

    void shiftDenseMatrix(DenseMatrix &mat, DenseMatrix &recvBuffer, int dst) {
        MPI_Status stat;
        auto t = start_clock();

        MPI_Sendrecv(mat.data(), mat.size(), MPI_DOUBLE,
                pMod(rankInLayer + 1, p / c), 0,
                recvBuffer.data(), recvBuffer.size(), MPI_DOUBLE,
                MPI_ANY_SOURCE, 0,
                grid->GetLayerWorld(), &stat);
        stop_clock_and_add(t, "Cyclic Shift Time");

        mat = recvBuffer;
    }

    /*
     * This function does an auto-fusion of the SDDMM / SpMM kernels using
     * a temporary buffer. 
     */
    void fusedSpMM(DenseMatrix &localA, DenseMatrix &localB, DenseMatrix &result, KernelMode mode) {
        int recvRowSliceRows, recvRowSliceCols, accumBufferRows, accumBufferCols;

        if(mode == k_spmmA) {
            assert(localA.rows() == result.rows() && localA.cols() == result.cols());
            recvRowSliceRows = localB.rows();
            recvRowSliceCols = localB.cols();
            accumBufferRows = localA.rows() * c;
            accumBufferCols = R;
        } 
        else {
            assert(localB.rows() == result.rows() && localB.cols() == result.cols());
            recvRowSliceRows = localA.rows();
            recvRowSliceCols = localA.cols();
            accumBufferRows = localB.rows() * c;
            accumBufferCols = R;
        }

        int nnz_processed = 0;

        // Temporary buffer to hold the received portion of matrix B.
        DenseMatrix recvRowSlice(recvRowSliceRows, recvRowSliceCols);

		// Temporary buffer that holds the results of the local ops; this buffer
		// is sharded and then reduced to local portions of the
		DenseMatrix broadcast_buffer = DenseMatrix::Constant(accumBufferRows, accumBufferCols, 0.0); 
		DenseMatrix accumulation_buffer = DenseMatrix::Constant(accumBufferRows, accumBufferCols, 0.0); 

        for(int i = 0; i < c; i++) {
            auto t = start_clock();
            double* dst = accumulation_buffer.data() + accumBufferRows / c * i;

            // TODO: There is definitely a cleaner way to write this
            if(mode == k_spmmA) {
                if(i == rankInFiber) {
                    memcpy(dst, localA.data(), localA.size() * sizeof(double));
                }
                MPI_Bcast((void*) dst, localA.size(), MPI_DOUBLE, i, grid->GetFiberWorld()); 
            }
            else {
                if(i == rankInFiber) {
                    memcpy(dst, localB.data(), localB.size() * sizeof(double));
                }
                MPI_Bcast((void*) dst, localB.size(), MPI_DOUBLE, i, grid->GetFiberWorld()); 
            }
            stop_clock_and_add(t, "Dense Broadcast Time");
        }

        // Temporary SDDMM buffer
        VectorXd sddmm_buffer = like_S_values(0.0);

        for(int i = 0; i < p / c; i++) {
            int block_id = pMod((rankInLayer - i) * c + rankInFiber, p);

            auto t = start_clock();
            if(mode == k_spmmA) { 
                nnz_processed += kernel->spmm_local(
                    S,
                    S.Svalues,
                    accumulation_buffer,
                    localB,
                    Amat,
                    blockStarts[block_id],
                    blockStarts[block_id + 1]);
            }
            else if(mode == k_spmmB) {
                nnz_processed += kernel->spmm_local(
                    S,
                    S.Svalues,
                    accumulation_buffer,
                    localB,
                    Bmat,
                    blockStarts[block_id],
                    blockStarts[block_id + 1]);
            }
            shiftDenseMatrix(localB, recvRowSlice, pMod(rankInLayer + 1, p / c));
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    /*
     * Set the mode to take an SDDMM, SpMM with A as the output matrix, or 
     * SpMM with B as the output matrix. 
     */
    void algorithm(         DenseMatrix &localA, 
                            DenseMatrix &localB, 
                            VectorXd &SValues, 
                            VectorXd *sddmm_result_ptr, 
                            KernelMode mode
                            ) {
        MPI_Status stat;
        MPI_Request send_request;
        MPI_Request recv_request;
    
        int nnz_processed = 0;

        // Temporary buffer to hold the received portion of matrix B.
        DenseMatrix recvRowSlice(localB.rows(), localB.cols());

		// Temporary buffer that holds the results of the local ops; this buffer
		// is sharded and then reduced to local portions of the
		DenseMatrix accumulation_buffer = DenseMatrix::Constant(localA.rows() * c, R, 0.0); 

        if(mode == k_spmmB || mode == k_sddmm) {
            auto t = start_clock();
            MPI_Allgather(localA.data(), localA.size(), MPI_DOUBLE,
                            accumulation_buffer.data(), localA.size(), MPI_DOUBLE, grid->GetFiberWorld());
            stop_clock_and_add(t, "Dense Broadcast Time");
        }

        for(int i = 0; i < p / c; i++) {
            int block_id = pMod((rankInLayer - i) * c + rankInFiber, p);

            assert(blockStarts[block_id] < S.local_nnz);
            assert(blockStarts[block_id + 1] <= S.local_nnz);

            auto t = start_clock();
            if(mode == k_sddmm) {
                nnz_processed += kernel->sddmm_local(
                    S,
                    SValues,
                    accumulation_buffer,
                    localB,
                    *sddmm_result_ptr,
                    blockStarts[block_id],
                    blockStarts[block_id + 1]);
            }
            else if(mode == k_spmmA) { 
                nnz_processed += kernel->spmm_local(
                    S,
                    SValues,
                    accumulation_buffer,
                    localB,
                    Amat,
                    blockStarts[block_id],
                    blockStarts[block_id + 1]);
            }
            else if(mode == k_spmmB) {
                nnz_processed += kernel->spmm_local(
                    S,
                    SValues,
                    accumulation_buffer,
                    localB,
                    Bmat,
                    blockStarts[block_id],
                    blockStarts[block_id + 1]);
            }
            stop_clock_and_add(t, "Computation Time"); 
            shiftDenseMatrix(localB, recvRowSlice, pMod(rankInLayer + 1, p / c));
            MPI_Barrier(MPI_COMM_WORLD);
        }

        if(mode == k_spmmA) {
            auto t = start_clock(); 
            MPI_Reduce_scatter(accumulation_buffer.data(), 
                    localA.data(), recvCounts.data(),
                       MPI_DOUBLE, MPI_SUM, grid->GetFiberWorld());
            stop_clock_and_add(t, "Dense Reduction Time");
        }

        int total_processed;
        MPI_Reduce(&nnz_processed, &total_processed, 1, MPI_INT,
                MPI_SUM, 0, MPI_COMM_WORLD);

        if(proc_rank == 0 && verbose) {
            cout << "Total Nonzeros Processed: " << total_processed << endl;
        }
    }
};
