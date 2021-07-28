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
#include "als_conjugate_gradients.h"
#include "distributed_sparse.h"

using namespace std;
using namespace combblas;
using namespace Eigen;

/*
 * Layout for redistributing the sparse matrix. 
 */
class ShardedBlockCyclicColumn: public NonzeroDistribution {
public:
    int p, c;

    ShardedBlockCyclicColumn(int p, int c) {
        this->p = p;
        this->c = c;
        world = MPI_COMM_WORLD;
    }

	int getOwner(int row, int col, int transpose) {
        int rDim = transpose ? this->N : this->M;
        int cDim = transpose ? this->M : this->N;

        int rows_in_block = divideAndRoundUp(rDim, p) * c; 
        int cols_in_block = divideAndRoundUp(cDim, p); 

        if(transpose) {
            int temp = row;
            row = col;
            col = temp;
        }
        
        int block_row = row / rows_in_block;
        int block_col = col / cols_in_block;

        int rowRank = block_row;
        int layerRank = block_col % c;

        return p / c * layerRank + rowRank;
    }
};

/*
 * Unlike its non-striped counterpart, this algorithm uses reductions of smaller
 * messages instead of one large AllReduce 
 */
class Sparse15D_MDense_Shift_Striped : public Distributed_Sparse {
public:
    int c; // Replication factor for the 1.5D Algorithm 

    shared_ptr<CommGrid> grid;

    int rankInFiber, rankInLayer;
    MPI_Comm fiber_axis, layer_axis; 

    bool auto_fusion;

    // This is only used for the fused algorithm
    unique_ptr<SpmatLocal> ST;

    /*
     * This is a debugging function.
     */
    void print_nonzero_distribution() {
        for(int i = 0; i < p; i++) {
            if(proc_rank == i) {
                cout << "Process " << i << ":" << endl;
                cout << "Rank in Fiber: " << rankInFiber << endl;
                cout << "Rank in Layer: " << rankInLayer << endl;

                for(int j = 0; j < S->coords.size(); j++) {
                    cout << S->coords[j].string_rep() << endl;
                }
                cout << "==================" << endl;
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    Sparse15D_MDense_Shift_Striped(SpmatLocal* S_input, int R, int c, bool fused, KernelImplementation* k) 
        : Distributed_Sparse(k) 
    {
        this->fused = fused;
        this->R = R;
        this->c = c;

        if(p % c != 0) {
            if(proc_rank == 0) {
                cout << "Error, for 1.5D algorithm, must have c divide num_procs!" << endl;
                exit(1);
            }
        }

        algorithm_name = "1.5D Block Row Replicated S Striped AB Cyclic Shift with Reduce";
        proc_grid_names = {"# Block Rows per Layer", "Layers"};
        proc_grid_dimensions = {p/c, c};

        perf_counter_keys = 
                {"Dense Broadcast Time",
                "Cyclic Shift Time",
                "Computation Time" 
                };

        /*
         * WARNING: DO NOT TRANSPOSE THE GRID WITHOUT ALSO CHANGING THE
         * DATA DISTRIBUTION INDEXING!!! 
         */
        grid.reset(new CommGrid(MPI_COMM_WORLD, c, p / c));
        rankInFiber = grid->GetRankInProcCol();
        rankInLayer = grid->GetRankInProcRow();
        layer_axis = grid->GetRowWorld();
        fiber_axis = grid->GetColWorld();

        localAcols = R;
        localBcols = R; 

        r_split = false;
        
        ShardedBlockCyclicColumn nonzero_dist(p, c);

        // Copies the nonzeros of the sparse matrix locally (so we can do whatever
        // we want with them; this does incur a memory overhead)
        S.reset(S_input->redistribute_nonzeros(&nonzero_dist, false, false));

        if(fused) {
            ST.reset(S->redistribute_nonzeros(&nonzero_dist, true, false));
        }

        this->M = S->M;
        this->N = S->N;
        localArows = divideAndRoundUp(this->M, p);
        localBrows = divideAndRoundUp(this->N, p);

        // Postprocessing nonzeros for easy feeding to local kernels 
        for(int i = 0; i < S->coords.size(); i++) {
            S->coords[i].r %= localArows * c;
        }
        S->divideIntoBlockCols(localBrows, p, true); 

        if(fused) {
            for(int i = 0; i < ST->coords.size(); i++) {
                ST->coords[i].r %= localBrows * c;
            } 
            ST->divideIntoBlockCols(localArows, p, true);
        }

        check_initialized();
    }
 
    void initial_synchronize(DenseMatrix *localA, DenseMatrix *localB, VectorXd *SValues) {
        // Empty method, no initialization needed 
    }

    void shiftDenseMatrix(DenseMatrix &mat, DenseMatrix &recvBuffer, int dst) {
        MPI_Status stat;
        auto t = start_clock();

        MPI_Sendrecv(mat.data(), mat.size(), MPI_DOUBLE,
                pMod(rankInLayer + 1, p / c), 0,
                recvBuffer.data(), recvBuffer.size(), MPI_DOUBLE,
                MPI_ANY_SOURCE, 0,
                layer_axis, &stat);
        stop_clock_and_add(t, "Cyclic Shift Time");

        mat = recvBuffer;
    }

    VectorXd like_ST_values(double value) {
        return VectorXd::Constant(ST->coords.size(), value); 
    }

    /*
     * This function does an auto-fusion of the SDDMM / SpMM kernels using
     * a temporary buffer (needs to be supplied, since this function does)
     * auto-fusion.
     *
     * One matrix plays the role of the ``A matrix", which is broadcast to
     * the accumulation buffer,
     * and the other plays the role of the B matrix, which is cyclic shifted. 
     * The result is reduce-scattered in the same data distribution as
     * the A-matrix role. 
     */
    void fusedSpMM(DenseMatrix &localA, DenseMatrix &localB, VectorXd &Svalues, VectorXd &sddmm_buffer, DenseMatrix &result, MatMode mode) {
        assert(this->fused); 
        DenseMatrix *Arole, *Brole;
        SpmatLocal* choice;

        if(mode == Amat) {
            assert(localA.rows() == result.rows() && localA.cols() == result.cols());
            assert(Svalues.size() == S->coords.size());
            Arole = &localA;
            Brole = &localB;
            choice = S.get();
        } 
        else if(mode == Bmat) {
            assert(localB.rows() == result.rows() && localB.cols() == result.cols());
            assert(Svalues.size() == ST->coords.size());
            Arole = &localB;
            Brole = &localA;
            choice = ST.get(); 
        }
        else {
            assert(false);
        }

        int nnz_processed = 0;

        // Temporary buffer to hold the received portion of B-role matrix. 
        DenseMatrix recvRowSlice(Brole->rows(), Brole->cols());

		// Temporary buffer that holds the results of the local ops; this buffer
		// is sharded and then reduced to local portions of the
		DenseMatrix broadcast_buffer = DenseMatrix::Constant(Arole->rows() * c, R, 0.0); 
		DenseMatrix accumulation_buffer = DenseMatrix::Constant(Arole->rows() * c, R, 0.0); 

        auto t = start_clock();
        MPI_Allgather(Arole->data(), Arole->size(), MPI_DOUBLE,
                        broadcast_buffer.data(), Arole->size(), MPI_DOUBLE, fiber_axis);
        stop_clock_and_add(t, "Dense Broadcast Time");

        for(int i = 0; i < p / c; i++) {
            int block_id = pMod((rankInLayer - i) * c + rankInFiber, p);

            auto t = start_clock();

            // TODO: Here, need a conditional for auto-fusion or not 
            nnz_processed += kernel->sddmm_local(
                *choice,
                Svalues,
                broadcast_buffer,
                *Brole,
                sddmm_buffer,
                choice->blockStarts[block_id],
                choice->blockStarts[block_id + 1]); 
             
            nnz_processed += kernel->spmm_local(
                *choice,
                sddmm_buffer,
                accumulation_buffer,
                *Brole,
                Amat,
                choice->blockStarts[block_id],
                choice->blockStarts[block_id + 1]); 
            stop_clock_and_add(t, "Computation Time");
   
            shiftDenseMatrix(*Brole, recvRowSlice, pMod(rankInLayer + 1, p / c));
            MPI_Barrier(MPI_COMM_WORLD);
        }

        vector<int> recvCounts;
        for(int i = 0; i < c; i++) {
            recvCounts.push_back(Arole->rows() * R);
        }

        t = start_clock();
        MPI_Reduce_scatter(accumulation_buffer.data(), 
                result.data(), recvCounts.data(),
                    MPI_DOUBLE, MPI_SUM, fiber_axis);
        stop_clock_and_add(t, "Dense Broadcast Time");

        int total_processed;
        MPI_Reduce(&nnz_processed, &total_processed, 1, MPI_INT,
                MPI_SUM, 0, MPI_COMM_WORLD);
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
        int nnz_processed = 0;

        // Temporary buffer to hold the received portion of matrix B.
        DenseMatrix recvRowSlice(localB.rows(), localB.cols());

		// Temporary buffer that holds the results of the local ops; this buffer
		// is sharded and then reduced to local portions of the
		DenseMatrix accumulation_buffer = DenseMatrix::Constant(localA.rows() * c, R, 0.0); 

        if(mode == k_spmmB || mode == k_sddmm) {
            auto t = start_clock();
            MPI_Allgather(localA.data(), localA.size(), MPI_DOUBLE,
                            accumulation_buffer.data(), localA.size(), MPI_DOUBLE, fiber_axis);
            stop_clock_and_add(t, "Dense Broadcast Time");
        }

        for(int i = 0; i < p / c; i++) {
            int block_id = pMod((rankInLayer - i) * c + rankInFiber, p);

            assert(S->blockStarts[block_id] <= S->coords.size());
            assert(S->blockStarts[block_id + 1] <= S->coords.size());

            auto t = start_clock();

            if(mode == k_sddmm) {
                nnz_processed += kernel->sddmm_local(
                    *S,
                    SValues,
                    accumulation_buffer,
                    localB,
                    *sddmm_result_ptr,
                    S->blockStarts[block_id],
                    S->blockStarts[block_id + 1]);
            }
            else if(mode == k_spmmA) { 
                nnz_processed += kernel->spmm_local(
                    *S,
                    SValues,
                    accumulation_buffer,
                    localB,
                    Amat,
                    S->blockStarts[block_id],
                    S->blockStarts[block_id + 1]);
            }
            else if(mode == k_spmmB) {
                nnz_processed += kernel->spmm_local(
                    *S,
                    SValues,
                    accumulation_buffer,
                    localB,
                    Bmat,
                    S->blockStarts[block_id],
                    S->blockStarts[block_id + 1]);
            }

            stop_clock_and_add(t, "Computation Time"); 
            shiftDenseMatrix(localB, recvRowSlice, pMod(rankInLayer + 1, p / c));
            MPI_Barrier(MPI_COMM_WORLD);
        }

        if(mode == k_spmmA) {
            auto t = start_clock(); 

            vector<int> recvCounts;
            for(int i = 0; i < c; i++) {
                recvCounts.push_back(localArows * R);
            }

            MPI_Reduce_scatter(accumulation_buffer.data(), 
                    localA.data(), recvCounts.data(),
                       MPI_DOUBLE, MPI_SUM, fiber_axis);
            stop_clock_and_add(t, "Dense Broadcast Time");
        }

        int total_processed;
        MPI_Reduce(&nnz_processed, &total_processed, 1, MPI_INT,
                MPI_SUM, 0, MPI_COMM_WORLD);

        if(proc_rank == 0 && verbose) {
            cout << "Total Nonzeros Processed: " << total_processed << endl;
        }
    }
};
