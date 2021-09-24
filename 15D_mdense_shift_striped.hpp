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

class ShardedBlockCyclicColumn: public NonzeroDistribution {
public:
    int p, c;
    shared_ptr<FlexibleGrid> grid;

    ShardedBlockCyclicColumn(int M, int N, int p, int c, shared_ptr<FlexibleGrid> &grid) { 
        world = MPI_COMM_WORLD;
        this->p = p;
        this->c = c;
        this->grid = grid;
        rows_in_block = divideAndRoundUp(M, p) * c; 
        cols_in_block = divideAndRoundUp(N, p); 
    }

	int blockOwner(int row_block, int col_block) {
        int rowRank = row_block;
        int layerRank = col_block % c;

        return grid->get_global_rank(layerRank, rowRank, 0);
    }
};

/*
 * Unlike its non-striped counterpart, this algorithm uses reductions of smaller
 * messages instead of one large AllReduce 
 */
class Sparse15D_MDense_Shift_Striped : public Distributed_Sparse {
public:
    int c; // Replication factor for the 1.5D Algorithm 
    int fusionAproach;

    Sparse15D_MDense_Shift_Striped(SpmatLocal* S_input, int R, int c, int fusionApproach, KernelImplementation* k) 
        : Distributed_Sparse(k, R) 
    {
        this->fusionApproach = fusionApproach;
        this->c = c;

        if(p % c != 0) {
            if(proc_rank == 0) {
                cout << "Error, for 1.5D algorithm, must have c divide num_procs!" << endl;
                exit(1);
            }
        }

        algorithm_name = "1.5D Block Row Replicated S Striped AB Cyclic Shift";
        proc_grid_names = {"Layers", "# of Block Rows per Layer"};

        perf_counter_keys = 
                {"Dense Broadcast Time",
                "Cyclic Shift Time",
                "Computation Time" 
                };

        grid.reset(new FlexibleGrid(c, p / c, 1, 3));

        localAcols = R;
        localBcols = R; 

        r_split = false;

        this->M = S_input->M;
        this->N = S_input->N;
        ShardedBlockCyclicColumn standard_dist(M, N, p, c);
        ShardedBlockCyclicColumn transpose_dist(N, M, p, c);

        // Copies the nonzeros of the sparse matrix locally (so we can do whatever
        // we want with them; this does incur a memory overhead)
        S.reset(S_input->redistribute_nonzeros(&standard_dist, false, false));
        ST.reset(S->redistribute_nonzeros(&transpose_dist, true, false));

        localArows = divideAndRoundUp(this->M, p);
        localBrows = divideAndRoundUp(this->N, p);

        // Define submatrix boundaries 
        aSubmatrices.emplace_back(localArows * (grid->i + c * grid->j), 0, localArows, localAcols);
        bSubmatrices.emplace_back(localBrows * (grid->i + c * grid->j), 0, localBrows, localBcols);

        for(int i = 0; i < S->coords.size(); i++) {
            S->coords[i].r %= localArows * c;
        }
        S->divideIntoBlockCols(localBrows, p, true); 
 
        for(int i = 0; i < ST->coords.size(); i++) {
            ST->coords[i].r %= localBrows * c;
        } 
        ST->divideIntoBlockCols(localArows, p, true); 

        S->own_all_coordinates();
        ST->own_all_coordinates();

        assert(fusionApproach == 1 || fusionApproach == 2);

        bool local_tpose;
        if(fusionApproach == 1) {
            local_tpose = false;
        }
        else {
            local_tpose = true;
        }

        S->initializeCSRBlocks(localArows * c, localBrows, max_nnz, local_tpose);
        ST->initializeCSRBlocks(localBrows * c, localArows, max_nnz_tpose, local_tpose);

        check_initialized();
    }
 
    void initial_synchronize(DenseMatrix *localA, DenseMatrix *localB, VectorXd *SValues) {
        // Empty method, no initialization needed 
    }

    /*
     * This is fusion strategy 1.
     *
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

		// Temporary buffer that holds the results of the local ops; this buffer
		// is sharded and then reduced to local portions of the
		DenseMatrix broadcast_buffer = DenseMatrix::Constant(Arole->rows() * c, R, 0.0); 
		DenseMatrix accumulation_buffer = DenseMatrix::Constant(Arole->rows() * c, R, 0.0); 

        auto t = start_clock();
        MPI_Allgather(Arole->data(), Arole->size(), MPI_DOUBLE,
                        broadcast_buffer.data(), Arole->size(), MPI_DOUBLE, grid->row_world);
        stop_clock_and_add(t, "Dense Broadcast Time");

        for(int i = 0; i < p / c; i++) {
            int block_id = pMod((grid->rankInCol - i) * c + grid->rankInRow, p);

            auto t = start_clock();

            // TODO: Here, need a conditional for auto-fusion or not 
            nnz_processed += kernel->sddmm_local(
                *choice,
                Svalues,
                broadcast_buffer,
                *Brole,
                sddmm_buffer,
                block_id);
 
            nnz_processed += kernel->spmm_local(
                *choice,
                sddmm_buffer,
                accumulation_buffer,
                *Brole,
                Amat,
                block_id); 
            stop_clock_and_add(t, "Computation Time");

            t = start_clock();
            shiftDenseMatrix(*Brole, grid->col_world, pMod(grid->rankInCol + 1, p / c));
            stop_clock_and_add(t, "Cyclic Shift Time");

            MPI_Barrier(MPI_COMM_WORLD);
        }

        vector<int> recvCounts;
        for(int i = 0; i < c; i++) {
            recvCounts.push_back(Arole->rows() * R);
        }

        t = start_clock();
        MPI_Reduce_scatter(accumulation_buffer.data(), 
                result.data(), recvCounts.data(),
                    MPI_DOUBLE, MPI_SUM, grid->row_world);
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
		// Temporary buffer that holds the results of the local ops; this buffer
		// is sharded and then reduced to local portions of the
		DenseMatrix accumulation_buffer = DenseMatrix::Constant(localA.rows() * c, R, 0.0); 

        if(mode == k_spmmB || mode == k_sddmm) {
            auto t = start_clock();
            MPI_Allgather(localA.data(), localA.size(), MPI_DOUBLE,
                            accumulation_buffer.data(), localA.size(), MPI_DOUBLE, grid->row_world);
            stop_clock_and_add(t, "Dense Broadcast Time");
        }

        for(int i = 0; i < p / c; i++) {
            int block_id = pMod((grid->rankInCol - i) * c + grid->rankInRow, p);

            assert(S->blockStarts[block_id] <= S->coords.size());
            assert(S->blockStarts[block_id + 1] <= S->coords.size());

            auto t = start_clock();

            nnz_processed += kernel->triple_function(
                mode,
                *S,
                SValues,
                accumulation_buffer,
                localB,
                sddmm_result_ptr,
                S->blockStarts[block_id],
                S->blockStarts[block_id + 1]);

            stop_clock_and_add(t, "Computation Time"); 

            t = start_clock();
            shiftDenseMatrix(localB, layer_axis, pMod(rankInLayer + 1, p / c));
            stop_clock_and_add(t, "Cyclic Shift Time");

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
                       MPI_DOUBLE, MPI_SUM, grid->row_world);
            stop_clock_and_add(t, "Dense Broadcast Time");
        }
    }
};
