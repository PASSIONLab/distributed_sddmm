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

    DenseMatrix accumulation_buffer;

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

        return grid->get_global_rank(rowRank, layerRank, 0);
    }
};

/*
 * Unlike its non-striped counterpart, this algorithm uses reductions of smaller
 * messages instead of one large AllReduce 
 */
class Sparse15D_MDense_Shift_Striped : public Distributed_Sparse {
public:
    int c; // Replication factor for the 1.5D Algorithm 
    int fusionApproach;

    DenseMatrix accumulation_buffer;

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

        grid.reset(new FlexibleGrid(p/c, c, 1, 1));

        localAcols = R;
        localBcols = R; 

        r_split = false;

        this->M = S_input->M;
        this->N = S_input->N;
        ShardedBlockCyclicColumn standard_dist(M, N, p, c, grid);
        ShardedBlockCyclicColumn transpose_dist(N, M, p, c, grid);

        // Copies the nonzeros of the sparse matrix locally (so we can do whatever
        // we want with them; this does incur a memory overhead)
        S.reset(S_input->redistribute_nonzeros(&standard_dist, false, false));
        ST.reset(S->redistribute_nonzeros(&transpose_dist, true, false));

        localArows = divideAndRoundUp(this->M, p);
        localBrows = divideAndRoundUp(this->N, p);

        // Define submatrix boundaries 

        // TODO: I'm pretty sure that this is broken...
        aSubmatrices.emplace_back(localArows * (c * grid->i + grid->j), 0, localArows, localAcols);
        bSubmatrices.emplace_back(localBrows * (c * grid->i + grid->j), 0, localBrows, localBcols);

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
        if(fusionApproach == 2) {
            local_tpose = false;
        }
        else {
            local_tpose = true;
        }

        S->initializeCSRBlocks(localArows * c, localBrows, -1, local_tpose);
        ST->initializeCSRBlocks(localBrows * c, localArows, -1, local_tpose);

        check_initialized();
    }
 
    void initial_shift(DenseMatrix *localA, DenseMatrix *localB, KernelMode mode) {
        // Empty on purpose
    }


    void de_shift(DenseMatrix *localA, DenseMatrix *localB, KernelMode mode) {
        // Empty on purpose
    }

    void fusedSpMM(DenseMatrix &localA, 
            DenseMatrix &localB, 
            VectorXd &Svalues, 
            VectorXd &sddmm_buffer,
            MatMode mode) {

        if(fusionApproach == 1) {
            Distributed_Sparse::fusedSpMM(localA, 
                localB, 
                Svalues, 
                sddmm_buffer,
                mode);
            return;
        }

        DenseMatrix *Arole, *Brole;
        SpmatLocal* choice;

        if(mode == Amat) {
            assert(Svalues.size() == S->coords.size());
            Arole = &localA;
            Brole = &localB;
            choice = S.get();
        } 
        else if(mode == Bmat) {
            assert(Svalues.size() == ST->coords.size());
            Arole = &localB;
            Brole = &localA;
            choice = ST.get(); 
        }
        else {
            assert(false);
        }

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
            kernel->triple_function(
                k_sddmmA,
                *choice,
                broadcast_buffer,
                *Brole,
                block_id);

            // TODO: Slightly broken! We need to optimize the
            // copy operation between these two kernels
 
            kernel->triple_function(
                k_spmmA,
                *choice,
                accumulation_buffer,
                *Brole,
                block_id);

            stop_clock_and_add(t, "Computation Time");

            t = start_clock();
            shiftDenseMatrix(*Brole, grid->col_world, pMod(grid->rankInCol + 1, p / c), 55);
            stop_clock_and_add(t, "Cyclic Shift Time");

            MPI_Barrier(MPI_COMM_WORLD);
        }

        vector<int> recvCounts;
        for(int i = 0; i < c; i++) {
            recvCounts.push_back(Arole->rows() * R);
        }

        t = start_clock();
        MPI_Reduce_scatter(accumulation_buffer.data(), 
                Arole->data(), recvCounts.data(),
                    MPI_DOUBLE, MPI_SUM, grid->row_world);
        stop_clock_and_add(t, "Dense Broadcast Time");

        // TODO: This currently doesn't copy out the result SDDMM buffer...
    }

    VectorXd like_S_values(double value) {
        if(fusionApproach == 1) {
            return VectorXd::Constant(ST->owned_coords_end - ST->owned_coords_start, value); 
        }
        else {
            return VectorXd::Constant(S->owned_coords_end - S->owned_coords_start, value); 
        }
    }

    VectorXd like_ST_values(double value) {
        if(fusionApproach == 1) {
            return VectorXd::Constant(S->owned_coords_end - S->owned_coords_start, value); 
        }
        else {
            return VectorXd::Constant(ST->owned_coords_end - ST->owned_coords_start, value); 
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
                            KernelMode mode,
                            bool initial_replicate 
                            ) {

        DenseMatrix *Arole, *Brole;
        SpmatLocal* choice;
        if(mode == k_spmmA || mode == k_sddmmA) {
            Arole = &localB;
            Brole = &localA;
            choice = ST.get();
        } 
        else if(mode == k_spmmB || mode == k_sddmmB) {
            Arole = &localA;
            Brole = &localB;
            choice = S.get(); 
        }
        else {
            assert(false);
        }

		// Temporary buffer that holds the results of the local ops; this buffer
		// is sharded and then reduced to the local portions of the matrix. 
        if(initial_replicate) {
            auto t = start_clock();
            accumulation_buffer = DenseMatrix::Constant(Arole->rows() * c, R, 0.0); 
            MPI_Allgather(Arole->data(), Arole->size(), MPI_DOUBLE,
                            accumulation_buffer.data(), Arole->size(), MPI_DOUBLE, grid->row_world);
            stop_clock_and_add(t, "Dense Broadcast Time");  
        }

        auto t = start_clock();
        if(mode == k_sddmmA || mode == k_sddmmB) {
            choice->setValuesConstant(0.0);
        }
        else {
            choice->setCSRValues(SValues);
        }
        stop_clock_and_add(t, "Computation Time"); 
 
        for(int i = 0; i < p / c; i++) {
            int block_id = pMod((grid->rankInCol - i) * c + grid->rankInRow, p);

            assert(S->blockStarts[block_id] <= S->coords.size());
            assert(S->blockStarts[block_id + 1] <= S->coords.size());

            auto t = start_clock();
            kernel->triple_function(
                mode == k_spmmA ? k_spmmB : mode,
                *choice,
                accumulation_buffer,
                *Brole,
                block_id);
            stop_clock_and_add(t, "Computation Time"); 

            t = start_clock();
            shiftDenseMatrix(*Brole, grid->col_world, pMod(grid->rankInCol + 1, p / c), 55);
            MPI_Barrier(MPI_COMM_WORLD);
            stop_clock_and_add(t, "Cyclic Shift Time");
        }        

        if(mode == k_sddmmA || mode == k_sddmmB) {
            auto t = start_clock();
            *sddmm_result_ptr = SValues.cwiseProduct(choice->getCSRValues());
            stop_clock_and_add(t, "Computation Time"); 
        }

        // TODO: If the fusion mode is 1, need to add a terminal reduction! 
    }
};
