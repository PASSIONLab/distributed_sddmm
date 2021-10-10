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
class Floor2D : public NonzeroDistribution {
public:
    shared_ptr<FlexibleGrid> grid;

    Floor2D(int M, int N, int sqrtpc, int c, shared_ptr<FlexibleGrid> &grid) {
        this->grid = grid;
        world = MPI_COMM_WORLD;

        rows_in_block = divideAndRoundUp(M, sqrtpc);
        cols_in_block = divideAndRoundUp(N, sqrtpc);
    }

	int blockOwner(int row_block, int col_block) { 
        return grid->get_global_rank(row_block, col_block, 0); 
    }
};

class Sparse25D_Cannon_Sparse : public Distributed_Sparse {
public:
    int sqrtpc;

    void broadcastCoordinatesFromFloor(unique_ptr<SpmatLocal> &spmat) {
        int num_nnz = spmat->coords.size();
        MPI_Bcast(&num_nnz, 1, MPI_INT, 0, grid->fiber_world);
        if(grid->rankInFiber > 0) {
            spmat->coords.resize(num_nnz);
        } 
        MPI_Bcast(spmat->coords.data(), spmat->coords.size(), SPCOORD, 0, grid->fiber_world);
    }

    Sparse25D_Cannon_Sparse(SpmatLocal* S_input, int R, int c, KernelImplementation* k) : Distributed_Sparse(k, R) { 
        this->c = c;
        sqrtpc = (int) sqrt(p / c);
 
        if(sqrtpc * sqrtpc * c != p) {
            if(proc_rank == 0) {
                cout << "Error, for 2.5D algorithm, p / c must be a perfect square!" << endl;
                cout << p << " " << c << endl;
            }
            exit(1);
        }

        algorithm_name = "2.5D Cannon's Algorithm Replicating Sparse Matrix";
        proc_grid_names = {"# Rows", "# Cols", "# Layers"};

        perf_counter_keys = 
                {"Dense Cyclic Shift Time",
                 "Sparse Fiber Communication Time",
                 "Computation Time" 
                };

        grid.reset(new FlexibleGrid(sqrtpc, sqrtpc, c, 3));

        A_R_split_world = grid->colfiber_slice; 
        B_R_split_world = grid->colfiber_slice; 

        localAcols = R / (sqrtpc * c);
        localBcols = R / (sqrtpc * c);

        if(localAcols * sqrtpc * c != R) {
            cout << "Error, R must be divisible by sqrt(pc)!" << endl;
            exit(1);
        }

        r_split = true;

        this->M = S_input->M;
        this->N = S_input->N;
        localArows = divideAndRoundUp(this->M, sqrtpc);
        localBrows = divideAndRoundUp(this->N, sqrtpc);

        int shift = pMod(grid->j + grid->i, sqrtpc); 

        // Define submatrix boundaries 
        aSubmatrices.emplace_back(localArows * grid->i, localAcols * c * shift + grid->k * localAcols, localArows, localAcols);
        bSubmatrices.emplace_back(localBrows * grid->i, localBcols * c * shift + grid->k * localBcols, localBrows, localBcols);

        /* Distribute the nonzeros, storing them initially on the
         * bottom face of the cuboid and then broadcasting.  
         */
        Floor2D nonzero_dist(M, N, sqrtpc, c, grid);
        Floor2D transpose_dist(N, M, sqrtpc, c, grid);

        S.reset(S_input->redistribute_nonzeros(&nonzero_dist, false, false));
        ST.reset(S_input->redistribute_nonzeros(&transpose_dist, true, false));

        broadcastCoordinatesFromFloor(S);
        broadcastCoordinatesFromFloor(ST);

        // Each processor is only responsible for a fraction of its nonzeros 
        S->shard_across_layers(c, grid->k);
        ST->shard_across_layers(c, grid->k);

        // Postprocess the coordinates
        for(int i = 0; i < S->coords.size(); i++) {
            S->coords[i].r %= localArows;
            S->coords[i].c %= localBrows;
        }

        for(int i = 0; i < ST->coords.size(); i++) {
            ST->coords[i].r %= localBrows;
            ST->coords[i].c %= localArows;
        }

        // Commit to CSR format, then locally transpose ST
        S->monolithBlockColumn();
        ST->monolithBlockColumn();

	    S->initializeCSRBlocks(localArows, localBrows, -1, false); 
	    ST->initializeCSRBlocks(localBrows, localArows, -1, false);

        check_initialized();
    }

    void initial_shift(DenseMatrix *localA, DenseMatrix *localB, KernelMode mode) { 
        int tpose_dest = grid->get_global_rank(grid->j, grid->i, grid->k); 
        if(mode == k_sddmmA || mode == k_spmmA) {
            shiftDenseMatrix(*localB, MPI_COMM_WORLD, 
                    tpose_dest, 1);
        }
        else if(mode == k_sddmmB || mode == k_spmmB) {
            shiftDenseMatrix(*localA, MPI_COMM_WORLD, 
                    tpose_dest, 1);
        }

        //shiftDenseMatrix(*localA, grid->row_world, 
        //        pMod(grid->rankInRow - grid->rankInCol, sqrtpc), 1); 
        //shiftDenseMatrix(*localB, grid->col_world, 
        //        pMod(grid->rankInCol - grid->rankInRow, sqrtpc), 2);
    }

    void de_shift(DenseMatrix *localA, DenseMatrix *localB, KernelMode mode) {
        initial_shift(localA, localB, mode); 
    }

    void algorithm(     DenseMatrix &localA, 
                        DenseMatrix &localB, 
                        VectorXd &SValues, 
                        VectorXd *sddmm_result_ptr, 
                        KernelMode mode,
                        bool initial_replicate
                        ) {
        SpmatLocal* choice;

        DenseMatrix *Arole, *Brole;

        if(mode == k_spmmA || mode == k_sddmmA) {
            assert(SValues.size() == S->owned_coords_end - S->owned_coords_start);
            choice = S.get();
            Arole = &localA;
            Brole = &localB;
        } 
        else if(mode == k_spmmB || mode == k_sddmmB) {
            assert(SValues.size() == ST->owned_coords_end - ST->owned_coords_start);
            choice = ST.get();
            Arole = &localB;
            Brole = &localA;
        }

        int nnz_processed = 0;
		VectorXd accumulation_buffer = VectorXd::Constant(choice->coords.size(), 0.0); 

        if(mode == k_spmmA || mode == k_spmmB) {
            auto t = start_clock();
            MPI_Allgatherv(
                SValues.data(),
                SValues.size(),
                MPI_DOUBLE,
                accumulation_buffer.data(),
                choice->layer_coords_sizes.data(),
                choice->layer_coords_start.data(),
                MPI_DOUBLE,
                grid->fiber_world
                );

            stop_clock_and_add(t, "Sparse Fiber Communication Time");
        }

        auto t = start_clock();
        choice->setCSRValues(accumulation_buffer); 
        stop_clock_and_add(t, "Computation Time");

        KernelMode temp = mode;
        if(mode == k_sddmmB) {
            temp = k_sddmmA;
        }
        if(mode == k_spmmB){
            temp = k_spmmA;
        }

        for(int i = 0; i < sqrtpc; i++) {
            auto t = start_clock();

            nnz_processed += kernel->triple_function(
                temp, 
                *choice,
                *Arole,
                *Brole,
                0);

            stop_clock_and_add(t, "Computation Time");

            if(sqrtpc > 1) {
                t = start_clock();
                shiftDenseMatrix(*Arole, grid->row_world, 
                        pMod(grid->rankInRow + 1, sqrtpc), 1);
                shiftDenseMatrix(*Brole, grid->col_world, 
                        pMod(grid->rankInCol + 1, sqrtpc), 2);
                stop_clock_and_add(t, "Dense Cyclic Shift Time");
            }
        }

        if(mode == k_sddmmA || mode == k_sddmmB) {
            auto t = start_clock();
            accumulation_buffer = choice->getCSRValues();
            stop_clock_and_add(t, "Computation Time");

            t = start_clock();
            MPI_Reduce_scatter(accumulation_buffer.data(),
                sddmm_result_ptr->data(),
                choice->layer_coords_sizes.data(), 
                MPI_DOUBLE,
                MPI_SUM,
                grid->fiber_world
            );
            stop_clock_and_add(t, "Sparse Fiber Communication Time");

            t = start_clock();
            *sddmm_result_ptr = SValues.cwiseProduct(*sddmm_result_ptr);
            stop_clock_and_add(t, "Computation Time");
        }
    }
};
