#pragma once

#include <algorithm>
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
class Block_Cyclic25D : public NonzeroDistribution {
public:
    int sqrtpc;
    int c;

    shared_ptr<FlexibleGrid> grid;

    Block_Cyclic25D(int M, int N, int sqrtpc, int c, shared_ptr<FlexibleGrid> &grid) {
        world = MPI_COMM_WORLD;
        this->sqrtpc = sqrtpc;
        this->c = c;
        this->grid = grid;

        rows_in_block = divideAndRoundUp(M, sqrtpc * c) * c;
        cols_in_block = divideAndRoundUp(N, sqrtpc * c);
    }

	int blockOwner(int row_block, int col_block) {
        return grid->get_global_rank(row_block, col_block / c, col_block % c);
    }
};

class Sparse25D_Cannon_Dense : public Distributed_Sparse {
public:
    int c;
    int sqrtpc;

    vector<int> nnz_in_row_axis, nnz_in_row_axis_tpose;
    int sparse_shift;

    Sparse25D_Cannon_Dense(SpmatLocal* S_input, int R, int c, KernelImplementation* k) : Distributed_Sparse(k, R) { 
        this->c = c;
        sqrtpc = (int) sqrt(p / c);

        if(proc_rank == 0) {
            if(sqrtpc * sqrtpc * c != p) {
                cout << "Error, for 2.5D algorithm, p / c must be a perfect square!" << endl;
                cout << p << " " << c << endl;
                exit(1);
            }
        }

        algorithm_name = "2.5D Cannon's Algorithm Replicating Dense Matrices";
        proc_grid_names = {"# Rows", "# Cols", "# Layers"};

        perf_counter_keys = 
                {"Dense Cyclic Shift Time",
                 "Sparse Cyclic Shift Time",
                 "Dense Fiber Communication Time",
                 "Computation Time" 
                };

        grid.reset(new FlexibleGrid(sqrtpc, sqrtpc, c, 3));

        A_R_split_world = grid->row_world; 
        B_R_split_world = grid->row_world; 

        localAcols = R / sqrtpc;
        localBcols = R / sqrtpc; 

        if(localAcols * sqrtpc != R) {
            cout << "Error, R must be divisible by sqrt(p) / c!" << endl;
            exit(1);
        }

        r_split = true;

        this->M = S_input->M;
        this->N = S_input->N;
        localArows = divideAndRoundUp(this->M, sqrtpc * c);
        localBrows = divideAndRoundUp(this->N, sqrtpc * c);

        Block_Cyclic25D nonzero_dist(M, N, sqrtpc, c, grid);
        Block_Cyclic25D transpose_dist(N, M, sqrtpc, c, grid);

        S.reset(S_input->redistribute_nonzeros(&nonzero_dist, false, false));
        ST.reset(S_input->redistribute_nonzeros(&transpose_dist, true, false));

        nnz_in_row_axis.resize(sqrtpc);
        nnz_in_row_axis_tpose.resize(sqrtpc);

        int my_nnz = S->coords.size();
        int my_nnz_tpose = ST->coords.size();
        MPI_Allgather(&my_nnz, 1, MPI_INT, nnz_in_row_axis.data(), 
                1, MPI_INT, grid->row_world);

        MPI_Allgather(&my_nnz_tpose, 1, MPI_INT, nnz_in_row_axis_tpose.data(), 
                1, MPI_INT, grid->row_world);

        int max_nnz = *(std::max_element(nnz_in_row_axis.begin(), nnz_in_row_axis.end()));
        int max_nnz_tpose = *(std::max_element(nnz_in_row_axis_tpose.begin(), nnz_in_row_axis_tpose.end()));

        // Define submatrix boundaries 
        aSubmatrices.emplace_back(localArows * (grid->k + c * grid->i), localAcols * grid->j, localArows, localAcols);
        bSubmatrices.emplace_back(localBrows * (grid->k + c * grid->i), localBcols * grid->j, localBrows, localBcols);

        // Skew the S-matrix in preparation for repeated Cannon's algorithm.
        // TODO: Need to deal with the shift!

        int src = pMod(grid->rankInRow + grid->rankInCol, sqrtpc);
        int dst = pMod(grid->rankInRow - grid->rankInCol, sqrtpc);

        sparse_shift = src;
        S->shiftCoordinates(src, dst, grid->row_world, nnz_in_row_axis[src], 0);
        ST->shiftCoordinates(src, dst, grid->row_world, nnz_in_row_axis_tpose[src], 0);

        for(int i = 0; i < S->coords.size(); i++) {
            S->coords[i].r %= localArows * c;
            S->coords[i].c %= localBrows;
        }

        for(int i = 0; i < ST->coords.size(); i++) {
            ST->coords[i].r %= localBrows * c;
            ST->coords[i].c %= localArows;
        }

        S->own_all_coordinates();
        ST->own_all_coordinates();

        S->monolithBlockColumn();
        ST->monolithBlockColumn();

	    S->initializeCSRBlocks(localArows * c, localBrows, max_nnz, true);
	    ST->initializeCSRBlocks(localBrows * c, localArows, max_nnz_tpose, true);

        check_initialized(); 
    }

    void initial_synchronize(DenseMatrix *localA, DenseMatrix *localB, VectorXd *SValues) {
        if(localA != nullptr) {
            shiftDenseMatrix(*localA, grid->col_world, 
                    pMod(grid->rankInCol - grid->rankInRow, sqrtpc), 1);
        }

        if(localB != nullptr) {
            shiftDenseMatrix(*localB, grid->col_world, 
                    pMod(grid->rankInCol - grid->rankInRow, sqrtpc), 2);
        }
    }

    void algorithm(     DenseMatrix &localA, 
                        DenseMatrix &localB, 
                        VectorXd &SValues, 
                        VectorXd *sddmm_result_ptr, 
                        KernelMode mode
                        ) {

        SpmatLocal* choice;

        DenseMatrix *Arole, *Brole;
		DenseMatrix accumulation_buffer;
        vector<int> nnz_in_axis; 

        if(mode == k_spmmA) {
            assert(SValues.size() == ST->owned_coords_end - ST->owned_coords_start);
            choice = ST.get();
            Arole = &localB;
            Brole = &localA; 
		    accumulation_buffer = DenseMatrix::Constant(localBrows * c, localBcols, 0.0); 
            nnz_in_axis = nnz_in_row_axis_tpose;
        } 
        else if(mode == k_spmmB || mode == k_sddmm) {
            assert(SValues.size() == S->owned_coords_end - S->owned_coords_start);
            choice = S.get();
            Arole = &localA;
            Brole = &localB; 
		    accumulation_buffer = DenseMatrix::Constant(localArows * c, localAcols, 0.0);
            nnz_in_axis = nnz_in_row_axis;
        }

        if(mode == k_sddmm) {
            choice->setValuesConstant(0.0);
        }
        else {
            choice->setCSRValues(SValues);
        }

        auto t = start_clock();
        MPI_Allgather(Arole->data(), Arole->size(), MPI_DOUBLE,
                        accumulation_buffer.data(), Arole->size(), MPI_DOUBLE, grid->fiber_world);        
        stop_clock_and_add(t, "Dense Fiber Communication Time");

        for(int i = 0; i < sqrtpc; i++) {
            auto t = start_clock();

            // TODO: THIS NEEDS TO BE FIXED

            kernel->triple_function(
                mode == k_sddmm ? k_sddmm : k_spmmB,
                *choice,
                accumulation_buffer,
                *Brole,
                0);

            stop_clock_and_add(t, "Computation Time");

            if(sqrtpc > 1) {
                t = start_clock();
                shiftDenseMatrix(*Brole, grid->col_world, 
                        pMod(grid->rankInCol + 1, sqrtpc), 1);
                stop_clock_and_add(t, "Dense Cyclic Shift Time");

                t = start_clock();
                int src = pMod(grid->rankInRow - 1, sqrtpc);
                int dst = pMod(grid->rankInRow + 1, sqrtpc); 

                if(mode==k_sddmm) {
                    choice->shiftCoordinates(src, dst, grid->row_world, nnz_in_axis[pMod(sparse_shift - i - 1, sqrtpc)], 72);
                }
                else {
                    choice->csr_blocks[0]->shiftCSR(src, dst, grid->row_world, nnz_in_axis[pMod(sparse_shift - i - 1, sqrtpc)], 72);
                }

                MPI_Barrier(MPI_COMM_WORLD);

                stop_clock_and_add(t, "Sparse Cyclic Shift Time");
            }
        }

        if(mode == k_sddmm) {
            *sddmm_result_ptr = SValues.cwiseProduct(choice->getCoordValues());
        }
    }
};

