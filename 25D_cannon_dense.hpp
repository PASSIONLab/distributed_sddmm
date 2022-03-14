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
    int sqrtpc;

    vector<int> nnz_in_row_axis, nnz_in_row_axis_tpose;
    int sparse_shift;

    DenseMatrix accumulation_buffer;

    Sparse25D_Cannon_Dense(SpmatLocal* S_input, int R, int c, KernelImplementation* k) : Distributed_Sparse(k) { 
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
                 "Computation Time",
                 "Setup Shift Time" 
                };

        grid.reset(new FlexibleGrid(sqrtpc, sqrtpc, c, 3));

        A_R_split_world = grid->row_world; 
        B_R_split_world = grid->row_world; 

        r_split = true;

        this->M = S_input->M;
        this->N = S_input->N;
        localArows = divideAndRoundUp(this->M, sqrtpc * c);
        localBrows = divideAndRoundUp(this->N, sqrtpc * c);

        setRValue(R);

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

        #pragma omp parallel for
        for(int i = 0; i < S->coords.size(); i++) {
            S->coords[i].r %= localArows * c;
            S->coords[i].c %= localBrows;
        }

        #pragma omp parallel for
        for(int i = 0; i < ST->coords.size(); i++) {
            ST->coords[i].r %= localBrows * c;
            ST->coords[i].c %= localArows;
        }

        S->own_all_coordinates();
        ST->own_all_coordinates();

        S->monolithBlockColumn();
        ST->monolithBlockColumn();

	    S->initializeCSRBlocks(localArows * c, localBrows, max_nnz, true);
        vector<spcoord_t>().swap(S->coords);
	    ST->initializeCSRBlocks(localBrows * c, localArows, max_nnz_tpose, true);
        vector<spcoord_t>().swap(ST->coords);

        // Skew the S-matrix in preparation for repeated Cannon's algorithm.
        int src = pMod(grid->rankInRow + grid->rankInCol, sqrtpc);
        int dst = pMod(grid->rankInRow - grid->rankInCol, sqrtpc);

        sparse_shift = src;
        S->csr_blocks[0]->shiftCSR(src, dst, grid->row_world, nnz_in_row_axis[src], 0, both);
        S->blockStarts[1] = S->csr_blocks[0]->num_coords;
        ST->csr_blocks[0]->shiftCSR(src, dst, grid->row_world, nnz_in_row_axis_tpose[src], 0, both);
        ST->blockStarts[1] = ST->csr_blocks[0]->num_coords;

        check_initialized(); 
    }

    void setRValue(int R) {
        this->R = R;

        localAcols = R / sqrtpc;
        localBcols = R / sqrtpc; 

        if(localAcols * sqrtpc != R) {
            cout << "Error, R must be divisible by sqrt(p) / c!" << endl;
            exit(1);
        }

        aSubmatrices.clear();
        bSubmatrices.clear();

        // Define submatrix boundaries 
        aSubmatrices.emplace_back(localArows * (grid->k + c * grid->i), localAcols * grid->j, localArows, localAcols);
        bSubmatrices.emplace_back(localBrows * (grid->k + c * grid->i), localBcols * grid->j, localBrows, localBcols);
    }

    void initial_shift(DenseMatrix *localA, DenseMatrix *localB, KernelMode mode) {
        auto t = start_clock();
        if(mode == k_sddmmA || mode == k_spmmA) {
            if(localA != nullptr) {
                BufferPair aBuf(localA);
                shiftDenseMatrix(aBuf, grid->col_world, 
                        pMod(grid->rankInCol - grid->rankInRow, sqrtpc), 1);
                aBuf.sync_active();

            }
        }
        else if(mode == k_sddmmB || mode == k_spmmB) {
            if(localB != nullptr) {
                BufferPair bBuf(localB);
                shiftDenseMatrix(bBuf, grid->col_world, 
                        pMod(grid->rankInCol - grid->rankInRow, sqrtpc), 2);
                bBuf.sync_active();
            }
        }
        stop_clock_and_add(t, "Setup Shift Time");
    }


    void de_shift(DenseMatrix *localA, DenseMatrix *localB, KernelMode mode) {
        auto t = start_clock();
        if(mode == k_sddmmA || mode == k_spmmA) {
            if(localA != nullptr) {
                BufferPair aBuf(localA);
                shiftDenseMatrix(aBuf, grid->col_world, 
                        pMod(grid->rankInCol + grid->rankInRow, sqrtpc), 1);
                aBuf.sync_active();
            }
        }
        else if(mode == k_sddmmB || mode == k_spmmB) {
            if(localB != nullptr) {
                BufferPair bBuf(localB);
                shiftDenseMatrix(bBuf, grid->col_world, 
                        pMod(grid->rankInCol + grid->rankInRow, sqrtpc), 2);
                bBuf.sync_active();
            }
        }
        stop_clock_and_add(t, "Setup Shift Time");
    }


    VectorXd like_S_values(double value) { 
        return VectorXd::Constant(ST->blockStarts[1], value); 
    }

    VectorXd like_ST_values(double value) { 
        return VectorXd::Constant(S->blockStarts[1], value); 
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
        vector<int> nnz_in_axis; 

        if(mode == k_spmmA || mode == k_sddmmA) {
            assert(SValues.size() == ST->blockStarts[1]); 
            choice = ST.get();
            Arole = &localB;
            Brole = &localA; 
            nnz_in_axis = nnz_in_row_axis_tpose;
        } 
        else if(mode == k_spmmB || mode == k_sddmmB) {
            assert(SValues.size() == S->blockStarts[1]); 
            choice = S.get();
            Arole = &localA;
            Brole = &localB; 
            nnz_in_axis = nnz_in_row_axis;
        }

        BufferPair bBuf(Brole); 

        auto t = start_clock();
        if(mode == k_sddmmA || mode == k_sddmmB) {
            choice->setValuesConstant(0.0);
        }
        else {
            choice->setCSRValues(SValues);
        }
        stop_clock_and_add(t, "Computation Time");

        if(initial_replicate) {
            if(c > 1) {
                t = start_clock();
                accumulation_buffer = DenseMatrix::Constant(Arole->rows() * c, Arole->cols(), 0.0);
                MPI_Allgather(Arole->data(), Arole->size(), MPI_DOUBLE,
                                accumulation_buffer.data(), Arole->size(), MPI_DOUBLE, grid->fiber_world);        
                stop_clock_and_add(t, "Dense Fiber Communication Time");
            }
        }

        for(int i = 0; i < sqrtpc; i++) {
            auto t = start_clock();
            kernel->triple_function(
                mode == k_spmmA ? k_spmmB : mode, // Need to account for SDDMMB here!
                *choice,
                c > 1 ? accumulation_buffer : *Arole,
                *(bBuf.getActive()),
                0,
                localAcols * grid->j 
                );

            stop_clock_and_add(t, "Computation Time");

            if(sqrtpc > 1) {
                t = start_clock();
                shiftDenseMatrix(bBuf, grid->col_world, 
                        pMod(grid->rankInCol + 1, sqrtpc), 1);
                stop_clock_and_add(t, "Dense Cyclic Shift Time");

                t = start_clock();
                int src = pMod(grid->rankInRow - 1, sqrtpc);
                int dst = pMod(grid->rankInRow + 1, sqrtpc); 

                if(mode==k_sddmmA || mode==k_sddmmB) {
                    choice->csr_blocks[0]->shiftCSR(src, dst, grid->row_world, nnz_in_axis[pMod(sparse_shift - i - 1, sqrtpc)], 72, coo);
                    choice->blockStarts[1] = choice->csr_blocks[0]->num_coords;
                }
                else {
                    choice->csr_blocks[0]->shiftCSR(src, dst, grid->row_world, nnz_in_axis[pMod(sparse_shift - i - 1, sqrtpc)], 72, csr);
                    choice->blockStarts[1] = choice->csr_blocks[0]->num_coords;
                }
                MPI_Barrier(MPI_COMM_WORLD);
                stop_clock_and_add(t, "Sparse Cyclic Shift Time");
            }
        }

        bBuf.sync_active();

        t = start_clock();
        if(mode == k_sddmmA || mode == k_sddmmB) {
            *sddmm_result_ptr = SValues.cwiseProduct(choice->getCSRValues());
        }
        stop_clock_and_add(t, "Computation Time");
    }
};

