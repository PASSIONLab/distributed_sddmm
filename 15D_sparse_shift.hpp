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


class ShardedBlockRow : public NonzeroDistribution {
public:
    int p, c;
    shared_ptr<FlexibleGrid> grid;

    DenseMatrix accumulation_buffer;

    ShardedBlockRow(int M, int N, int p, int c, shared_ptr<FlexibleGrid> &grid) { 
        world = MPI_COMM_WORLD;
        this->p = p;
        this->c = c;
        this->grid = grid;
        rows_in_block = divideAndRoundUp(M, p); 
        cols_in_block = N; 
    }

	int blockOwner(int row_block, int col_block) {
        int rowRank = row_block / c;
        int layerRank = row_block % c;

        return grid->get_global_rank(rowRank, layerRank, 0);
    }
};


class Sparse15D_Sparse_Shift : public Distributed_Sparse {
public:
    DenseMatrix accumulation_buffer;

    Sparse15D_Sparse_Shift(SpmatLocal* S_input, int R, int c, KernelImplementation* k) 
        : Distributed_Sparse(k) 
    {
        this->c = c;

        if(p % c != 0) {
            if(proc_rank == 0) {
                cout << "Error, for 1.5D algorithm, must have c divide num_procs!" << endl;
                exit(1);
            }
        }

        algorithm_name = "1.5D Block Row Replicated S Striped AB Cyclic Shift";
        proc_grid_names = {"# Rows", "# Layers"};

        perf_counter_keys = 
                {"Dense Broadcast Time",
                "Cyclic Shift Time",
                "Computation Time" 
                };

        grid.reset(new FlexibleGrid(p/c, c, 1, 1));

        r_split = true;

        // TODO: Need to set the R-split world! 

        this->M = S_input->M;
        this->N = S_input->N;

        ShardedBlockRow standard_dist(M, N, p, c, grid);
        ShardedBlockRow transpose_dist(N, M, p, c, grid);

        // Copies the nonzeros of the sparse matrix locally (so we can do whatever
        // we want with them; this does incur a memory overhead)
        S.reset(S_input->redistribute_nonzeros(&standard_dist, false, false));
        ST.reset(S->redistribute_nonzeros(&transpose_dist, true, false));

        localArows = divideAndRoundUp(this->M, p) * p / c;
        localBrows = divideAndRoundUp(this->N, p) * p / c;

        setRValue(R);

        for(int i = 0; i < S->coords.size(); i++) {
            S->coords[i].r %= localArows / p * c;
        }
 
        for(int i = 0; i < ST->coords.size(); i++) {
            ST->coords[i].r %= localBrows / p * c;
        } 

        S->own_all_coordinates();
        ST->own_all_coordinates();

        S->monolithBlockColumn();
        ST->monolithBlockColumn();

        S->initializeCSRBlocks(localArows / p * c, localBrows, -1, false);
        ST->initializeCSRBlocks(localBrows / p * c, localArows, -1, false);
        check_initialized();
    }

    void setRValue(int R) {
        this->R = R;

        localAcols = R * c / p;
        localBcols = R * c / p; 

        if(localAcols * p / c != R) {
            cout << "Error, R must be divisible by p / c!" << endl;
        }

        aSubmatrices.clear();
        bSubmatrices.clear();

        for(int i = 0; i < p; i++) {
            aSubmatrices.emplace_back(localArows / p * c * (grid->j + c * i), 
                    localAcols * grid->i, localArows / p * c, localAcols);
            bSubmatrices.emplace_back(localBrows / p * c * (grid->j + c * i), 
                    localBcols * grid->i, localBrows / p * c, localBcols);
        }
    }

    void initial_shift(DenseMatrix *localA, DenseMatrix *localB, KernelMode mode) {
        // Empty on purpose
    }

    void de_shift(DenseMatrix *localA, DenseMatrix *localB, KernelMode mode) {
        // Empty on purpose
    }


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
            Arole = &localA;
            Brole = &localB;
            choice = S.get();
        } 
        else if(mode == k_spmmB || mode == k_sddmmB) {
            Arole = &localB;
            Brole = &localA;
            choice = ST.get(); 
        }
        else {
            assert(false);
        }

		// Temporary buffer that holds the results of the local ops; this buffer
		// is sharded and then reduced to the local portions of the matrix. 
        if(initial_replicate) {
            auto t = start_clock();
            accumulation_buffer = DenseMatrix::Constant(Brole->rows() * c, R, 0.0); 
            MPI_Allgather(Brole->data(), Brole->size(), MPI_DOUBLE,
                            accumulation_buffer.data(), Brole->size(), MPI_DOUBLE, grid->row_world);
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

            auto t = start_clock();

            // TODO: Need to offset the Arole matrix here by the block id 

            kernel->triple_function(
                mode == k_spmmB ? k_spmmA : mode,
                *choice,
                *Arole,
                accumulation_buffer,
                0,
                0); // TODO: Need to modify the internal offset! 
            stop_clock_and_add(t, "Computation Time"); 

            t = start_clock();
            /*
            if(mode==k_sddmmA || mode==k_sddmmB) {
                choice->csr_blocks[0]->shiftCSR(src, dst, grid->row_world, nnz_in_axis[pMod(sparse_shift - i - 1, sqrtpc)], 72, coo);
                choice->blockStarts[1] = choice->csr_blocks[0]->num_coords;
            }
            else {
                choice->csr_blocks[0]->shiftCSR(src, dst, grid->row_world, nnz_in_axis[pMod(sparse_shift - i - 1, sqrtpc)], 72, csr);
                choice->blockStarts[1] = choice->csr_blocks[0]->num_coords;
            }
            */

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
