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
    int c;
    int sqrtpc;

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

        A_R_split_world = grid->rowfiber_slice; 
        B_R_split_world = grid->rowfiber_slice; 

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

        // Define submatrix boundaries 
        aSubmatrices.emplace_back(localArows * grid->i, localAcols * c * grid->j + grid->k * localAcols, localArows, localAcols);
        bSubmatrices.emplace_back(localBrows * grid->j, localBcols * c * grid->i + grid->k * localBcols, localBrows, localBcols);

        /* Distribute the nonzeros, storing them initially on the
         * bottom face of the cuboid and then broadcasting.  
         */
        Floor2D nonzero_dist(M, N, sqrtpc, c, grid);
        S.reset(S_input->redistribute_nonzeros(&nonzero_dist, false, false));
        int num_nnz = S->coords.size();
        MPI_Bcast(&num_nnz, 1, MPI_INT, 0, grid->fiber_world);
        if(grid->rankInFiber > 0) {
            S->coords.resize(num_nnz);
        } 
        MPI_Bcast(S->coords.data(), S->coords.size(), SPCOORD, 0, grid->fiber_world);

        // Virtually shards the sparse matrix across layers. TODO: Need to do this for the
        // transpose as well!

        vector<int> ccount_in_layer;
        int share = divideAndRoundUp(S->coords.size(), c);
        for(int i = 0; i < S->coords.size(); i += share) {
            if(share < S->coords.size() - i) {
                ccount_in_layer.push_back(share);
            }
            else {
                ccount_in_layer.push_back(S->coords.size() - i);
            }
        }
        owned_coords_start = share * grid->k;
        owned_coords_end = owned_coords_start + ccount_in_layer[grid->k];
        nnz_buffer_size = share;

        // Postprocess the coordinates 
        for(int i = 0; i < S->coords.size(); i++) {
            S->coords[i].r %= localArows;
            S->coords[i].c %= localBrows;
        }

        check_initialized(); 
    }

    void initial_synchronize(DenseMatrix *localA, DenseMatrix *localB, VectorXd *SValues) { 
        shiftDenseMatrix(*localA, grid->row_world, 
                pMod(grid->rankInRow - grid->rankInCol, sqrtpc)); 
        shiftDenseMatrix(*localB, grid->col_world, 
                pMod(grid->rankInCol - grid->rankInRow, sqrtpc));
    }

    void algorithm(         DenseMatrix &localA, 
                            DenseMatrix &localB, 
                            VectorXd &SValues, 
                            VectorXd *sddmm_result_ptr, 
                            KernelMode mode
                            ) {

        int nnz_processed; 

        // TODO: Need to modify if using the transposed matrix! 
		VectorXd accumulation_buffer = VectorXd::Constant(S->coords.size(), 0.0); 

        if(mode != k_sddmm) {
            auto t = start_clock();
            MPI_Allgather(
                SValues.data(),
                SValues.size(),
                MPI_DOUBLE,
                accumulation_buffer.data(),
                nnz_buffer_size,
                MPI_DOUBLE,
                grid->fiber_world
                ); 
            S->setValues(accumulation_buffer);
            stop_clock_and_add(t, "Sparse Fiber Communication Time");

        }

        for(int i = 0; i < sqrtpc; i++) {
            auto t = start_clock();
            nnz_processed += kernel->triple_function(
                mode,
                *S,
                localA,
                localB,
                0,
                S->coords.size());
            stop_clock_and_add(t, "Computation Time");

            if(sqrtpc > 1) {
                t = start_clock();
                shiftDenseMatrix(localA, grid->row_world, 
                        pMod(grid->rankInRow + 1, sqrtpc));
                shiftDenseMatrix(localB, grid->col_world, 
                        pMod(grid->rankInCol + 1, sqrtpc));
                stop_clock_and_add(t, "Dense Cyclic Shift Time");
            }
        }

        if(mode == k_sddmm) {
            accumulation_buffer = S->getValues();
            auto t = start_clock();

            vector<int> temp(c, nnz_buffer_size);
            MPI_Reduce_scatter(accumulation_buffer.data(),
                sddmm_result_ptr->data(),
                temp.data(),
                MPI_DOUBLE,
                MPI_SUM,
                grid->fiber_world
            );
            stop_clock_and_add(t, "Sparse Fiber Communication Time");
            *sddmm_result_ptr = SValues.cwiseProduct(*sddmm_result_ptr);
        }
    }
};

