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
class Block_Cyclic25D : public NonzeroDistribution {
public:
    int sqrtpc;
    int c;

    Block_Cyclic25D(int M, int N, int sqrtpc, int c) {
        world = MPI_COMM_WORLD;
        this->sqrtpc = sqrtpc;
        this->c = c;
        
        rows_in_block = divideAndRoundUp(M, sqrtpc);
        cols_in_block = divideAndRoundUp(N, sqrtpc * c);
    }

	int blockOwner(int row_block, int col_block) {
        return (col_block % c) * sqrtpc * sqrtpc + row_block * sqrtpc +
            col_block / c;
    }
};

class Sparse25D_Cannon_Dense : public Distributed_Sparse {
public:
    shared_ptr<CommGrid3D> grid;

    int c;
    int sqrtpc;
    int rankInRow, rankInCol, rankInFiber;

    MPI_Comm row_axis, col_axis, fiber_axis;

    vector<int> nnz_in_row_axis;

    void print_nonzero_distribution(DenseMatrix &localA, DenseMatrix &localB) {
        if(proc_rank == 0) {
            cout << "===============================" << endl;
            cout << "===============================" << endl;
        }

        for(int i = 0; i < p; i++) {
            if(proc_rank == i) {
                cout << "Process " << i << ":" << endl;
                cout << "Rank in Row: " << rankInRow << endl;
                cout << "Rank in Column: " << rankInCol << endl;
                cout << "Rank in Fiber: " << rankInFiber << endl;

                for(int j = 0; j < S->coords.size(); j++) {
                    cout << S->coords[j].string_rep() << endl;
                }

                cout << "==================" << endl;
                cout << "A matrix: " << endl;
                cout << localA << endl; 
                cout << "==================" << endl;

                cout << "==================" << endl;
                cout << "B matrix: " << endl;
                cout << localB << endl; 
                cout << "==================" << endl;
            }

            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    // Debugging function to deterministically initialize the A and B matrices.
    void dummyInitialize(DenseMatrix &loc) {
        int firstRow = loc.rows() * rankInCol;
        int firstCol = loc.cols() * (c * rankInFiber + rankInRow); 
        for(int i = 0; i < loc.rows(); i++) {
            for(int j = 0; j < loc.cols(); j++) {
                loc(i, j) = (firstRow + i) * R + firstCol + j;
            }
        }
    }

    Sparse25D_Cannon_Dense(SpmatLocal* S_input, int R, int c, KernelImplementation* k) : Distributed_Sparse(k, R) { 
        this->c = c;
        int sqrtpc = (int) sqrt(p / c);

        proc_grid_dimensions = {sqrtpc, sqrtpc, c};

        if(proc_rank == 0) {
            if(sqrtpc * sqrtpc * c != p) {
                cout << "Error, for 2.5D algorithm, p / c must be a perfect square!" << endl;
                cout << p << " " << c << endl;
                exit(1);
            }
        }

        algorithm_name = "2.5D Cannon's Algorithm Replicating Dense Matrices";
        proc_grid_names = {"# Rows", "# Cols", "# Layers"};
        proc_grid_dimensions = {sqrtpc, sqrtpc, c};

        perf_counter_keys = 
                {"Dense Cyclic Shift Time",
                 "Sparse Cyclic Shift Time",
                 "Dense Fiber Communication Time",
                 "Computation Time" 
                };

        grid.reset(new CommGrid3D(MPI_COMM_WORLD, c, sqrtpc, sqrtpc));
        rankInRow = grid->GetCommGridLayer()->GetRankInProcRow();
        rankInCol = grid->GetCommGridLayer()->GetRankInProcCol();
        rankInFiber = grid->GetRankInFiber();

        row_axis   = grid->GetCommGridLayer()->GetRowWorld();
        col_axis   = grid->GetCommGridLayer()->GetColWorld();
        fiber_axis = grid->GetFiberWorld();

        A_R_split_world = row_axis;
        B_R_split_world = row_axis;

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

        Block_Cyclic25D nonzero_dist(M, N, sqrtpc, c);
        S.reset(S_input->redistribute_nonzeros(&nonzero_dist, false, false));

        nnz_in_row_axis.resize(sqrtpc);
        int my_nnz = S->coords.size();
        MPI_Allgather(&my_nnz, 
                1, 
                MPI_INT, 
                nnz_in_row_axis.data(), 
                1,
                MPI_INT,
                row_axis
                );

        int dst = pMod(rankInRow - rankInCol, sqrtpc);
        // Skew the S-matrix in preparation for repeated Cannon's algorithm. 
        shiftSparseMatrix(row_axis, dst, nnz_in_row_axis[dst]);

        for(int i = 0; i < S->coords.size(); i++) {
            //S->coords[i].r %= localArows;
            //S->coords[i].c %= localBrows;
        }

        check_initialized();    
    }

    void initial_synchronize(DenseMatrix *localA, DenseMatrix *localB, VectorXd *SValues) { 
        shiftDenseMatrix(*localB, col_axis, 
                pMod(rankInCol - rankInRow, sqrtpc));
    }

    void algorithm(         DenseMatrix &localA, 
                            DenseMatrix &localB, 
                            VectorXd &SValues, 
                            VectorXd *sddmm_result_ptr, 
                            KernelMode mode
                            ) {

        int nnz_processed; 

        // TODO: Can eliminate some unecessary copies here... 
        if(mode != k_sddmm) {
            S->setValues(SValues);
        }

		DenseMatrix accumulation_buffer = DenseMatrix::Constant(localArows * c, localAcols, 0.0); 

        if(mode == k_spmmB || mode == k_sddmm) {
            auto t = start_clock();
            MPI_Allgather(localA.data(), localA.size(), MPI_DOUBLE,
                            accumulation_buffer.data(), localA.size(), MPI_DOUBLE, fiber_axis);
            stop_clock_and_add(t, "Dense Broadcast Time");
        }

        for(int i = 0; i < sqrtpc; i++) {
            auto t = start_clock();
            nnz_processed += kernel->triple_function(
                mode,
                *S,
                accumulation_buffer,
                localB,
                0,
                S->coords.size()); 
            stop_clock_and_add(t, "Computation Time");

            if(sqrtpc > 1) {
                t = start_clock();
                shiftDenseMatrix(localB, col_axis, 
                        pMod(rankInCol + 1, sqrtpc));
                stop_clock_and_add(t, "Dense Cyclic Shift Time");

                t = start_clock();
                int dst = pMod(rankInRow + 1, sqrtpc);
                shiftSparseMatrix(row_axis, dst, nnz_in_row_axis[dst]);
                stop_clock_and_add(t, "Sparse Cyclic Shift Time");
            }
        }

        // TODO: Can eliminate some unecessary copies here... 
        if(mode == k_sddmm) {
            *sddmm_result_ptr = SValues.cwiseProduct(S->getValues());
        }

        if(mode == k_spmmA) {
            auto t = start_clock(); 

            vector<int> recvCounts;
            for(int i = 0; i < c; i++) {
                recvCounts.push_back(localArows * localAcols);
            }

            MPI_Reduce_scatter(accumulation_buffer.data(), 
                    localA.data(), recvCounts.data(),
                       MPI_DOUBLE, MPI_SUM, fiber_axis);
            stop_clock_and_add(t, "Dense Fiber Communication Time");
        }
    }
};

