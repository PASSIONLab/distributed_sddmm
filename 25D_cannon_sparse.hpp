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
    int sqrtpc;
    int c;

    Floor2D(int M, int N, int sqrtpc, int c) {
        world = MPI_COMM_WORLD;
        this->sqrtpc = sqrtpc;
        this->c = c;

        rows_in_block = divideAndRoundUp(M, sqrtpc);
        cols_in_block = divideAndRoundUp(N, sqrtpc);
    }

	int blockOwner(int row_block, int col_block) {
        return row_block * sqrtpc + col_block;
    }
};

class Sparse25D_Cannon_Sparse : public Distributed_Sparse {
public:
    shared_ptr<CommGrid3D> grid;

    int c;
    int sqrtpc;
    int rankInRow, rankInCol, rankInFiber;

    MPI_Comm row_axis, col_axis, fiber_axis;

    vector<int> nnz_in_row_axis;

    int sparse_shift;

    void print_nonzero_distribution(DenseMatrix &localA, DenseMatrix &localB) {
        for(int i = 0; i < p; i++) {
            if(proc_rank == i) {
                cout << "==================================" << endl;
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
                cout << "B matrix: " << endl;
                cout << localB << endl; 
                cout << "==================================" << endl;
            }

            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    // Debugging function to deterministically initialize the A and B matrices.
    void dummyInitialize(DenseMatrix &loc) {
        /*int firstRow = loc.rows() * (rankInFiber + c * rankInCol);
        int firstCol = loc.cols() * (rankInRow); 
        for(int i = 0; i < loc.rows(); i++) {
            for(int j = 0; j < loc.cols(); j++) {
                loc(i, j) = (firstRow + i) * R + firstCol + j;
            }
        }*/
    }

    Sparse25D_Cannon_Sparse(SpmatLocal* S_input, int R, int c, KernelImplementation* k) : Distributed_Sparse(k, R) { 
        this->c = c;
        sqrtpc = (int) sqrt(p / c);

        proc_grid_dimensions = {sqrtpc, sqrtpc, c};

        if(proc_rank == 0) {
            if(sqrtpc * sqrtpc * c != p) {
                cout << "Error, for 2.5D algorithm, p / c must be a perfect square!" << endl;
                cout << p << " " << c << endl;
                exit(1);
            }
        }

        algorithm_name = "2.5D Cannon's Algorithm Replicating Sparse Matrix";
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

        A_R_split_world = row_axis; // The split worlds need to be fixed! 
        B_R_split_world = row_axis; // This also needs to be fixed! 

        localAcols = R / (sqrtpc * c);
        localBcols = R / (sqrtpc * c); 

        if(localAcols * sqrtpc * c != R) {
            cout << "Error, R must be divisible by sqrt(p) * c!" << endl;
            exit(1);
        }

        r_split = true;

        this->M = S_input->M;
        this->N = S_input->N;
        localArows = divideAndRoundUp(this->M, sqrtpc);
        localBrows = divideAndRoundUp(this->N, sqrtpc);

        /* Distribute the nonzeros, storing them initially on the
         * bottom face of the cuboid and then broadcasting.  
         */
        Floor2D nonzero_dist(M, N, sqrtpc, c);
        S.reset(S_input->redistribute_nonzeros(&nonzero_dist, false, false));
        int num_nnz = S->coords.size();
        MPI_Bcast(&num_nnz, 1, MPI_INT, 0, fiber_axis);
        if(rankInFiber > 0) {
            S->coords.resize(num_nnz);
        } 
        MPI_Bcast(S->coords.data(), S->coords.size(), SPCOORD, 0, fiber_axis);

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

        // Postprocess the coordinates 
        for(int i = 0; i < S->coords.size(); i++) {
            S->coords[i].r %= localArows;
            S->coords[i].c %= localBrows;
        }

        check_initialized(); 
    }

    void initial_synchronize(DenseMatrix *localA, DenseMatrix *localB, VectorXd *SValues) { 
        // TODO: Properly do the initial synchronization!

        //shiftDenseMatrix(*localB, col_axis, 
        //        pMod(rankInCol - rankInRow, sqrtpc));
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

		//DenseMatrix accumulation_buffer = DenseMatrix::Constant(localArows * c, localAcols, 0.0); 

        /*if(mode == k_spmmB || mode == k_sddmm) {
            auto t = start_clock();
            MPI_Allgather(localA.data(), localA.size(), MPI_DOUBLE,
                            accumulation_buffer.data(), localA.size(), MPI_DOUBLE, fiber_axis);
            stop_clock_and_add(t, "Dense Fiber Communication Time");
        }*/

        for(int i = 0; i < sqrtpc; i++) {
            auto t = start_clock();
            /*nnz_processed += kernel->triple_function(
                mode,
                *S,
                accumulation_buffer,
                localB,
                0,
                S->coords.size());*/
            stop_clock_and_add(t, "Computation Time");

            if(sqrtpc > 1) {
                t = start_clock();
                shiftDenseMatrix(localA, row_axis, 
                        pMod(rankInCol + 1, sqrtpc));
                shiftDenseMatrix(localB, row_axis, 
                        pMod(rankInCol + 1, sqrtpc));
                stop_clock_and_add(t, "Dense Cyclic Shift Time");
            }

            // ENTER THE DEBUGGING ZONE =========================== 
            /*for(int i = 0; i < p; i++) {
                if(proc_rank == i) {
                    cout << "Accumulation Buffer: " << endl
                        << accumulation_buffer << endl; 
                }
                MPI_Barrier(MPI_COMM_WORLD);
            }

            if(proc_rank == 0) {
                cout << "Completed round!" << endl;
            }*/
            // END THE DEBUGGING ZONE =========================== 

        }

        // TODO: Can eliminate some unecessary copies here... 
        if(mode == k_sddmm) {
            *sddmm_result_ptr = SValues.cwiseProduct(S->getValues());
        }

        /*if(mode == k_spmmA) {
            auto t = start_clock(); 

            vector<int> recvCounts;
            for(int i = 0; i < c; i++) {
                recvCounts.push_back(localArows * localAcols);
            }

            MPI_Reduce_scatter(accumulation_buffer.data(), 
                    localA.data(), recvCounts.data(),
                       MPI_DOUBLE, MPI_SUM, fiber_axis);
            stop_clock_and_add(t, "Dense Fiber Communication Time");
        }*/
    }
};

