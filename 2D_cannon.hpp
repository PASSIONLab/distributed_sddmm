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
class Standard2D: public NonzeroDistribution {
public:
    int sqrtp;

    Standard2D(int M, int N, int sqrtp) {
        world = MPI_COMM_WORLD;
        this->sqrtp = sqrtp;
        
        rows_in_block = divideAndRoundUp(M, sqrtp);
        cols_in_block = divideAndRoundUp(N, sqrtp);
    }

	int blockOwner(int row_block, int col_block) {
        return sqrtp * row_block + col_block;
    }
};

/*
 * This version operates by shifting  
 */
class Sparse2D_Cannon : public Distributed_Sparse {
public:
    shared_ptr<CommGrid> grid;

    int sqrtp;

    int rankInRow, rankInCol;
    MPI_Comm row_axis, col_axis;

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
        int firstCol = loc.cols() * rankInRow; 
        for(int i = 0; i < loc.rows(); i++) {
            for(int j = 0; j < loc.cols(); j++) {
                loc(i, j) = (firstRow + i) * R + firstCol + j;
            }
        }
    }

    Sparse2D_Cannon(SpmatLocal* S_input, int R, KernelImplementation* k) : Distributed_Sparse(k, R) { 
        sqrtp = (int) sqrt(p);

        if(sqrtp * sqrtp != p) {
            if(proc_rank == 0) {
                cout << "Error: for 2D algorithms, number of processes must be a square!" << endl; 
            }
        }

        algorithm_name = "2D Cannon's Algorithm";
        proc_grid_names = {"# Rows", "# Cols"};
        proc_grid_dimensions = {sqrtp, sqrtp};

        perf_counter_keys = 
                {"Dense Cyclic Shift Time",
                 "Sparse Cyclic Shift Time",
                "Computation Time" 
                };

        grid.reset(new CommGrid(MPI_COMM_WORLD, sqrtp, sqrtp));
        rankInRow = grid->GetRankInProcRow();
        rankInCol = grid->GetRankInProcCol();
        row_axis = grid->GetRowWorld();
        col_axis = grid->GetColWorld();

        A_R_split_world = row_axis;
        B_R_split_world = row_axis;

        localAcols = R / sqrtp;
        localBcols = R / sqrtp; 

        if(localAcols * sqrtp != R) {
            cout << "Error, R must be divisible by sqrt(p)!" << endl;
            exit(1);
        }

        r_split = true;

        this->M = S_input->M;
        this->N = S_input->N;
        localArows = divideAndRoundUp(this->M, sqrtp);
        localBrows = divideAndRoundUp(this->N, sqrtp);

        Standard2D nonzero_dist(M, N, sqrtp);
        S.reset(S_input->redistribute_nonzeros(&nonzero_dist, false, false));

        nnz_in_row_axis.resize(sqrtp);
        int my_nnz = S->coords.size();
        MPI_Allgather(&my_nnz, 
                1, 
                MPI_INT, 
                nnz_in_row_axis.data(), 
                1,
                MPI_INT,
                row_axis
                );


        int dst = pMod(rankInRow - rankInCol, sqrtp);
        // Skew the S-matrix in preparation for repeated Cannon's algorithm. 
        shiftSparseMatrix(row_axis, dst, nnz_in_row_axis[dst]);

        for(int i = 0; i < S->coords.size(); i++) {
            S->coords[i].r %= localArows;
            S->coords[i].c %= localBrows;
        }

        check_initialized();    
    }

    void initial_synchronize(DenseMatrix *localA, DenseMatrix *localB, VectorXd *SValues) { 
        shiftDenseMatrix(*localB, col_axis, 
                pMod(rankInCol - rankInRow, sqrtp));
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

        for(int i = 0; i < sqrtp; i++) {
            /*if((i == 0 && proc_rank == 0) || (i == 1 && rankInRow == 1 && rankInCol == 0)) {
                cout << *sddmm_result_ptr << endl;
                cout << "==================" << endl;
                cout << localA << endl;
                cout << "==================" << endl;
                cout << localB << endl;
                cout << "==================" << endl;
            }*/
            //print_nonzero_distribution(localA, localB);

            /*if(proc_rank == 0) {
                cout << "Starting iteration " << i << endl;
            }*/

            auto t = start_clock();
            nnz_processed += kernel->triple_function(
                mode,
                *S,
                localA,
                localB,
                0,
                S->coords.size()); 
            stop_clock_and_add(t, "Computation Time");

            if(sqrtp > 1) {
                t = start_clock();
                shiftDenseMatrix(localB, col_axis, 
                        pMod(rankInCol + 1, sqrtp));
                stop_clock_and_add(t, "Dense Cyclic Shift Time");


                t = start_clock();
                int dst = pMod(rankInRow + 1, sqrtp);
                shiftSparseMatrix(row_axis, dst, nnz_in_row_axis[dst]);
                stop_clock_and_add(t, "Sparse Cyclic Shift Time");
            }
        }

        // TODO: Can eliminate some unecessary copies here... 
        if(mode == k_sddmm) {
            *sddmm_result_ptr = SValues.cwiseProduct(S->getValues());
        }
    }
};

