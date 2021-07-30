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
    int p;

    Standard2D(int p) {
        this->p = p;
        world = MPI_COMM_WORLD;
    }

	int getOwner(int row, int col, int transpose) {
        int rDim = transpose ? this->N : this->M;
        int cDim = transpose ? this->M : this->N;

        int rows_in_block = divideAndRoundUp(rDim, p) * c; 
        int cols_in_block = divideAndRoundUp(cDim, p); 

        if(transpose) {
            int temp = row;
            row = col;
            col = temp;
        }
        
        int block_row = row / rows_in_block;
        int block_col = col / cols_in_block;

        int rowRank = block_row;
        int layerRank = block_col % c;

        return p / c * layerRank + rowRank;
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

    void print_nonzero_distribution() {
        for(int i = 0; i < p; i++) {
            if(proc_rank == i) {
                cout << "Process " << i << ":" << endl;
                cout << "Rank in Row: " << rankInRow << endl;
                cout << "Rank in Column: " << rankInCol << endl;

                for(int j = 0; j < S->coords.size(); j++) {
                    cout << S->coords[j].string_rep() << endl;
                }
                cout << "==================" << endl;
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    Sparse2D_Cannon(SpmatLocal* S_input, int R, KernelImplementation* k) : Distributed_Sparse(k) {
        this->fused = false;
        this->R = R;

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
                {"Cyclic Shift Time",
                "Computation Time" 
                };

        grid.reset(new CommGrid(MPI_COMM_WORLD, sqrtp, sqrtp));
        rankInRow = grid->GetRankInProcRow();
        rankInCol = grid->GetRankInProcCol();
        row_axis = grid->GetRowWorld();
        col_axis = grid->GetColWorld();

        localAcols = R / sqrtp;
        localBcols = R / sqrtp; 

        if(localAcols * sqrtp != R) {
            cout << "Error, R must be divisible by sqrt(p)!" << endl;
            exit(1);
        }

        r_split = true;

        Standard2D nonzero_dist(p);

        S.reset(S_input->redistribute_nonzeros(&nonzero_dist, false, false));
        this->M = S->M;
        this->N = S->N;

        localArows = divideAndRoundUp(this->M, sqrtp);
        localBrows = divideAndRoundUp(this->N, sqrtp);

        for(int i = 0; i < S->coords.size(); i++) {
            S->coords[i].r %= localArows;
            S->coords[i].c %= localBrows;
        }

        check_initialized();
    }

    void shiftDenseMatrix(DenseMatrix &mat, DenseMatrix &recvBuffer) {
        MPI_Status stat;
        auto t = start_clock();

        MPI_Sendrecv(mat.data(), mat.size(), MPI_DOUBLE,
                pMod(rankInCol + 1, sqrtp), 0,
                recvBuffer.data(), recvBuffer.size(), MPI_DOUBLE,
                MPI_ANY_SOURCE, 0,
                col_axis, &stat);
        stop_clock_and_add(t, "Cyclic Shift Time");

        mat = recvBuffer;
    }

    /*
     * This overwrites data in the local sparse matrix buffer. 
     */
    void shiftSparseMatrix(VectorXd &Svalues) {
        int nnz_to_send, nnz_to_receive;
        nnz_to_send = S->coords.size();

        int dst = pMod(rankInRow + 1, sqrtp);

        MPI_Status stat;
        auto t = start_clock();

        // Send the buffer sizes 
        MPI_Sendrecv(&nnz_to_send, 1, MPI_INT,
                dst, 0,
                &nnz_to_receive, 1, MPI_INT,
                MPI_ANY_SOURCE, 0,
                row_axis, &stat);

        VectorXd Svalues_recv(nnz_to_receive);
        vector<spcoord_t> coords_recv;
        coords_recv.resize(nnz_to_receive);

        /*
         * To-do: we can probably dispatch these requests
         * asynchronously, something to optimize in the near future.
         */

        MPI_Sendrecv(Svalues.data(), nnz_to_send, MPI_DOUBLE,
                dst, 0,
                Svalues_recv.data(), nnz_to_receive, MPI_DOUBLE,
                MPI_ANY_SOURCE, 0,
                row_axis, &stat);

        MPI_Sendrecv(S->coords.data(), nnz_to_send, MPI_DOUBLE,
                dst, 0,
                coords_recv.data(), nnz_to_receive, MPI_DOUBLE,
                MPI_ANY_SOURCE, 0,
                row_axis, &stat);

        stop_clock_and_add(t, "Cyclic Shift Time");

        Svalues = Svalues_recv;
        S->coords = coords_recv; 
    }

    void initial_synchronize(DenseMatrix *localA, DenseMatrix *localB, VectorXd *SValues) {
        // Empty method, 2D Cannon needs no initial synchronization 
    }


    void algorithm(         DenseMatrix &localA, 
                            DenseMatrix &localB, 
                            VectorXd &SValues, 
                            VectorXd *sddmm_result_ptr, 
                            KernelMode mode
                            ) {

        DenseMatrix &recvBuffer(localB.rows(), localB.cols());

        for(int i = 0; i < sqrtp; i++) {
            if(mode == k_sddmm) {
                nnz_processed += kernel->sddmm_local(
                    *S,
                    SValues,
                    localA,
                    localB,
                    *sddmm_result_ptr,
                    0,
                    SValues.size());
            }
            else if(mode == k_spmmA) { 
                nnz_processed += kernel->spmm_local(
                    *S,
                    SValues,
                    localA,
                    localB,
                    Amat,
                    0,
                    SValues.size());
            }
            else if(mode == k_spmmB) {
                nnz_processed += kernel->spmm_local(
                    *S,
                    SValues,
                    localA,
                    localB,
                    Bmat,
                    0,
                    SValues.size());
            }
        }

        shiftDenseMatrix(localB, recvBuffer);
        shiftSparseMatrix(SValues);
    }
}

