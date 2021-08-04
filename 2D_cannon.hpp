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

        this->M = S_input->M;
        this->M = S_input->N;
        Standard2D nonzero_dist(M, N, sqrtp);

        S.reset(S_input->redistribute_nonzeros(&nonzero_dist, false, false));

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
    void shiftSparseMatrix(VectorXd &Svalues, VectorXd *sddmm_result) {
        int nnz_to_send, nnz_to_receive;
        nnz_to_send = S->coords.size();

        int dst = pMod(rankInRow + 1, sqrtp);

        MPI_Status stat;
        auto t = start_clock();

        // Send the buffer sizes; we can definitely optimize this operation. 
        MPI_Sendrecv(&nnz_to_send, 1, MPI_INT,
                dst, 0,
                &nnz_to_receive, 1, MPI_INT,
                MPI_ANY_SOURCE, 0,
                row_axis, &stat);

        VectorXd Svalues_recv(nnz_to_receive);
        vector<spcoord_t> coords_recv;
        coords_recv.resize(nnz_to_receive);

        /*
        * To-do: we can do an MPI_allgather at initialization to avoid sending
        * the number of coordinates first.
        */

        MPI_Sendrecv(Svalues.data(), nnz_to_send, MPI_DOUBLE,
                dst, 0,
                Svalues_recv.data(), nnz_to_receive, MPI_DOUBLE,
                MPI_ANY_SOURCE, 0,
                row_axis, &stat);
        Svalues = Svalues_recv;

        if(sddmm_result != nullptr) {
            VectorXd sddmm_result_recv(nnz_to_receive);
            MPI_Sendrecv(sddmm_result->data(), nnz_to_send, MPI_DOUBLE,
                    dst, 0,
                    sddmm_result_recv.data(), nnz_to_receive, MPI_DOUBLE,
                    MPI_ANY_SOURCE, 0,
                    row_axis, &stat);

            *sddmm_result = sddmm_result_recv;
        }

        MPI_Sendrecv(S->coords.data(), nnz_to_send, MPI_DOUBLE,
                dst, 0,
                coords_recv.data(), nnz_to_receive, MPI_DOUBLE,
                MPI_ANY_SOURCE, 0,
                row_axis, &stat);
        S->coords = coords_recv; 

        stop_clock_and_add(t, "Cyclic Shift Time");
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

        DenseMatrix recvBuffer(localB.rows(), localB.cols());
        int nnz_processed;

        for(int i = 0; i < sqrtp; i++) {
            nnz_processed += kernel->triple_function(
                mode,
                *S,
                SValues,
                localA,
                localB,
                sddmm_result_ptr,
                0,
                S->coords.size());
        }

        shiftDenseMatrix(localB, recvBuffer);
        shiftSparseMatrix(SValues, sddmm_result_ptr);
    }
};

