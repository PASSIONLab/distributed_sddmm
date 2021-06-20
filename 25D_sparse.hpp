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
#include "io_utils.h"
#include "als_conjugate_gradients.h"
#include "distributed_sparse.h"

using namespace std;
using namespace combblas;

class Sparse25D : public Distributed_Sparse {
public:
    int sqrtpc; // Square grid size on each layer
    int c;      // Number of Layers

    // Communicators and grids
    unique_ptr<CommGrid3D> grid;

    void constructor_helper(bool readFromFile, 
            int logM, 
            int nnz_per_row, 
            string filename, 
            int R, 
            int c) {

        algorithm_name = "2.5D No-Staged Replicating ABS with Allreduce";
        proc_grid_names = {"Block Rows per Layer", "Block Columns per Layer", "# Layers"};

        this->R = R;
        this->c = c;
        this->nnz_per_row = nnz_per_row;

        // STEP 1: Make sure the replication factor is valid for the number
        // of processes running
        sqrtpc = (int) round(sqrt(p / c));

        proc_grid_dimensions = {sqrtpc, sqrtpc, c};

        if(proc_rank == 0) {
            if(sqrtpc * sqrtpc * c != p) {
                cout << "Error, for 2.5D algorithm, p / c must be a perfect square!" << endl;
                cout << p << " " << c << endl;
                exit(1);
            }
            if(R / c * c != R) {
                cout << "Error, R-dimension must be divisble by C!" << endl;
                exit(1);
            }
            if(R / c < 8) {
                cout << "Error, can't cut the R-dimension smaller than 8!" << endl;
                exit(1);
            }
        }

        perf_counter_keys = {"Dense Allreduction Time", "Sparse Allreduction Time", "Computation Time"};

        // Step 2: Create a communication grid
        grid.reset(new CommGrid3D(MPI_COMM_WORLD, c, sqrtpc, sqrtpc));

        // Step 3: Use either the R-mat generator or a file reader to get a sparse matrix. 

        if(grid->GetRankInFiber() == 0) {
            if(! readFromFile) {
                generateRandomMatrix(logM, nnz_per_row,
                    grid->GetCommGridLayer(),
                    S,
                    input_Svalues 
                );

                if(proc_rank == 0) {
                    cout << "R-mat generator created " << S.dist_nnz << " nonzeros." << endl;
                }
            }
            else {
                loadMatrixFromFile(filename, grid->GetCommGridLayer(), S, input_Svalues);
                if(proc_rank == 0) {
                    cout << "File reader read " << S.dist_nnz << " nonzeros." << endl;
                }
            }
            this->M = S.distrows;
            this->N = S.distcols;
        }

        // Step 4: broadcast nonzero counts across fibers, allocate the SpMat arrays 
        MPI_Bcast(&(this->M), 1, MPI_INT, 0, grid->GetFiberWorld());
        MPI_Bcast(&(this->N), 1, MPI_INT, 0, grid->GetFiberWorld());
        MPI_Bcast(&(S.local_nnz), 1, MPI_INT, 0, grid->GetFiberWorld());

        // Step 5: These two steps weren't here earlier... why?
        S.rCoords.resize(S.local_nnz);
        S.cCoords.resize(S.local_nnz);
        input_Svalues.resize(S.local_nnz);

        // Step 6: broadcast the sparse matrices (the coordinates, not the values)
        MPI_Bcast(S.rCoords.data(), S.local_nnz, MPI_UINT64_T, 0, grid->GetFiberWorld());
        MPI_Bcast(S.cCoords.data(), S.local_nnz, MPI_UINT64_T, 0, grid->GetFiberWorld());

        // Step 7: allocate buffers for the dense matrices; over-allocate the dense matrices,
        // that's ok. Although we do have better tools to do this... see the 1.5D allocation.
        localArows = this->M / sqrtpc + 1;
        localBrows = this->N / sqrtpc + 1;
        localAcols = this->R / c;
        localBcols = this->R / c;

        // Step 8: Indicate which axes the A and B matrices are split along 
        A_R_split_world = grid->GetFiberWorld();
        B_R_split_world = grid->GetFiberWorld();

        check_initialized();
    }

    Sparse25D(int logM, int nnz_per_row, int R, int c, KernelImplementation *k) : 
        Distributed_Sparse(k) { 
        constructor_helper(false, logM, nnz_per_row, "", R, c);
    }

    Sparse25D(string filename, int R, int c, KernelImplementation *k) :
        Distributed_Sparse(k) {
        constructor_helper(true, 0, -1, filename, R, c);
    }

    void initial_synchronize(DenseMatrix *localA, DenseMatrix *localB, VectorXd *SValues) {
        shared_ptr<CommGrid> commGridLayer = grid->GetCommGridLayer();

        if(localA != nullptr) {
            MPI_Bcast((void*) localA->data(), localA->rows() * localA->cols(), MPI_DOUBLE, 0, commGridLayer->GetRowWorld());
        }
        if(localB != nullptr) {
            MPI_Bcast((void*) localB->data(), localB->rows() * localB->cols(), MPI_DOUBLE, 0, commGridLayer->GetColWorld());
        }
        if(SValues != nullptr) {
            MPI_Bcast((void*) SValues->data(), SValues->size(), MPI_DOUBLE,     0, grid->GetFiberWorld());
        }
    }

    void spmmA(DenseMatrix &localA, DenseMatrix &localB, VectorXd &SValues) {
        shared_ptr<CommGrid> commGridLayer = grid->GetCommGridLayer();
        algorithm(localA, localB, SValues, nullptr, k_spmmA);
        auto t = start_clock();
        MPI_Allreduce(MPI_IN_PLACE, localA.data(), localA.size(), MPI_DOUBLE, MPI_SUM, commGridLayer->GetRowWorld());
        stop_clock_and_add(t, "Dense Allreduction Time");
    }

    void spmmB(DenseMatrix &localA, DenseMatrix &localB, VectorXd &SValues) {
        shared_ptr<CommGrid> commGridLayer = grid->GetCommGridLayer();
        algorithm(localA, localB, SValues, nullptr, k_spmmB);
        auto t = start_clock();
        MPI_Allreduce(MPI_IN_PLACE, localB.data(), localB.size(), MPI_DOUBLE, MPI_SUM, commGridLayer->GetColWorld());
        stop_clock_and_add(t, "Dense Allreduction Time");
    }

    void sddmm(DenseMatrix &localA, DenseMatrix &localB, VectorXd &SValues, VectorXd &sddmm_result) { 
        algorithm(localA, localB, SValues, &sddmm_result, k_sddmm);
        auto t = start_clock();
        MPI_Allreduce(MPI_IN_PLACE, SValues.data(), SValues.size(), MPI_DOUBLE, MPI_SUM, grid->GetFiberWorld()); 
        stop_clock_and_add(t, "Sparse Allreduction Time");
    }

    void algorithm( DenseMatrix &localA, 
                    DenseMatrix &localB, 
                    VectorXd &SValues, 
                    VectorXd *sddmm_result_ptr, 
                    KernelMode mode
                    ) {

        int nnz_processed = 0;
        
        // Perform a local SDDMM 
        auto t = start_clock();

        if(mode == k_sddmm) {
            nnz_processed += kernel->sddmm_local(
                S,
                SValues,
                localA,
                localB,
                *sddmm_result_ptr,
                0, 
                S.local_nnz); 
        }
        else if(mode == k_spmmA) {
            nnz_processed += kernel->spmm_local(
                S,
                SValues,
                localA,
                localB,
                Amat,
                0,
                S.local_nnz);
        }
        else if(mode == k_spmmB) {
            nnz_processed += kernel->spmm_local(
                S,
                SValues,
                localA,
                localB,
                Bmat,
                0,
                S.local_nnz);
        }
        stop_clock_and_add(t, "Computation Time");

        // Debugging only: print out the total number of dot products taken, reduce across
        // each layer world as a sanity check 

        int total_processed;
        MPI_Reduce(&nnz_processed, &total_processed, 1, MPI_INT,
                MPI_SUM, 0, grid->GetLayerWorld());

        if(proc_rank == 0 && verbose) {
            cout << "Total Nonzeros Processed: " << total_processed << endl;
        } 
    }
};