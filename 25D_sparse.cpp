#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <utility>
#include <vector>
#include <string.h>
#include <vector>
#include <utility>
#include <random>
#include <fstream>
#include <cassert>
#include <mpi.h>
#include <cblas.h>
#include <algorithm>

#include "sddmm.h"
#include "common.h"
#include "io_utils.h"

// CombBLAS includes 
#include <memory>
#include "CombBLAS/CombBLAS.h"

using namespace std;
using namespace combblas;

class Sparse25D {
public:
    // Matrix Dimensions, R is the small inner dimension
    int M, N, R;

    int nnz_per_row;

    int p;      // Total number of processes
    int sqrtpc; // Square grid size on each layer
    int c;      // Number of Layers

    int proc_rank;     // Global process rank

    // Communicators and grids
    unique_ptr<CommGrid3D> grid;

    // Parallel coordinate arrays for the sparse matrix
    int64_t local_nnz; 
    vector<int64_t> rCoords, cCoords;

    // These are the local dense matrix buffers (first two)
    // and the buffer for the local nonzeros 
    int nrowsA, nrowsB, ncolsLocal;

    VectorXd Svalues, result;
    DenseMatrix localA, localB;

    // Performance timers 
    int nruns;
    double broadcast_time;
    double computation_time;
    double reduction_time;

    Sparse25D(int logM, int nnz_per_row, int R, int c) {
        
        /*
         * Vivek: should we rename nnz_per_row to edgefactor, since that interpretation
         * may not be correct for general R-mat matrices/ 
         */

        // Assume square matrix for now... 
        this->M = 1 << logM;
        this->N = this->M;
        this->R = R;
        this->c = c;
        this->nnz_per_row = nnz_per_row;

        MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &p);

        // STEP 1: Make sure the replication factor is valid for the number
        // of processes running
        sqrtpc = (int) sqrt(p / c);

        if(proc_rank == 0) {
            if(sqrtpc * sqrtpc * c != p) {
                cout << "Error, for 2.5D algorithm, p / c must be a perfect square!" << endl;
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

        // Step 2: Create a communication grid
        grid.reset(new CommGrid3D(MPI_COMM_WORLD, c, sqrtpc, sqrtpc));

        // Step 3: Use the R-Mat generator to create a distributed
        // edge list. Only the bottom-most layer needs to do the
        // generation, we can broadcast it to everybody else

        if(grid->GetRankInFiber() == 0) {
            int total_nnz;
            generateRandomMatrix(logM, nnz_per_row,
                grid->GetCommGridLayer(),
                &total_nnz,
                &rCoords,
                &cCoords,
                &Svalues 
            );
            local_nnz = Svalues.size();
            if(proc_rank == 0) {
                cout << "Generated " << total_nnz << " nonzeros." << endl;
            }
        }

        // Step 4: broadcast nonzero counts across fibers, allocate SpMat arrays 
        MPI_Bcast(&local_nnz, 1, MPI_INT, 0, grid->GetFiberWorld());

        // Step 6: broadcast the sparse matrices (the coordinates, not the values)
        MPI_Bcast(rCoords.data(), local_nnz, MPI_UINT64_T, 0, grid->GetFiberWorld());
        MPI_Bcast(cCoords.data(), local_nnz, MPI_UINT64_T, 0, grid->GetFiberWorld());

        // Step 7: allocate buffers for the dense matrices; over-allocate the dense matrices,
        // that's ok. Although we do have better tools to do this... see the 1.5D allocation.
        nrowsA = this->M / sqrtpc + 1;
        nrowsB = this->N / sqrtpc + 1;
        ncolsLocal = this->R / c;

        new (&localA) DenseMatrix(nrowsA, ncolsLocal);
        new (&localB) DenseMatrix(nrowsB, ncolsLocal);
        new (&result) VectorXd(local_nnz);
    }

    void reset_performance_timers() {
        nruns = 0;
        broadcast_time = 0;
        computation_time = 0;
        reduction_time = 0;
        if(proc_rank == 0) {
            cout << "Performance timers reset..." << endl;
        }
    }

    /*
     * Hmmm... this looks very similar to Johnson's algorithm. What's the difference between
     * this and Edgar's 2.5D algorithms? 
     */
    void algorithm(bool verbose) {
        int nnz_processed = 0;
        nruns++;

        if(proc_rank == 0 && verbose) {
            cout << "Benchmarking 2.5D Replicating ABC Algorithm..." << endl;
            cout << "Matrix Dimensions: " 
            << this->M << " x " << this->N << endl;
            cout << "Nonzeros Per row: " << nnz_per_row << endl;
            cout << "R-Value: " << this->R << endl;
            
            cout << "Grid Dimensions: "
            << sqrtpc << " x " << sqrtpc << " x " << c << endl;
        }

        // Assume that the matrices begin distributed across two faces of the 3D grid,
        // but not distributed yet

        shared_ptr<CommGrid> commGridLayer = grid->GetCommGridLayer();

        auto t = start_clock();
        MPI_Bcast((void*) localA.data(), localA.rows() * localA.cols(), MPI_DOUBLE, 0, commGridLayer->GetRowWorld());
        MPI_Bcast((void*) localB.data(), localB.rows() * localB.cols(), MPI_DOUBLE, 0, commGridLayer->GetColWorld());
        stop_clock_and_add(t, &broadcast_time);

        // Perform a local SDDMM 
        t = start_clock();
        nnz_processed += kernel(rCoords.data(),
            cCoords.data(),
            localA.data(),
            localB.data(),
            ncolsLocal,
            result.data(),
            0, 
            local_nnz); 
        stop_clock_and_add(t, &computation_time);

        // Reduction across layers (fiber world)
        t = start_clock();

        void* sendBuf, *recvBuf;
        if(grid->GetRankInFiber() == 0) {
            sendBuf = MPI_IN_PLACE;
            recvBuf = result.data();
        }
        else {
            sendBuf = result.data();
            recvBuf = NULL;
        }
        MPI_Reduce(sendBuf, recvBuf, result.size(), MPI_DOUBLE,
                MPI_SUM, 0, grid->GetFiberWorld());

        stop_clock_and_add(t, &reduction_time);

        // Debugging only: print out the total number of dot products taken, but reduce 
        // across the layer world 
        int total_processed;
        MPI_Reduce(&nnz_processed, &total_processed, 1, MPI_INT,
                MPI_SUM, 0, grid->GetLayerWorld());

        if(proc_rank == 0 && verbose) {
            cout << "Total Nonzeros Processed: " << total_processed << endl;
        } 
    }

    void print_statistics() {
        double sum_broadcast_time, sum_comp_time, sum_reduce_time;

        MPI_Allreduce(&broadcast_time, &sum_broadcast_time, 1,
                    MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        MPI_Allreduce(&computation_time, &sum_comp_time, 1,
                    MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        MPI_Allreduce(&reduction_time, &sum_reduce_time, 1,
                    MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        if(proc_rank == 0) {
            cout << "Avg. Broadcast Time\t"
                << "Avg. Computation Time\t"
                << "Avg. Reduction Time" << endl;

            sum_broadcast_time /= p * nruns;
            sum_comp_time      /= p * nruns;
            sum_reduce_time    /= p * nruns;
            cout 
            << sum_broadcast_time << "\t"
            << sum_comp_time << "\t"
            << sum_reduce_time << endl;
            cout << "=================================" << endl;
        }
    }

    void benchmark() {
        algorithm(true);

        reset_performance_timers();

        int nruns = 10;
        for(int i = 0; i < nruns; i++) {
            algorithm(false);
        }
        print_statistics();
    }

    ~Sparse25D() {
        // Destructor 
    }
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int proc_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

    int num_procs;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Arguments:
    // 1. Log of side length of sparse matrix
    // 2. NNZ per row
    // 3. R-Dimension Length
    // 4. Replication factor

    Sparse25D x(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]));
    x.benchmark();
}