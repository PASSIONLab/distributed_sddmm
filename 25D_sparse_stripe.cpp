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

#include "spmat_reader.h"
#include "sddmm.h"

// CombBLAS includes 
#include <memory>
#include "CombBLAS/CombBLAS.h"

using namespace std;
using namespace combblas;

typedef SpParMat < int64_t, int, SpDCCols<int32_t,int> > PSpMat_s32p64_Int;

class Sparse25D {
public:
    // Matrix Dimensions, R is the small inner dimen 
    int M;
    int N;
    int R;

    int p;     // Total number of processes
    int sqrtpc; // Square grid size on each layer
    int c;     // Number of Layers

    int proc_rank;     // Global process rank

    // Communicators and grids
    unique_ptr<CommGrid3D> grid;

    // Parallel coordinate arrays for the sparse matrix
    int64_t local_nnz; 
    vector<int64_t> rCoords;
    vector<int64_t> cCoords;

    // These all relate to the local buffers
    int nrowsA;
    int nrowsB;
    int ncolsLocal;

    double* localA;
    double* localB;
    double* localResult;
    double* recvResultSlice;


    // Performance timers 
    int nruns;
    double broadcast_time;
    double computation_time;
    double reduction_time;

    Sparse25D(int logM, int R, int c, int nnz_per_row) {
        
        /*
         * Vivek: should we rename nnz_per_row to edgefactor, since that interpretation
         * may not be correct for general R-mat matrices/ 
         */

        // Assume square for now 
        this->M = 1 << logM;
        this->N = this->M;
        this->R = R;
        this->c = c;

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

        PSpMat_s32p64_Int * G; 
        if(grid->GetRankInFiber() == 0) {
            DistEdgeList<int64_t> * DEL = new DistEdgeList<int64_t>(grid->GetCommGridLayer());

            double initiator[4] = {0.25, 0.25, 0.25, 0.25};
            assert(abs(initiator[0] + initiator[1] + initiator[2] + initiator[4] - 1.0) < 1e-7);
            unsigned long int scale      = logM;

            DEL->GenGraph500Data(initiator, scale, nnz_per_row);
            PermEdges(*DEL);
            RenameVertices(*DEL);	
            G = new PSpMat_s32p64_Int(*DEL, false);

            int64_t total_nnz = G->getnnz(); 
            if(proc_rank == 0) {
                cout << "Total Nonzeros Generated: " << total_nnz << endl;
            }

            local_nnz = G->seq().getnnz();
            rowAwidth = G->seq().getnrow();
            delete DEL;
        }

        // Step 4: broadcast nonzero counts across fibers, allocate SpMat arrays 
        MPI_Bcast(&local_nnz, 1, MPI_INT, 0, grid->GetFiberWorld());
        MPI_Bcast(&rowAwidth, 1, MPI_UINT64_T, 0, grid->GetFiberWorld());
        rowBwidth = (int) ceil((float) N  * c / p);

        rCoords.resize(local_nnz);
        cCoords.resize(local_nnz);

        // Step 5: Sort and unpack the coordinates on the lowest level of the grid 
        if(grid->GetRankInFiber() == 0) {
            SpTuples<int64_t,int> tups(G->seq()); 
            tups.SortColBased();

            tuple<int64_t, int64_t, int>* values = tups.tuples;  
            
            for(int i = 0; i < tups.getnnz(); i++) {
                rCoords[i] = get<0>(values[i]);
                cCoords[i] = get<1>(values[i]); // So that we have valid indexing 
            }
            delete G;
        }

        // Step 6: broadcast the sparse matrices (the coordinates, not the values)
        MPI_Bcast(rCoords.data(), local_nnz, MPI_UINT64_T, 0, grid->GetFiberWorld());
        MPI_Bcast(cCoords.data(), local_nnz, MPI_UINT64_T, 0, grid->GetFiberWorld());

        // Step 7: allocate buffers for the dense matrices; over-allocate the dense matrices,
        // that's ok.
        nrowsA = this->M / sqrtpc + 1;
        nrowsB = this->N / sqrtpc + 1;
        ncolsLocal = this->R / c;

        localA          = new double[nrowsA * ncolsLocal]; 
        localB          = new double[nrowsB * ncolsLocal];
        localResult     = new double[local_nnz]; 
        recvResultSLice = new double[local_nnz]; 

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

    chrono::time_point<std::chrono::steady_clock> start_clock() {
        return std::chrono::steady_clock::now();
    }

    void stop_clock_and_add(chrono::time_point<std::chrono::steady_clock> &start, double* timer) {
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> diff = end - start;
        *timer += diff.count();
    }

    /*
     * I guess it's best to describe this as Johnson's algorithm. 
     */
    void algorithm() {
        int nnz_processed = 0;

        // Assume that the matrices begin distributed across two faces of the 3D grid,
        // but not distributed yet

        auto t = start_clock();
        MPI_Bcast((void*) Aslice, local_nnz, MPI_DOUBLE, 0, grid->GetLayerWorld());
        stop_clock_and_add(t, &broadcast_time);

        // Perform a local SDDMM 
        t = start_clock();

        stop_clock_and_add(t, &computation_time);

        // Reduction across layers (fiber world)
        t = start_clock();
        MPI_Reduce(result, recvResultSlice, local_nnz, MPI_DOUBLE,
                MPI_SUM, 0, grid->GetFiberWorld());
        stop_clock_and_add(t, &reduction_time);

        // Debugging only: print out the total number of dot products taken, but reduce 
        // across the layer world 
        int total_processed;
        MPI_Reduce(&nnz_processed, &total_processed, 1, MPI_INT,
                MPI_SUM, 0, grid->GetLayerWorld());

        if(proc_rank == 0) {
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
            sum_broadcast_time /= p * nruns;
            sum_comp_time      /= p * nruns;
            sum_reduce_time    /= p * nruns;
            //cout << "Average Broadcast Time: " << sum_broadcast_time<< endl;
            //cout << "Average Cyclic Shift Time: " << sum_shift_time << endl;
            //cout << "Average Computation Time:   " << sum_comp_time << endl;
            //cout << "Average Reduction Time: " << sum_reduce_time << endl;
            cout << "Aggregate: " 
            << sum_broadcast_time << " "
            << sum_comp_time << " "
            << sum_reduce_time << endl;
        }
    }

    void benchmark() {
        reset_performance_timers();

        int nruns = 10;
        for(int i = 0; i < nruns; i++) {
            algorithm();
        }
        print_statistics();
    }

    ~Sparse25D() {
        delete[] localA;
        delete[] localB;
        delete[] localResult;
        delete[] recvResultSlice;
    }

}