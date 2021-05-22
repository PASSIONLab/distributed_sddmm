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
#include <algorithm>
#include <Eigen/Dense>

#include "sddmm.h"
#include "pack.h"

// This code implements a 1.5D Sparse Matrix Multiplication Algorithm

// CombBLAS includes 
#include <memory>
#include "CombBLAS/CombBLAS.h"

using namespace std;
using namespace combblas;
using namespace Eigen;

typedef Matrix<double, Dynamic, Dynamic, RowMajor> DenseMatrix;
typedef SpParMat < int64_t, int, SpDCCols<int32_t,int> > PSpMat_s32p64_Int;

class Sparse15D {
public:
    // Matrix Dimensions, R is the short inner dimension
    int M;
    int N;
    int R;

    int nnz_per_row;

    int p; // Total number of processes 
    int c; // Number of Layers

    int proc_rank;     // Global process rank

    // Communicators and grids
    unique_ptr<CommGrid3D> grid;

    // Parallel coordinate arrays for the sparse matrix
    int64_t local_nnz; 
    vector<int64_t> rCoords;
    vector<int64_t> cCoords; 
    vector<int64_t> blockStarts;

    // Values of the sparse matrix, and the final result
    // of the calculation 
    VectorXd Svalues; 
    VectorXd result;

    DenseMatrix localA;
    DenseMatrix localB;
    
    // Local dimensions
    int64_t localSrows;
    int64_t localBrows;

    // Performance timers 
    int nruns;
    double broadcast_time;
    double cyclic_shift_time;
    double computation_time;
    double reduction_time;

    // Initiates the algorithm for a Graph500 benchmark 
    Sparse15D(int logM, int nnz_per_row, int R, int c) {
        this->M = 1 << logM;
        this->N = this->M;
        this->R = R;
        this->c = c;
        this->nnz_per_row = nnz_per_row; 

        MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &p);

        // STEP 1: Make sure the replication factor is valid for the number
        // of processes running
        if(p % (c * c) != 0) {
            if(proc_rank == 0) {
                cout << "Error, for 1.5D algorithm, must have c^2 divide num_procs!" << endl;
                exit(1);
            }
        }

        // Step 2: Create a communication grid
        grid.reset(new CommGrid3D(MPI_COMM_WORLD, c, p / c, 1));

        // Step 3: Use the R-Mat generator to create a distributed
        // edge list. Only the bottom-most layer needs to do the
        // generation, we can broadcast it to everybody else

        PSpMat_s32p64_Int * G; 
        if(grid->GetRankInFiber() == 0) {
            DistEdgeList<int64_t> * DEL = new DistEdgeList<int64_t>(grid->GetCommGridLayer());

            double initiator[4] = {0.25, 0.25, 0.25, 0.25};
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
            localSrows = G->seq().getnrow();
            delete DEL;
        }

        // Step 4: broadcast nonzero counts across fibers, allocate SpMat arrays 
        MPI_Bcast(&local_nnz, 1, MPI_INT, 0, grid->GetFiberWorld());
        MPI_Bcast(&localSrows, 1, MPI_UINT64_T, 0, grid->GetFiberWorld());
        localBrows = (int) ceil((float) N  * c / p);

        rCoords.resize(local_nnz);
        cCoords.resize(local_nnz);

        // Step 5: sort the coordinates
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

        // Step 5: broadcast the sparse matrices (the coordinates, not the values)
        MPI_Bcast(rCoords.data(), local_nnz, MPI_UINT64_T, 0, grid->GetFiberWorld());
        MPI_Bcast(cCoords.data(), local_nnz, MPI_UINT64_T, 0, grid->GetFiberWorld());

        // Step 6: Locate block starts within the sparse matrix 
        int currentStart = 0;
        for(int i = 0; i < local_nnz; i++) {
            while(cCoords[i] >= currentStart) {
                blockStarts.push_back(i);
                currentStart += localBrows;
            }

            // So that indexing in the kernel becomes easier...
            cCoords[i] %= localBrows;
        }
        while(blockStarts.size() < p / c + 1) {
            blockStarts.push_back(local_nnz);
        }

        // Step 7: Allocate buffers to receive entries. 
        new (&Svalues) VectorXd(local_nnz);
        new (&localA) DenseMatrix(localSrows, R);
        new (&localB) DenseMatrix(localBrows, R);
        new (&result) VectorXd(local_nnz);

        // Randomly initialize; should find a better way to handle this later...
        for(int i = 0; i < localA.rows() * localA.cols(); i++) {
            localA.data()[i] = rand();
        }
        for(int i = 0; i < localB.rows() * localB.cols(); i++) {
            localB.data()[i] = rand();
        }
    };

    void reset_performance_timers() {
        nruns = 0;
        broadcast_time = 0;
        cyclic_shift_time = 0;
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

    void algorithm(bool verbose) {
        nruns++;
        if(proc_rank == 0 && verbose) {
            cout << "Starting algorithm..." << endl;
        }

        if(proc_rank == 0 && verbose) {
            cout << "Matrix Dimensions: " 
            << this->M << " x " << this->N << endl;
            cout << "Nonzeros Per row: " << nnz_per_row << endl;
            cout << "R-Value: " << this->R << endl;
            
            cout << "Grid Dimensions: "
            << p / c << " x " << c << endl;
        }

        MPI_Status stat;
        MPI_Request send_request;
        MPI_Request recv_request;

        int rankInFiber = grid->GetRankInFiber();
        int rankInLayer = grid->GetRankInLayer();

        int shift = rankInFiber * p / (c * c);
    
        int nnz_processed = 0;

        auto t = start_clock();
        // This assumes that the matrices permanently live in the bottom processor layer 
        MPI_Bcast((void*) Svalues.data(), Svalues.size(), MPI_DOUBLE,     0, grid->GetFiberWorld());

        MPI_Bcast((void*) localA.data(), localA.rows() * localA.cols(), MPI_DOUBLE, 0, grid->GetFiberWorld());
        MPI_Bcast((void*) localB.data(), localB.rows() * localB.cols(), MPI_DOUBLE, 0, grid->GetFiberWorld());
        stop_clock_and_add(t, &broadcast_time);

        // Initial shift

        // Temporary buffer to hold the received portion of matrix B.
        DenseMatrix recvRowSlice(localB.rows(), localB.cols());

        t = start_clock();
        MPI_Isend(localB.data(), localB.rows() * localB.cols(), MPI_DOUBLE, 
                    (rankInLayer + shift) % (p / c), 0,
                    grid->GetLayerWorld(), &send_request);

        MPI_Irecv(recvRowSlice.data(), recvRowSlice.rows() * recvRowSlice.cols(), MPI_DOUBLE, MPI_ANY_SOURCE,
                0, grid->GetLayerWorld(), &recv_request);

        MPI_Wait(&send_request, MPI_STATUS_IGNORE); 
        MPI_Wait(&recv_request, MPI_STATUS_IGNORE);
        stop_clock_and_add(t, &cyclic_shift_time);

        localB = recvRowSlice;

        MPI_Barrier(MPI_COMM_WORLD);

        for(int i = 0; i < p / (c * c); i++) {
            t = start_clock();

            int block_id = ( (p / c) + rankInLayer - shift - i) % (p / c);

            assert(blockStarts[block_id] < local_nnz);
            assert(blockStarts[block_id + 1] <= local_nnz);

            nnz_processed += kernel(rCoords.data(),
                cCoords.data(),
                localA.data(),
                localB.data(),
                R,
                result.data(),
                blockStarts[block_id],
                blockStarts[block_id + 1]);

            stop_clock_and_add(t, &computation_time);

            t = start_clock();
            MPI_Isend(localB.data(), localB.rows() * localB.cols(), MPI_DOUBLE, 
                        ((p/c) + rankInLayer + 1) % (p/c), 0,
                        grid->GetLayerWorld(), &send_request);

            MPI_Irecv(recvRowSlice.data(), recvRowSlice.rows() * recvRowSlice.cols(), MPI_DOUBLE, MPI_ANY_SOURCE,
                    0, grid->GetLayerWorld(), &recv_request);

            MPI_Wait(&send_request, MPI_STATUS_IGNORE); 
            MPI_Wait(&recv_request, MPI_STATUS_IGNORE);
            stop_clock_and_add(t, &cyclic_shift_time);

            localB = recvRowSlice;

            MPI_Barrier(MPI_COMM_WORLD);
        }

        // Reduction across layers
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

        // Debugging only: print out the total number of dot products taken 

        int total_processed;
        MPI_Reduce(&nnz_processed, &total_processed, 1, MPI_INT,
                MPI_SUM, 0, MPI_COMM_WORLD);

        if(proc_rank == 0 && verbose) {
            cout << "Total Nonzeros Processed: " << total_processed << endl;
        }
    }

    void print_statistics() {
        double sum_broadcast_time, sum_comp_time, sum_shift_time, sum_reduce_time;

        MPI_Allreduce(&broadcast_time, &sum_broadcast_time, 1,
                    MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        MPI_Allreduce(&cyclic_shift_time, &sum_shift_time, 1,
                    MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        MPI_Allreduce(&computation_time, &sum_comp_time, 1,
                    MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        MPI_Allreduce(&reduction_time, &sum_reduce_time, 1,
                    MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);


        if(proc_rank == 0) {
            cout << "Avg. Broadcast Time\t"
                << "Avg. Cyclic Shift Time\t"
                << "Avg. Computation Time\t"
                << "Avg. Reduction Time" << endl;

            sum_broadcast_time /= p * nruns;
            sum_shift_time     /= p * nruns;
            sum_comp_time      /= p * nruns;
            sum_reduce_time    /= p * nruns;

            cout 
            << sum_broadcast_time << "\t"
            << sum_shift_time << "\t"
            << sum_comp_time << "\t"
            << sum_reduce_time << endl;
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

        if(proc_rank == 0) {
            cout << "=================================" << endl;
        }
    }

    ~Sparse15D() {
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

    Sparse15D x(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]));
    x.benchmark();
}