#include <chrono>
#include <cmath>
#include <iostream>
#include <utility>
#include <vector>
#include <string.h>
#include <vector>
#include <utility>
#include <cassert>
#include <mpi.h>
#include <Eigen/Dense>

#include "sparse_kernels.h"
#include "common.h"
#include "io_utils.h"
#include "pack.h"

// This code implements a 1.5D Sparse Matrix Multiplication Algorithm

/*
 * Clients are responsible for allreducing their results and keeping
 * matrices in sync. Data fields are exposed public to permit this. 
 */

// CombBLAS includes 
#include <memory>
#include "CombBLAS/CombBLAS.h"

using namespace std;
using namespace combblas;
using namespace Eigen;

class Sparse15D {
public:
    // Matrix Dimensions, R is the short inner dimension
    int M, N, R;
    int nnz_per_row;

    int p, c; // Total number of processes, number of layers

    int proc_rank;     // Global process rank

    // Communicators and grids
    unique_ptr<CommGrid3D> grid;

    spmat_local_t S;
    vector<int64_t> blockStarts;
 
    // Local dimensions
    int64_t localSrows;
    int64_t localBrows;

    // Performance timers 
    int nruns;
    double  cyclic_shift_time,
            computation_time;

    int rankInFiber, rankInLayer, shift;

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

        if(grid->GetRankInFiber() == 0) {
            generateRandomMatrix(logM, nnz_per_row,
                grid->GetCommGridLayer(),
                S,
                SValues
            );
            localSrows = S.nrows;
            if(proc_rank == 0) {
                cout << "Generated " << S.dist_nnz << " nonzeros." << endl;
            }
        }

        // Step 4: broadcast nonzero counts across fibers, allocate SpMat arrays 
        MPI_Bcast(&(S.local_nnz), 1, MPI_INT, 0, grid->GetFiberWorld());
        MPI_Bcast(&localSrows, 1, MPI_UINT64_T, 0, grid->GetFiberWorld());
        localBrows = (int) ceil((float) N  * c / p);

        S.rCoords.resize(S.local_nnz);
        S.cCoords.resize(S.local_nnz);

        // Step 5: broadcast the sparse matrices (the coordinates, not the values)
        MPI_Bcast(S.rCoords.data(), S.local_nnz, MPI_UINT64_T, 0, grid->GetFiberWorld());
        MPI_Bcast(S.cCoords.data(), S.local_nnz, MPI_UINT64_T, 0, grid->GetFiberWorld());

        // Step 6: Locate block starts within the local sparse matrix (i.e. divide a long
        // block row into subtiles) 
        int currentStart = 0;
        for(int i = 0; i < S.local_nnz; i++) {
            while(S.cCoords[i] >= currentStart) {
                blockStarts.push_back(i);
                currentStart += localBrows;
            }

            // This modding step helps indexing. 
            S.cCoords[i] %= localBrows;
        }
        while(blockStarts.size() < p / c + 1) {
            blockStarts.push_back(S.local_nnz);
        }

        rankInFiber = grid->GetRankInFiber();
        rankInLayer = grid->GetRankInLayer();
        shift = rankInFiber * p / (c * c);
    };

    // Factory functions: allocate dense matrices that can be used
    // as buffers with the algorithm

    VectorXd like_S_values(VectorXd &to_fill) {
        VectorXd res(S.local_nnz);
        return res; 
    }

    DenseMatrix like_A_matrix(DenseMatrix &to_fill) {
        DenseMatrix res(localSrows, R); 
        return res;
    }

    DenseMatrix like_B_matrix(DenseMatrix &to_fill) {
        DenseMatrix res(localBrows, R); 
        return res;
    }

    void reset_performance_timers() {
        nruns = 0;
        cyclic_shift_time = 0;
        computation_time  = 0;
        if(proc_rank == 0) {
            cout << "Performance timers reset..." << endl;
        }
    }

    void print_algorithm_info() {
        cout << "1.5D Replicating ABC Algorithm..." << endl;
        cout << "Matrix Dimensions: " 
        << this->M << " x " << this->N << endl;
        cout << "Nonzeros Per row: " << nnz_per_row << endl;
        cout << "R-Value: " << this->R << endl;
        cout << "Grid Dimensions: " << p / c << " x " << c << endl;
    }

    // Forget the initial broadcast now... 
    void initial_broadcast(DenseMatrix &localA, DenseMatrix &localB, VectorXd &SValues) {
        MPI_Bcast((void*) SValues.data(), SValues.size(), MPI_DOUBLE,     0, grid->GetFiberWorld());
        MPI_Bcast((void*) localA.data(), localA.rows() * localA.cols(), MPI_DOUBLE, 0, grid->GetFiberWorld());
        MPI_Bcast((void*) localB.data(), localB.rows() * localB.cols(), MPI_DOUBLE, 0, grid->GetFiberWorld());
    }

    int pMod(int num, int denom) {
        return ((num % denom) + denom) % denom;
    }

    void sddmm(DenseMatrix &localA, DenseMatrix &localB, VectorXd &SValues, VectorXd &sddmm_result, bool verbose) {
        nruns++;

        if(proc_rank == 0 && verbose) {
            print_algorithm_info();
            cout << "Executing SDDMM..." << endl;
        }

        MPI_Status stat;
        MPI_Request send_request;
        MPI_Request recv_request;
    
        int nnz_processed = 0;

        // Temporary buffer to hold the received portion of matrix B.
        DenseMatrix recvRowSlice(localB.rows(), localB.cols());

        auto t = start_clock();
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

            assert(blockStarts[block_id] < S.local_nnz);
            assert(blockStarts[block_id + 1] <= S.local_nnz);

            nnz_processed += sddmm_local(
                S,
                SValues,
                localA,
                localB,
                sddmm_result,
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

        int total_processed;
        MPI_Reduce(&nnz_processed, &total_processed, 1, MPI_INT,
                MPI_SUM, 0, MPI_COMM_WORLD);

        if(proc_rank == 0 && verbose) {
            cout << "Total Nonzeros Processed: " << total_processed << endl;
        }
    }

    void print_statistics() {
        double sum_comp_time, sum_shift_time; 

        MPI_Allreduce(&cyclic_shift_time, &sum_shift_time, 1,
                    MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        MPI_Allreduce(&computation_time, &sum_comp_time, 1,
                    MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        if(proc_rank == 0) {
            cout << "Avg. Cyclic Shift Time\t"
                 << "Avg. Computation Time" << endl;
                


            sum_shift_time     /= p * nruns;
            sum_comp_time      /= p * nruns;

            cout 
            << sum_shift_time << "\t"
            << sum_comp_time << "\t" << endl;

            cout << "=================================" << endl;
        }

    }

    void benchmark() {
        //sddmm(true);
        reset_performance_timers();

        int nruns = 10;
        for(int i = 0; i < nruns; i++) {
            //sddmm(false);
        }
        print_statistics();
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

    Sparse15D* x = new Sparse15D(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]));
    x->benchmark();

    delete x;

    MPI_Finalize();
}