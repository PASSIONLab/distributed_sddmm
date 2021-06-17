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
#include "als_conjugate_gradients.h"
#include "distributed_sparse.h"

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

class Sparse15D : public Distributed_Sparse {
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
            computation_time,
            reduction_time;

    int rankInFiber, rankInLayer, shift;

    // We can either read from a file or use the R-mat generator for testing purposes
    void constructor_helper(bool readFromFile, int logM, int nnz_per_row, string filename, int R, int c, KernelImplementation* k) {
        this->kernel = k;
        this->R = R;
        this->c = c;
        this->nnz_per_row = nnz_per_row; 
        
        verbose = false;

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

        VectorXd SValues;                 // For the R-Mat generator, ignore the actual values 
        if(grid->GetRankInFiber() == 0) {
            if(! readFromFile) {
                generateRandomMatrix(logM, nnz_per_row,
                    grid->GetCommGridLayer(),
                    S,
                    SValues
                );

                if(proc_rank == 0) {
                    cout << "R-mat generator created " << S.dist_nnz << " nonzeros." << endl;
                }
            }
            else {
                loadMatrixFromFile(filename, grid->GetCommGridLayer(), S, SValues);
                if(proc_rank == 0) {
                    cout << "File reader read " << S.dist_nnz << " nonzeros." << endl;
                }

            }
            this->M = S.distrows;
            this->N = S.distcols;
            localSrows = S.nrows;
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

        A_R_split_world = grid->GetCommGridLayer()->GetRowWorld();
        B_R_split_world = grid->GetCommGridLayer()->GetRowWorld();
    }

    // Initiates the algorithm for a Graph500 benchmark 
    Sparse15D(int logM, int nnz_per_row, int R, int c, KernelImplementation* k) {
        constructor_helper(false, logM, nnz_per_row, "", R, c, k);
    }

    // Reads the underlying sparse matrix from a file
    Sparse15D(string &filename, int R, int c, KernelImplementation* k) {
        constructor_helper(true, 0, 0, filename, R, c, k);
    }

    // Factory functions: allocate dense matrices that can be used
    // as buffers with the algorithm

    VectorXd like_S_values(double value) {
        return VectorXd::Constant(S.local_nnz, value); 
    }

    DenseMatrix like_A_matrix(double value) {
        return DenseMatrix::Constant(localSrows, R, value);  
    }

    DenseMatrix like_B_matrix(double value) {
        return DenseMatrix::Constant(localBrows, R, value);  
    }

    void reset_performance_timers() {
        nruns = 0;
        cyclic_shift_time = 0;
        computation_time  = 0;
        reduction_time = 0;
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

    // Synchronizes data across three levels of the processor grid
    void initial_synchronize(DenseMatrix *localA, DenseMatrix *localB, VectorXd *SValues) {
        MPI_Bcast((void*) localA->data(), localA->rows() * localA->cols(), MPI_DOUBLE, 0, grid->GetFiberWorld());
        MPI_Bcast((void*) localB->data(), localB->rows() * localB->cols(), MPI_DOUBLE, 0, grid->GetFiberWorld());

        if(SValues != nullptr) {
            MPI_Bcast((void*) SValues->data(), SValues->size(), MPI_DOUBLE,     0, grid->GetFiberWorld());
        }
    }

    void spmmA(DenseMatrix &localA, DenseMatrix &localB, VectorXd &SValues) {
        algorithm(localA, localB, SValues, nullptr, k_spmmA);
        auto t = start_clock();
        MPI_Allreduce(MPI_IN_PLACE, localA.data(), localA.size(), MPI_DOUBLE, MPI_SUM, grid->GetFiberWorld());
        stop_clock_and_add(t, &reduction_time);
    }

    void spmmB(DenseMatrix &localA, DenseMatrix &localB, VectorXd &SValues) {
        algorithm(localA, localB, SValues, nullptr, k_spmmB);
        auto t = start_clock();
        MPI_Allreduce(MPI_IN_PLACE, localB.data(), localB.size(), MPI_DOUBLE, MPI_SUM, grid->GetFiberWorld());
        stop_clock_and_add(t, &reduction_time);
    }

    void sddmm(DenseMatrix &localA, DenseMatrix &localB, VectorXd &SValues, VectorXd &sddmm_result) { 
        algorithm(localA, localB, SValues, &sddmm_result, k_sddmm);
        auto t = start_clock();
        MPI_Allreduce(MPI_IN_PLACE, SValues.data(), SValues.size(), MPI_DOUBLE, MPI_SUM, grid->GetFiberWorld()); 
        stop_clock_and_add(t, &reduction_time);
    }

    /*
     * Set the mode to take an SDDMM, SpMM with A as the output matrix, or 
     * SpMM with B as the output matrix. 
     */
    void algorithm(         DenseMatrix &localA, 
                            DenseMatrix &localB, 
                            VectorXd &SValues, 
                            VectorXd *sddmm_result_ptr, 
                            KernelMode mode
                            ) {
  
        nruns++;

        if(proc_rank == 0 && verbose) {
            print_algorithm_info();
        }

        MPI_Status stat;
        MPI_Request send_request;
        MPI_Request recv_request;
    
        int nnz_processed = 0;

        // Temporary buffer to hold the received portion of matrix B.
        DenseMatrix recvRowSlice(localB.rows(), localB.cols());

        auto t = start_clock();
        MPI_Isend(localB.data(), localB.rows() * localB.cols(), MPI_DOUBLE, 
                    pMod(rankInLayer + shift, p / c), 0,
                    grid->GetLayerWorld(), &send_request);

        MPI_Irecv(recvRowSlice.data(), recvRowSlice.rows() * recvRowSlice.cols(), MPI_DOUBLE, MPI_ANY_SOURCE,
                0, grid->GetLayerWorld(), &recv_request);

        MPI_Wait(&send_request, MPI_STATUS_IGNORE); 
        MPI_Wait(&recv_request, MPI_STATUS_IGNORE);
        stop_clock_and_add(t, &cyclic_shift_time);

        localB = recvRowSlice;

        MPI_Barrier(MPI_COMM_WORLD);

        for(int i = 0; i < p / (c * c); i++) {

            int block_id = pMod(rankInLayer - shift - i, p / c);

            assert(blockStarts[block_id] < S.local_nnz);
            assert(blockStarts[block_id + 1] <= S.local_nnz);

            t = start_clock();
            if(mode == k_sddmm) {
                nnz_processed += kernel->sddmm_local(
                    S,
                    SValues,
                    localA,
                    localB,
                    *sddmm_result_ptr,
                    blockStarts[block_id],
                    blockStarts[block_id + 1]);
            }
            else if(mode == k_spmmA) {
                nnz_processed += kernel->spmm_local(
                    S,
                    SValues,
                    localA,
                    localB,
                    Amat,
                    blockStarts[block_id],
                    blockStarts[block_id + 1]);
            }
            else if(mode == k_spmmB) {
                nnz_processed += kernel->spmm_local(
                    S,
                    SValues,
                    localA,
                    localB,
                    Bmat,
                    blockStarts[block_id],
                    blockStarts[block_id + 1]);
            }

            stop_clock_and_add(t, &computation_time);

            t = start_clock();
            MPI_Isend(localB.data(), localB.rows() * localB.cols(), MPI_DOUBLE, 
                        pMod(rankInLayer + 1, p / c), 0,
                        grid->GetLayerWorld(), &send_request);

            MPI_Irecv(recvRowSlice.data(), recvRowSlice.rows() * recvRowSlice.cols(), MPI_DOUBLE, MPI_ANY_SOURCE,
                    0, grid->GetLayerWorld(), &recv_request);

            MPI_Wait(&send_request, MPI_STATUS_IGNORE); 
            MPI_Wait(&recv_request, MPI_STATUS_IGNORE);
            stop_clock_and_add(t, &cyclic_shift_time);

            localB = recvRowSlice;

            MPI_Barrier(MPI_COMM_WORLD);
        }

        // Send the B matrix back to its original position
        t = start_clock();
        MPI_Isend(localB.data(), localB.rows() * localB.cols(), MPI_DOUBLE, 
                    pMod(rankInLayer - (p / c * c) - shift, p / c), 0,
                    grid->GetLayerWorld(), &send_request);

        MPI_Irecv(recvRowSlice.data(), recvRowSlice.rows() * recvRowSlice.cols(), MPI_DOUBLE, MPI_ANY_SOURCE,
                0, grid->GetLayerWorld(), &recv_request);
        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Wait(&send_request, MPI_STATUS_IGNORE); 
        MPI_Wait(&recv_request, MPI_STATUS_IGNORE);
        stop_clock_and_add(t, &cyclic_shift_time);

        int total_processed;
        MPI_Reduce(&nnz_processed, &total_processed, 1, MPI_INT,
                MPI_SUM, 0, MPI_COMM_WORLD);

        if(proc_rank == 0 && verbose) {
            cout << "Total Nonzeros Processed: " << total_processed << endl;
        }
    }

    void print_statistics() {
        double sum_comp_time, sum_reduction_time, sum_shift_time; 

        MPI_Allreduce(&cyclic_shift_time, &sum_shift_time, 1,
                    MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        MPI_Allreduce(&reduction_time, &sum_reduction_time, 1,
                    MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        MPI_Allreduce(&computation_time, &sum_comp_time, 1,
                    MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);


        if(proc_rank == 0) {
            cout << "Avg. Cyclic Shift Time\t"
                 << "Avg. Reduction Time\t" 
                 << "Avg. Computation Time" << endl;
                
            sum_shift_time          /= p * nruns;
            sum_reduction_time      /= p * nruns;
            sum_comp_time           /= p * nruns;

            cout 
            << sum_shift_time << "\t"
            << sum_reduction_time << "\t" 
            << sum_comp_time << endl;

            cout << "=================================" << endl;
        }

    }

    ~Sparse15D() {
        // Destructor
    }
};


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    StandardKernel local_ops;
    Sparse15D* d_ops = new Sparse15D(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), &local_ops);
    Distributed_ALS* x = new Distributed_ALS(d_ops, d_ops->grid->GetLayerWorld());

    x->run_cg(5);

    // Arguments:
    // 1. Log of side length of sparse matrix
    // 2. NNZ per row
    // 3. R-Dimension Length
    // 4. Replication factor

    //StandardKernel kernel;
/*    ALS15D* x = new ALS15D(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]));
    x->run_cg(20);
    delete x;*/

    //delete x;
    //delete d_ops;

    MPI_Finalize();
}