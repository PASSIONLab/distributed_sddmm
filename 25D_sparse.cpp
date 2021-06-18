#include <iostream>
#include <chrono>
#include <cmath>
#include <utility>
#include <vector>
#include <string.h>
#include <vector>
#include <utility>
#include <cassert>
#include <mpi.h>
#include <cblas.h>
#include <algorithm>

#include "sparse_kernels.h"
#include "common.h"
#include "io_utils.h"
#include "als_conjugate_gradients.h"
#include "distributed_sparse.h"

// CombBLAS includes 
#include <memory>
#include "CombBLAS/CombBLAS.h"

using namespace std;
using namespace combblas;

/*
 * Clients are responsible for allreducing their results and keeping
 * matrices in sync. Data fields are exposed public to permit this. 
 */

class Sparse25D : public Distributed_Sparse{
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

    // These are the local dense matrix buffers (first two)
    // and the buffer for the local nonzeros 
    int nrowsA, nrowsB, ncolsLocal;

    // Pointer to object implementing the local SDDMM / SPMM Operations 
    KernelImplementation *kernel; 

    int n_dense_reductions, n_sparse_reductions;

    // Performance timers 
    int nruns;
    double computation_time;
    double dense_reduction_time;
    double sparse_reduction_time;

    void constructor_helper(bool readFromFile, 
            int logM, 
            int nnz_per_row, 
            string filename, 
            int R, 
            int c, 
            KernelImplementation *k) {


        MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &p);

        this->kernel = k;
        this->R = R;
        this->c = c;
        this->nnz_per_row = nnz_per_row;


        // STEP 1: Make sure the replication factor is valid for the number
        // of processes running
        sqrtpc = (int) round(sqrt(p / c));

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
        nrowsA = this->M / sqrtpc + 1;
        nrowsB = this->N / sqrtpc + 1;
        ncolsLocal = this->R / c;

        // Step 8: Indicate which axes the A and B matrices are split along 
        A_R_split_world = grid->GetFiberWorld();
        B_R_split_world = grid->GetFiberWorld(); 
    }

    Sparse25D(int logM, int nnz_per_row, int R, int c, KernelImplementation *k) { 
        constructor_helper(false, logM, nnz_per_row, "", R, c, k);
    }

    Sparse25D(string filename, int R, int c, KernelImplementation *k) {
        constructor_helper(true, 0, 0, filename, R, c, k);
    }

    VectorXd like_S_values(double value) {
        return VectorXd::Constant(S.local_nnz, value); 
    }

    DenseMatrix like_A_matrix(double value) {
        return DenseMatrix::Constant(nrowsA, ncolsLocal, value);  
    }

    DenseMatrix like_B_matrix(double value) {
        return DenseMatrix::Constant(nrowsB, ncolsLocal, value);  
    }

    void reset_performance_timers() {
        nruns = 0; 
        computation_time = 0;
        dense_reduction_time = 0;
        sparse_reduction_time = 0;

        n_dense_reductions = 0;
        n_sparse_reductions = 0;

        if(proc_rank == 0) {
            cout << "Performance timers reset..." << endl;
        }
    }

    void print_algorithm_info() {
        cout << "2.5D Striped AB Replicating S Algorithm..." << endl;
        cout << "Matrix Dimensions: " 
        << this->M << " x " << this->N << endl;
        cout << "Nonzeros Per row: " << nnz_per_row << endl;
        cout << "R-Value: " << this->R << endl;
        cout << "Grid Dimensions: " << p / c << " x " << c << endl;
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
        n_dense_reductions++;
        shared_ptr<CommGrid> commGridLayer = grid->GetCommGridLayer();
        algorithm(localA, localB, SValues, nullptr, k_spmmA);
        auto t = start_clock();
        MPI_Allreduce(MPI_IN_PLACE, localA.data(), localA.size(), MPI_DOUBLE, MPI_SUM, commGridLayer->GetRowWorld());
        stop_clock_and_add(t, &dense_reduction_time);
    }

    void spmmB(DenseMatrix &localA, DenseMatrix &localB, VectorXd &SValues) {
        n_dense_reductions++;
        shared_ptr<CommGrid> commGridLayer = grid->GetCommGridLayer();
        algorithm(localA, localB, SValues, nullptr, k_spmmB);
        auto t = start_clock();
        MPI_Allreduce(MPI_IN_PLACE, localB.data(), localB.size(), MPI_DOUBLE, MPI_SUM, commGridLayer->GetColWorld());
        stop_clock_and_add(t, &dense_reduction_time);
    }

    void sddmm(DenseMatrix &localA, DenseMatrix &localB, VectorXd &SValues, VectorXd &sddmm_result) { 
        n_sparse_reductions++;
        algorithm(localA, localB, SValues, &sddmm_result, k_sddmm);
        auto t = start_clock();
        MPI_Allreduce(MPI_IN_PLACE, SValues.data(), SValues.size(), MPI_DOUBLE, MPI_SUM, grid->GetFiberWorld()); 
        stop_clock_and_add(t, &sparse_reduction_time);
    }

    /*
     * Hmmm... should we just call this Johnson's Algorithm? 
     *
     */
    void algorithm(         DenseMatrix &localA, 
                            DenseMatrix &localB, 
                            VectorXd &SValues, 
                            VectorXd *sddmm_result_ptr, 
                            KernelMode mode
                            ) {

        int nnz_processed = 0;
        nruns++;
        
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
        stop_clock_and_add(t, &computation_time);

        // Debugging only: print out the total number of dot products taken, reduce across
        // each layer world as a sanity check 

        int total_processed;
        MPI_Reduce(&nnz_processed, &total_processed, 1, MPI_INT,
                MPI_SUM, 0, grid->GetLayerWorld());

        if(proc_rank == 0 && verbose) {
            cout << "Total Nonzeros Processed: " << total_processed << endl;
        } 

    }

    void print_statistics() {
        double sum_comp_time, sum_dense_reduction_time, sum_sparse_reduction_time; 

        MPI_Allreduce(&dense_reduction_time, &sum_dense_reduction_time, 1,
                    MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        MPI_Allreduce(&sparse_reduction_time, &sum_sparse_reduction_time, 1,
                    MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        MPI_Allreduce(&computation_time, &sum_comp_time, 1,
                    MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        if(proc_rank == 0) {
            cout 
                 << "Avg. Dense Allreduction Time\t" 
                 << "Avg. Sparse Allreduction Time\t" 
                 << "Avg. Computation Time" << endl;
                
            sum_dense_reduction_time       /= p * n_dense_reductions;
            sum_sparse_reduction_time      /= p * n_sparse_reductions;
            sum_comp_time                  /= p * nruns;

            cout 
            << sum_dense_reduction_time << "\t"
            << sum_sparse_reduction_time << "\t" 
            << sum_comp_time << endl;

            cout << "=================================" << endl;
        }
    }

    ~Sparse25D() {
        // Destructor 
    }
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    string fname(argv[1]);
    int R = atoi(argv[2]);
    int c = atoi(argv[3]);

    StandardKernel local_ops;
    //Sparse25D* d_ops25D = new Sparse25D(fname, R, c, &local_ops);
    Sparse25D* d_ops25D = new Sparse25D(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), &local_ops);

    d_ops25D->reset_performance_timers();
    Distributed_ALS* x = new Distributed_ALS(d_ops25D, d_ops25D->grid->GetLayerWorld(), true); 
    //d_ops->setVerbose(true);
    x->run_cg(5);

    d_ops25D->print_statistics();

    MPI_Finalize();
}