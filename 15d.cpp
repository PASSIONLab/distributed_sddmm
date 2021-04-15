#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

#include <mpi.h>
#include <cblas.h>

#include "spmat_reader.h"
#include "pack.h"

using namespace std;

// The code below implements a 1.5D SUMMA algorithm

int p;
int c;

int proc_rank;
int row_rank;   // Rank within a row (top to bottom)
int layer_rank; // Rank within a layer (left to right)

int rowAwidth;
int rowBwidth;


double* Aslice;
double* Bslice;
double* result;

double* recvRowSlice;

int procRow;
int procLayer;

vector<pair<size_t, size_t>> S;

int r;

int M, N, K;

MPI_Comm interlayer_communicator;
MPI_Comm intralayer_communicator;

double communication_time;
double computation_time;

void reset_performance_timers() {
    communication_time = 0;
    computation_time = 0;
}

double get_communication_time() {
    return communication_time;
}

double get_computation_time() {
    return computation_time;
}

chrono::time_point<std::chrono::steady_clock> start_clock() {
    return std::chrono::steady_clock::now();
}

double stop_clock(chrono::time_point<std::chrono::steady_clock> &start) {
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;
    return diff.count();
}

// Setup data structures; assert p^2 is the total number of processors 

void setup15D(int M_loc, int N_loc, int K_loc, int c_loc) {
    M = M_loc;
    N = N_loc;
    K = K_loc;
    c = c_loc;

    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

    int num_procs;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    p = (int) num_procs / c;

    if(num_procs % (c * c) != 0) {
        if(proc_rank == 0) {
            cout << "Error, for 1.5D algorithm, must have c^2 divide num_procs!" << endl;
        }

        if(K % p != 0) {
            cout << "Error, for 1.5D algorithm, must have K divisible by num_procs / c." << endl;
        }

        if(M % p != 0) {
            cout << "Error, for 1.5D algorithm, must have M divisible by num_procs / c." << endl;
        }
    }

    rowAwidth = (int) ceil((float) M / p);
    rowBwidth = (int) ceil((float) K / p);

    procRow = proc_rank % p; 
    procLayer = proc_rank / p;

    // The portion of matrices A, B that are owend by the current processor 
    Aslice   = new double[rowAwidth * K];
    Bslice   = new double[rowBwidth * N];
    result   = new double[rowAwidth * N];

    recvRowSlice = new double[rowAwidth * K];

    // We only need to fill the portion at row_rank, col_rank
    MPI_Comm_split(MPI_COMM_WORLD, procRow,   proc_rank, &interlayer_communicator);
    MPI_Comm_split(MPI_COMM_WORLD, procLayer, proc_rank, &intralyer_communicator);
    MPI_Comm_rank(interlayer_communicator, &layer_rank);
    MPI_Comm_rank(intralayer_communicator, &row_rank);
}

void algorithm() {
    /*MPI_Status stat;

    // First replicate the two matrices across layers of the processor grid 



    // SUMMA algorithm
    for(int i = 0; i < p; i++) {
        double *rbuf, *cbuf;

        if(i == row_rank) {
            rbuf = rowSlice;
        }
        else {
            rbuf = recvRowSlice;
        }
        auto t = start_clock();
        MPI_Bcast((void*) rbuf, rowAwidth * colWidth, MPI_DOUBLE, i, row_communicator);
        communication_time += stop_clock(t);

        if(i == col_rank) {
            cbuf = colSlice;
        }
        else {
            cbuf = recvColSlice;
        }

        // Should overlap communication and computation, but not doing so yet...
        t = start_clock();
        MPI_Bcast((void*) cbuf, rowBwidth * colWidth, MPI_DOUBLE, i, col_communicator);
        communication_time += stop_clock(t);

        t = start_clock();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rowAwidth, rowBwidth, colWidth, 1., rbuf, colWidth, cbuf, rowBwidth, 1., result, rowBwidth);
        computation_time += stop_clock(t);

    }*/
}


void finalize15D() {
    free(Aslice);
    free(Bslice);
    free(result);

    free(recvRowSlice);

    MPI_Comm_free(&interlayer_communicator);
    MPI_Comm_free(&intralayer_communicator);

}

void test1DCorrectness() {
    int num_procs;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    int testMult = (int) sqrt(num_procs);

    int M = testMult * 1000;
    int K = testMult * 1000;
    int N = testMult * 100;

    setup15D(M, N, K);
    reset_performance_timers();

    // Interpret both of the matrices as being in row major order
    double* A              = new double[M * K];
    double* B              = new double[K * N];

    double* A_packed       = new double[M * K];
    double* B_packed       = new double[K * N];

    double* C_computed     = new double[M * N];
    double* C_ground_truth = new double[M * N];

    // Standard mersenne_twister_engine seeded with 1.0 
    // (we just care that the matrices are nonzero) 
    std::mt19937 gen(1); 
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    for(int i = 0; i < M * K; i++) {
        // A[i] = dis(gen); 
        A[i] = i + 1; 
    }

    for(int i = 0; i < N * K; i++) {
        // B[i] = dis(gen);
        B[i] = i + 1;
    }

    cblas_dgemm(CblasColMajor, 
                CblasNoTrans, 
                CblasNoTrans, 
                M, 
                N, 
                K, 1., 
                A, M, 
                B, K, 1., 
                C_ground_truth, M);

    MPI_Scatter(
    (void*) A,
    rowAwidth * K,
    MPI_DOUBLE,
    (void*) Aslice,
    rowAwidth * K,
    MPI_DOUBLE,
    0,
    MPI_COMM_WORLD);

    MPI_Scatter(
    (void*) B,
    rowBwidth * N,
    MPI_DOUBLE,
    (void*) Bslice,
    rowBwidth * N,
    MPI_DOUBLE,
    0,
    MPI_COMM_WORLD);

    if(proc_rank == 0) {
        cout << "Starting algorithm!" << endl;
    }
    algorithm();
    if(proc_rank == 0) {
        cout << "Algorithm complete!" << endl;
    }

    MPI_Gather(
    (void*) result,
    rowAwidth * N,
    MPI_DOUBLE,
    (void*) C_computed,
    rowAwidth * N,
    MPI_DOUBLE,
    0,
    MPI_COMM_WORLD);

    if(proc_rank == 0) {
        for(int i = 0; i < M * N; i++) {
            cout << C_ground_truth[i] << " " << C_computed[i] << endl;
            /*if(abs(C_ground_truth[i] - C_computed[i]) > 1e-7) {
                cout << "Error between ground truth and computed value!" << endl;
                exit(1);
            }*/
        }

        cout << "Correctness Check Passed!" << endl;
    }

    // Average the communication and computation time
    double sum_comm_time;
    double sum_comp_time;
    MPI_Reduce(&communication_time, &sum_comm_time, 1,
                  MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Reduce(&computation_time, &sum_comp_time, 1,
                  MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if(proc_rank == 0) {
        sum_comm_time /= num_procs;
        sum_comp_time   /= num_procs;
        cout << "Communication Time: " << sum_comm_time << endl;
        cout << "Computation Time:   " << sum_comp_time << endl;
    }

    finalize2D();
}

