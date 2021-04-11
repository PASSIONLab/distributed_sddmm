#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <utility>
#include <vector>
#include <string>
#include <mpi.h>
#include <cblas.h>

#include "spmat_reader.h"
#include "pack.h"

using namespace std;

// The code below implements a 3D SUMMA algorithm

int p;

int proc_rank;
int row_rank;
int col_rank;
int pencil_rank;

int rowAwidth;
int rowBwidth;
int colWidth;

double* rowSlice;
double* colSlice;
double* result;

double* recvRowSlice;
double* recvColSlice;
double* recvResultSlice;

int procRow;
int procCol;
int procPencil;

vector<pair<size_t, size_t>> S;

int r;

int M, N, K;

MPI_Comm row_communicator;
MPI_Comm col_communicator;
MPI_Comm pencil_communicator;

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

void finalize3D() { 
    free(rowSlice);
    free(colSlice);
    free(result);

    free(recvRowSlice); 
    free(recvColSlice); 
    free(recvResultSlice); 

    MPI_Comm_free(&row_communicator);
    MPI_Comm_free(&col_communicator);
    MPI_Comm_free(&pencil_communicator);
}

// Setup data structures; assert p^2 is the total number of processors 

void setup3D(int M, int N, int K) {
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

    int num_procs;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    p = (int) cbrt(num_procs);

    if(p * p * p != num_procs ) {
        if(proc_rank == 0) {
            cout << "Error, for 3D algorithm, must have a cube number of MPI ranks!" << endl;
        }
        finalize3D();
        exit(1);
    }

    rowAwidth = (int) ceil((float) M / p);
    rowBwidth = (int) ceil((float) N / p);
    colWidth =  (int) ceil((float) K / p);

    // This processor grid stores processors in column-major order 
    procRow    = proc_rank % p; 
    procCol    = (proc_rank / p) % p;
    procPencil = proc_rank / (p * p);

    // The portion of matrices A, B that are owend by the current processor 
    rowSlice = new double[rowAwidth * colWidth];
    colSlice = new double[rowBwidth * colWidth];
    result   = new double[rowAwidth * rowBwidth];

    recvRowSlice = new double[rowAwidth * colWidth];
    recvColSlice = new double[rowBwidth * colWidth];
    recvResultSlice = new double[rowAwidth * rowBwidth];

    MPI_Comm_split(MPI_COMM_WORLD, procRow * p + procPencil,    proc_rank, &row_communicator);
    MPI_Comm_split(MPI_COMM_WORLD, procCol * p + procPencil,    proc_rank, &col_communicator);
    MPI_Comm_split(MPI_COMM_WORLD, procRow * p + procCol,       proc_rank, &pencil_communicator);

    MPI_Comm_rank(row_communicator,    &row_rank);
    MPI_Comm_rank(col_communicator,    &col_rank);
    MPI_Comm_rank(pencil_communicator, &pencil_rank);
}

void algorithm() {
    MPI_Status stat;
    double *rbuf, *cbuf;

    if(procRow == 0) {
        rbuf = rowSlice;
    }
    else {
        rbuf = recvRowSlice;
    }

    auto t = start_clock();
    MPI_Bcast((void*) rbuf, rowAwidth * colWidth, MPI_DOUBLE, 0, col_communicator);
    communication_time += stop_clock(t);

    // Print out the local portion of the matrix owned by the processor
    string x = "[";

    for(int i = 0; i < rowAwidth * colWidth; i++) {
        x += " " + to_string(rbuf[i]);
    }

    x += "]\n";
    // cout << "Rank " << proc_rank << " " << x << endl;


    if(procCol == 0) {
        cbuf = colSlice;
    }
    else {
        cbuf = recvColSlice;
    }

    t = start_clock();
    MPI_Bcast((void*) cbuf, rowBwidth * colWidth, MPI_DOUBLE, 0, row_communicator);
    communication_time += stop_clock(t);

    t = start_clock();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rowAwidth, rowBwidth, colWidth, 1., rbuf, colWidth, cbuf, rowBwidth, 1., result, rowBwidth);
    computation_time += stop_clock(t);

    // cout << "Rank " << proc_rank << " multiplies " << *rbuf << " " << *cbuf << endl;
    // cout << "Rank " << proc_rank << result[0] << endl; 

    t = start_clock();
    MPI_Reduce(result, recvResultSlice, rowAwidth * rowBwidth,
                  MPI_DOUBLE, MPI_SUM, 0, pencil_communicator);
    communication_time += stop_clock(t);
}

void test3DCorrectness() {
    int num_procs;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    int testMult = (int) cbrt(num_procs);

    int M = testMult * 100;
    int N = testMult * 100;
    int K = testMult * 100;

    setup3D(M, N, K);

    reset_performance_timers();
    MPI_Comm sliceRow;
    MPI_Comm sliceCol;
    MPI_Comm slicePencil;

    MPI_Comm_split(MPI_COMM_WORLD, procRow,    p * procPencil + procCol, &sliceRow);
    MPI_Comm_split(MPI_COMM_WORLD, procCol,    p * procRow + procPencil, &sliceCol);
    MPI_Comm_split(MPI_COMM_WORLD, procPencil, p * procRow + procCol, &slicePencil);

    // Interpret both of the matrices as being in row major order
    double* A              = new double[M * K];
    double* B              = new double[K * N];

    double* A_packed       = new double[M * K];
    double* B_packed       = new double[K * N];

    double* C_computed     = new double[M * N];
    double* C_ground_truth = new double[M * N];
    double* C_packed       = new double[M * N];

    // Standard mersenne_twister_engine seeded with 1.0 
    // (we just care that the matrices are nonzero) 
    std::mt19937 gen(1); 
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    for(int i = 0; i < M * K; i++) {
        A[i] = dis(gen); 
        // A[i] = i + 1;
    }

    for(int i = 0; i < N * K; i++) {
        B[i] = dis(gen);
        // B[i] = i + 1;
    }

    tile_spec_t Atile[2] = {{rowAwidth, colWidth,  1}, {-1, 0, 0}};
    tile_spec_t Btile[2] = {{colWidth,  rowBwidth, 1}, {-1, 0, 0}};
    tile_spec_t Ctile[2] = {{rowAwidth, rowBwidth, 1}, {-1, 0, 0}};

    pack(A, A_packed, Atile, 2, M, K); 
    pack(B, B_packed, Btile, 2, K, N); 

    cblas_dgemm(CblasColMajor, 
                CblasNoTrans, 
                CblasNoTrans, 
                M, 
                N, 
                K, 1., 
                A, M, 
                B, K, 1., 
                C_ground_truth, M);

    if(procRow == 0) {
        MPI_Scatter(
        (void*) A_packed,
        rowAwidth * colWidth,
        MPI_DOUBLE,
        (void*) rowSlice,
        rowAwidth * colWidth,
        MPI_DOUBLE,
        0,
        sliceRow);
    }

    if(procCol == 0) {
        MPI_Scatter(
        (void*) B_packed,
        rowBwidth * colWidth,
        MPI_DOUBLE,
        (void*) colSlice,
        rowBwidth * colWidth,
        MPI_DOUBLE,
        0,
        sliceCol);
    }

    if(proc_rank == 0) {
        cout << "Starting algorithm!" << endl;
    }
    algorithm();
    if(proc_rank == 0) {
        cout << "Algorithm complete!" << endl;
    }

    if(procPencil == 0) {
        MPI_Gather(
        (void*) recvResultSlice,
        rowAwidth * rowBwidth,
        MPI_DOUBLE,
        (void*) C_packed,
        rowAwidth * rowBwidth,
        MPI_DOUBLE,
        0,
        slicePencil);
    }

    // cout << proc_rank << " has " << recvResultSlice[0] << endl;

    if(proc_rank == 0) {
        unpack(C_computed, C_packed, Ctile, 2, M, N);
        for(int i = 0; i < M * N; i++) {
            // cout << C_ground_truth[i] << " " << C_computed[i] << " " << C_packed[i] << endl;
            if(abs(C_ground_truth[i] - C_computed[i]) > 1e-7) {
                cout << "Error between ground truth and computed value!" << endl;
                // exit(1);
            }
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
        sum_comp_time /= num_procs;
        cout << "Average Communication Time: " << sum_comm_time << endl;
        cout << "Average Computation Time:   " << sum_comp_time << endl;
    }

    finalize3D();

    MPI_Comm_free(&sliceRow);
    MPI_Comm_free(&sliceCol);
    MPI_Comm_free(&slicePencil);
}

