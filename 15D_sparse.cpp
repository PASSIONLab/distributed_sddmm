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
double* Cslice;
double* result;

double* recvRowSlice;
double* recvResultSlice;

int procRow;
int procLayer;

vector<pair<size_t, size_t>> S;
vector<size_t> blockStarts;

// Meant to sort the sparse matrix in column-major order
struct column_major_sorter 
{
    inline bool operator() (const pair<size_t, size_t> &p1, const pair<size_t, size_t>& p2)
    {
        if(p1.second == p2.second) {
            return (p1.first < p2.first);
        }
        else {
            return p1.second < p2.second;
        }
    }
};

int r;

int M, N, K;

MPI_Comm interlayer_communicator;
MPI_Comm intralayer_communicator;

double broadcast_time;
double cyclic_shift_time;
double computation_time;
double reduction_time;

void reset_performance_timers() {
    broadcast_time = 0;
    cyclic_shift_time = 0;
    computation_time = 0;
    reduction_time = 0;
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

void setup15D(int M_loc, int N_loc, int K_loc, int c_loc, char* filename) {

    M = M_loc;
    N = N_loc;
    K = K_loc;
    c = c_loc;

    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

    int num_procs;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    p = (int) (num_procs / c);

    if(num_procs % (c * c) != 0) {
        if(proc_rank == 0) {
            cout << "Error, for 1.5D algorithm, must have c^2 divide num_procs!" << endl;
            exit(1);
        }

        /*if(K % p != 0) {
            cout << "Error, for 1.5D algorithm, must have K divisible by num_procs / c." << endl;
        }*/

        /*if(M % p != 0) {
            cout << "Error, for 1.5D algorithm, must have M divisible by num_procs / c." << endl;
        }*/
    }

    rowAwidth = (int) ceil((float) M / p);
    rowBwidth = (int) ceil((float) K / p);

    procRow = proc_rank % p; 
    procLayer = proc_rank / p;

    // We only need to fill the portion at row_rank, col_rank
    MPI_Comm_split(MPI_COMM_WORLD, procRow,   proc_rank, &interlayer_communicator);
    MPI_Comm_split(MPI_COMM_WORLD, procLayer, proc_rank, &intralayer_communicator);
    MPI_Comm_rank(interlayer_communicator, &row_rank);
    MPI_Comm_rank(intralayer_communicator, &layer_rank);

    // All processes simultaneously read in the sparse matrix, but only the portion that they own 
    ifstream ifs(filename, std::ifstream::in);
    ifs.ignore(1024, '\n'); // Ignore the first line, which assumes general symmetricity 
    int M_read, K_read, NNZ;
    ifs >> M_read >> K_read >> NNZ;

    //assert(M_read == M && K_read == K);
    M = M_read;
    K = K_read;

    for(int i = 0; i < NNZ; i++) {
        int r, c, v;
        ifs >> r >> c >> v;
        r--;
        //c--;  Because of an error, we don't want to subtract 1 here...
        if(rowAwidth * layer_rank <= r && r < rowAwidth * (layer_rank + 1)) {
            S.emplace_back(r, c);
        }
    }

    ifs.close();

    for(int i = 0; i < S.size(); i++) {
        if(S[i].first > M || S[i].second > K) {
            cout << "STOPPED BEFORE SORT AT " << i << endl;
            exit(1);
        }
    }

    // Further sort the coordinates into blocks 
    std::sort(S.begin(), S.end(), column_major_sorter());
    
    int currentStart = 0;
    for(int i = 0; i < S.size(); i++) {
        while(S[i].second >= currentStart) {
            blockStarts.push_back(i);
            currentStart += rowBwidth;
        }
    }
    while(blockStarts.size() < p + 1) {
        blockStarts.push_back(S.size());
    }

    for(int i = 0; i < S.size(); i++) {
        S[i].first %= rowAwidth;
        S[i].second %= rowBwidth;
    }

    /*if(proc_rank == 0) {
        for(int i = 0; i < blockStarts.size(); i++) {
            cout << blockStarts[i] << endl;
        }
    }*/

    // The portion of matrices A, B that are owend by the current processor 
    
    Aslice   = new double[S.size()];
    Bslice   = new double[rowBwidth * N];
    Cslice   = new double[rowAwidth * N];
    result   = new double[S.size()];
    
    // Fill values randomly

    
    for(int i = 0; i < rowBwidth * N; i++) {
        Bslice[i] = rand();
    }
    for(int i = 0; i < rowAwidth * N; i++) {
        Cslice[i] = rand();
    }
    
    recvRowSlice = new double[rowBwidth * N];
    recvResultSlice = new double[S.size()];


    if(proc_rank == 0) {
        cout << "Initialization complete!" << endl;
    } 
}

void swap(double* &A, double* &B) {
    double* temp = A;
    A = B;
    B = temp;
}

void algorithm() {
    MPI_Status stat;
    MPI_Request send_request;
    MPI_Request recv_request;

    // First replicate the two matrices across layers of the processor grid 

    int shift = procLayer * p / c;
    
    int nnz_processed = 0;

    auto t = start_clock();
    // This assumes that the matrices permanently live at the bottom-most layer 
    MPI_Bcast((void*) Aslice, S.size(), MPI_DOUBLE, 0, interlayer_communicator);
    MPI_Bcast((void*) Bslice, rowBwidth * N, MPI_DOUBLE, 0, interlayer_communicator);
    MPI_Bcast((void*) Cslice, rowAwidth * N, MPI_DOUBLE, 0, interlayer_communicator);
    broadcast_time += stop_clock(t);

    t = start_clock();
    // Initial shift
    MPI_Isend(Bslice, rowBwidth * N, MPI_DOUBLE, 
                (layer_rank + shift) % p, 0,
                intralayer_communicator, &send_request);

    MPI_Irecv(recvRowSlice, rowBwidth * N, MPI_DOUBLE, MPI_ANY_SOURCE,
              0, intralayer_communicator, &recv_request);

    MPI_Wait(&send_request, MPI_STATUS_IGNORE); 
    MPI_Wait(&recv_request, MPI_STATUS_IGNORE);

    MPI_Barrier(MPI_COMM_WORLD);
    cyclic_shift_time += stop_clock(t);

    swap(Bslice, recvRowSlice);

    for(int i = 0; i < p / c; i++) {

        t = start_clock();

        int block_id = (p + layer_rank - shift - i) % p;

        assert(blockStarts[block_id] < S.size());
        assert(blockStarts[block_id + 1] <= S.size());

        nnz_processed += kernel(S,
            Cslice,
            Bslice,
            N,
            result,
            blockStarts[block_id],
            blockStarts[block_id + 1]);

        computation_time += stop_clock(t);

        //cout << "Rank " << proc_rank << " multiplies " <<
        //    Aslice[((p + layer_rank + shift - i) % p) * rowBwidth] << endl;

        /*string res = "["; 
        for(int i = 0; i < rowAwidth; i++) {
            for(int j = 0; j < K; j++) {
                res += " " + to_string(Aslice[i * K + j]); 
            }
            res += "\n";
        }
        res += "]\n";
        cout << "Rank " << proc_rank << " has A matrix " << res << endl;

        res = "["; 
        for(int i = 0; i < rowBwidth; i++) {
            for(int j = 0; j < N; j++) {
                res += " " + to_string(Bslice[i * N + j]); 
            }
            res += "\n";
        }
        res += "]\n";
        cout << "Rank " << proc_rank << " has B matrix " << res << endl;

        res = "["; 
        for(int i = 0; i < rowAwidth; i++) {
            for(int j = 0; j < N; j++) {
                res += " " + to_string(result[i * N + j]); 
            }
            res += "\n";
        }
        res += "]\n";
        cout << "Rank " << proc_rank << " result: " << res << endl;
        */

        /*cout << "Process " << proc_rank << " with layer rank "
            << layer_rank << " sends to " << (layer_rank + 1) % p 
            << "." << endl;*/

        t = start_clock();
        
        MPI_Isend(Bslice, rowBwidth * N, MPI_DOUBLE, 
                    (p + layer_rank + 1) % p, 0,
                    intralayer_communicator, &send_request);

        MPI_Irecv(recvRowSlice, rowBwidth * N, MPI_DOUBLE, MPI_ANY_SOURCE,
                0, intralayer_communicator, &recv_request);

        MPI_Wait(&send_request, MPI_STATUS_IGNORE); 
        MPI_Wait(&recv_request, MPI_STATUS_IGNORE);
        
        MPI_Barrier(MPI_COMM_WORLD);

        cyclic_shift_time += stop_clock(t);

        swap(Bslice, recvRowSlice);
    }


    // Reduction across layers
    t = start_clock();
    MPI_Reduce(result, recvResultSlice, S.size(), MPI_DOUBLE,
               MPI_SUM, 0, interlayer_communicator);
    reduction_time += stop_clock(t);

    // Debugging only: print out the total number of dot products taken 

    int total_processed;
    MPI_Reduce(&nnz_processed, &total_processed, 1, MPI_INT,
               MPI_SUM, 0, MPI_COMM_WORLD);

    if(proc_rank == 0) {
        cout << "Total Nonzeros Processed: " << total_processed << endl;
    }
}

void finalize15D() {
    delete Aslice;
    delete Bslice;
    delete Cslice;
    delete result;
    delete recvRowSlice;

    MPI_Comm_free(&interlayer_communicator);
    MPI_Comm_free(&intralayer_communicator);
}

/*
 * SDDMM with N x N matrix, N x r matrix 
 */

void benchmark15D(int M_loc, int N_loc, int K_loc, int c_loc, char* filename) {
    // First, we are going to generate a sparse matrix; let's have every 
    // process generate its sparse matrix coordinates

    int num_procs;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    //int d = (int) (frac_nonzero * N);

    setup15D(M_loc, N_loc, K_loc, c_loc, filename);

    if(proc_rank == 0) {
        cout << "Number of Processes: " << num_procs <<
        ", N = " << N << ", K = " << K << 
        ", c=" << c << endl; 
    }

    reset_performance_timers();
    algorithm();
    
    double sum_broadcast_time, sum_comp_time, sum_shift_time, sum_reduce_time;

    MPI_Reduce(&broadcast_time, &sum_broadcast_time, 1,
                  MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Reduce(&cyclic_shift_time, &sum_shift_time, 1,
                  MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Reduce(&computation_time, &sum_comp_time, 1,
                  MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Reduce(&reduction_time, &sum_reduce_time, 1,
                  MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);


    if(proc_rank == 0) {
        sum_broadcast_time /= num_procs;
        sum_shift_time     /= num_procs;
        sum_comp_time      /= num_procs;
        sum_reduce_time    /= num_procs;
        //cout << "Average Broadcast Time: " << sum_broadcast_time<< endl;
        //cout << "Average Cyclic Shift Time: " << sum_shift_time << endl;
        //cout << "Average Computation Time:   " << sum_comp_time << endl;
        //cout << "Average Reduction Time: " << sum_reduce_time << endl;
        cout << sum_broadcast_time << " "
        << sum_shift_time << " "
        << sum_comp_time << " "
        << sum_reduce_time << endl;
    }

    finalize15D();
}

// Note: verified that this passes, commenting so that I can freely
// change some function signatures
void test1DCorrectness() {
    /*
    int num_procs;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    M = num_procs * 10;
    K = num_procs * 3;
    N = num_procs * 10;

    setup15D(M, N, K, 4);
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

    cblas_dgemm(CblasRowMajor, 
                CblasNoTrans, 
                CblasNoTrans, 
                M, 
                N, 
                K, 1., 
                A, K, 
                B, N, 1., 
                C_ground_truth, N);

    if(row_rank == 0) {
        MPI_Scatter(
        (void*) A,
        rowAwidth * K,
        MPI_DOUBLE,
        (void*) Aslice,
        rowAwidth * K,
        MPI_DOUBLE,
        0,
        intralayer_communicator);

        MPI_Scatter(
        (void*) B,
        rowBwidth * N,
        MPI_DOUBLE,
        (void*) Bslice,
        rowBwidth * N,
        MPI_DOUBLE,
        0,
        intralayer_communicator);
    }

    // cout << Bslice[0] << " " << Bslice[1] << endl;

    if(proc_rank == 0) {
        cout << "Starting algorithm!" << endl;
    }
    algorithm();
    if(row_rank == 0) {
        // cout << "Algorithm complete!" << endl;
        MPI_Gather(
            (void*) recvResultSlice,
            rowAwidth * N,
            MPI_DOUBLE,
            (void*) C_computed,
            rowAwidth * N,
            MPI_DOUBLE,
            0,
            intralayer_communicator);
    }

    if(proc_rank == 0) {
        for(int i = 0; i < M * N; i++) {
            //cout << C_ground_truth[i] << " " << C_computed[i] << endl;
            if(abs(C_ground_truth[i] - C_computed[i]) > 1e-7) {
                // cout << "Error between ground truth and computed value!" << endl;
                exit(1);
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
        sum_comp_time   /= num_procs;
        cout << "Communication Time: " << sum_comm_time << endl;
        cout << "Computation Time:   " << sum_comp_time << endl;
    }

    finalize15D();
    */
}

