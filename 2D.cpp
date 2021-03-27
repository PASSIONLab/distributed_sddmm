#include <iostream>
#include <utility>
#include <vector>
#include <cmath>
#include <mpi.h>
#include <cblas.h>

#include "spmat_reader.h"

using namespace std;

// The code below implements a 2D SUMMA algorithm

int p;

int proc_rank;
int row_rank;
int col_rank;

int rowAwidth;
int rowBwidth;
int colWidth;

double* rowSlice;
double* colSlice;
double* result;

double* recvRowSlice;
double* recvColSlice;

int procRow;
int procCol;

vector<pair<size_t, size_t>> S;

int r;

int M, N, K;

MPI_Comm row_communicator;
MPI_Comm col_communicator;

// Setup data structures; assert p^2 is the total number of processors 

void setup2D(int M, int N, int K) {
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

    int num_procs;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    p = (int) sqrt(num_procs);

    if(p * p != num_procs ) {
        if(proc_rank == 0) {
            cout << "Error, for 2D algorithm, must have a square number of MPI ranks!" << endl;
        }
    }

    int rowAwidth = (int) ceil((float) M / p);
    int rowBwidth = (int) ceil((float) N / p);
    int colWidth =  (int) ceil((float) K / p);

    // This processor grid stores processors in column-major order 
    procRow = proc_rank % p; 
    procCol = proc_rank / p;

    // The portion of matrices A, B that are owend by the current processor 
    rowSlice = new double[rowAwidth * colWidth];
    colSlice = new double[rowBwidth * colWidth];
    result   = new double[rowAwidth * rowBwidth];

    recvRowSlice = new double[rowAwidth * colWidth];
    recvColSlice = new double[rowBwidth * colWidth];

    // We only need to fill the portion at row_rank, col_rank
    MPI_Comm_split(MPI_COMM_WORLD, procRow, proc_rank, &row_communicator);
    MPI_Comm_split(MPI_COMM_WORLD, procCol, proc_rank, &col_communicator);
    MPI_Comm_rank(row_communicator, &row_rank);
    MPI_Comm_rank(col_communicator, &col_rank);
}

void algorithm() {
    MPI_Status stat;

    // SUMMA algorithm
    for(int i = 0; i < p; i++) {
        double *rbuf, *cbuf;

        if(i == row_rank) {
            rbuf = rowSlice;
        }
        else {
            rbuf = recvRowSlice;
        }
        MPI_Bcast((void*) rbuf, rowAwidth * colWidth, MPI_DOUBLE, i, row_communicator);

        if(i == row_rank) {
            cbuf = colSlice;
        }
        else {
            cbuf = recvColSlice;
        }

        // Should overlap communication and computation, but not doing so yet...
        MPI_Bcast((void*) cbuf, rowBwidth * colWidth, MPI_DOUBLE, i, col_communicator);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, rowAwidth, rowBwidth, colWidth, 1., rbuf, 1, cbuf, 1, 1., result, 1);
    }
}


void finalize2D() { 
    free(rowSlice);
    free(colSlice);

    free(recvRowSlice); 
    free(recvColSlice); 
}