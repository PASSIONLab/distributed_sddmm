#include <iostream>
#include <vector>
#include <utility>
#include <string.h>
#include <chrono>
#include <ctime>
#include <mpi.h>

#include "sddmm.h"
#include "15D_sparse.h"
#include "erdos_gen.h"

using namespace std;

int main(int argc, char** argv) {
    // Arguments:

    // 1. The side-length of the square sparse matrix
    // 2. # of NNZ per row (Erdos-Renyi)
    // 3. Embedding size (r-value)
    // 4. Replication factor

    MPI_Init(&argc, &argv);

    int proc_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

    int num_procs;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    benchmark15D(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]));

    MPI_Finalize();
}