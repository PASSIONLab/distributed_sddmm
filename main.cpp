#include <iostream>
#include <vector>
#include <utility>
#include <string.h>
#include <chrono>
#include <ctime>
#include <mpi.h>

#include "sddmm.h"
#include "spmat_reader.h"

using namespace std;

int proc_rank;
int num_procs;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

    if(argc < 3) {
        if(proc_rank == 0) {
            cout << "Usage: Provide filename of matrix market format, r value." << endl;
        }
        return 1; 
    }

    vector<pair<size_t, size_t>> coordinates 
        = read_sparse_matrix_fraction(proc_rank, argv[1], 0, 5, 0, 5);


    size_t r = (size_t) atoi(argv[2]);

    MPI_Finalize();
}