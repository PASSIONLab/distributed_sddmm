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

    /*if(argc < 3) {
        if(proc_rank == 0) {
            cout << "Usage: Provide filename of matrix market format, r value." << endl;
        }
        return 1; 
    }*/
    
    // test1DCorrectness(); 

    int spMatDim = 10000/num_procs * num_procs;

    //if(proc_rank == 0) {
    //    generateMatrix(spMatDim, spMatDim, "test.txt", atoi(argv[2]));
    //}

    /*generateMatrix(spMatDim, spMatDim, "0p005.txt", (int) (0.005 * spMatDim));
    cout << "Done 1" << endl;
    generateMatrix(spMatDim, spMatDim, "0p010.txt", (int) (0.010 * spMatDim));
    cout << "Done 2" << endl;
    generateMatrix(spMatDim, spMatDim, "0p020.txt", (int) (0.020 * spMatDim));
    cout << "Done 3" << endl;
    generateMatrix(spMatDim, spMatDim, "0p040.txt", (int) (0.040 * spMatDim));
    cout << "Done 4" << endl;
    generateMatrix(spMatDim, spMatDim, "0p080.txt", (int) (0.080 * spMatDim));
    cout << "Done 5" << endl;
    generateMatrix(spMatDim, spMatDim, "0p100.txt", (int) (0.100 * spMatDim));
    */

    benchmark15D(spMatDim, atoi(argv[1]), spMatDim, atoi(argv[2]), "0p010.txt");

    MPI_Finalize();
}