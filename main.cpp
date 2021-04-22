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
    MPI_Init(&argc, &argv);

    int proc_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

    /*if(argc < 3) {
        if(proc_rank == 0) {
            cout << "Usage: Provide filename of matrix market format, r value." << endl;
        }
        return 1; 
    }*/
    
    // test1DCorrectness(); 

    if(proc_rank == 0) {
        generateMatrix(2000, 2000, "test.txt", 100);
    }


    benchmark15D(2000, 100, 2000, 2, "test.txt");

    MPI_Finalize();
}