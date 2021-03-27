#include <iostream>
#include <vector>
#include <utility>
#include <string.h>
#include <chrono>
#include <ctime>
#include <mpi.h>

#include "sddmm.h"
#include "2D.h"

using namespace std;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    /*if(argc < 3) {
        if(proc_rank == 0) {
            cout << "Usage: Provide filename of matrix market format, r value." << endl;
        }
        return 1; 
    }*/

    // For now I'm going to use this to test the dense matrix multiplication algorithms
    int M = 1000;
    int N = 1000;
    int K = 1000;

    setup2D(M, N, K);
    algorithm();

    MPI_Finalize();
}