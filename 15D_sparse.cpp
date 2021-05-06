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

// This code implements a 1.5D Sparse Matrix Multiplication Algorithm

// CombBLAS includes 
#include <memory>
#include "CombBLAS/CombBLAS.h"

using namespace std;
using namespace combblas;

#define EDGEFACTOR 16

typedef SpParMat < int64_t, int, SpDCCols<int64_t,int> > PSpMat_s32p64_Int;

class Sparse15D {
public:
    // Matrix Dimensions, K is the inner dimension
    int M;
    int N;
    int K;

    int p; // Total number of processes 
    int c; // Number of Layers

    int proc_rank;     // Global process rank

    // Communicators and grids
    unique_ptr<CommGrid3D> grid;

    Sparse15D(int M, int N, int K, int c) {
        this->M = M;
        this->N = N;
        this->K = K;
        this->c = c;

        MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &p);

        // STEP 1: Make sure the replication factor is valid for the number
        // of processes running
        if(p % (c * c) != 0) {
            if(proc_rank == 0) {
                cout << "Error, for 1.5D algorithm, must have c^2 divide num_procs!" << endl;
                exit(1);
            }
        }

        // Step 2: Create a communication grid
        grid.reset(new CommGrid3D(MPI_COMM_WORLD, c, p / c, 1));

        // Step 3: Use the R-Mat generator to create a distributed
        // edge list. Only the bottom-most layer needs to do the
        // generation, we can broadcast it to everybody else 

        if(grid->GetRankInFiber() == 0) {
            DistEdgeList<int64_t> * DEL = new DistEdgeList<int64_t>(grid->GetCommGridLayer());

            double initiator[4] = {0.25, 0.25, 0.25, 0.25};
            unsigned long int scale      = 6;
            unsigned long int edgeFactor = 16;

            DEL->GenGraph500Data(initiator, scale, EDGEFACTOR);
            PermEdges(*DEL);
            RenameVertices(*DEL);	
            PSpMat_s32p64_Int * G = new PSpMat_s32p64_Int(*DEL, false); 

            SpDCCols<int64_t,int> cols = G->seq(); 
            SpTuples<int64_t,int> tups(cols); 
            tups.SortColBased();

            tuple<int64_t, int64_t, int>* values = tups.tuples;  
            
            /*if(grid->GetRankInLayer() == 3) {
                for(int i = 0; i < tups.getnnz(); i++) {
                    cout << std::get<0>(values[i]) << " " << std::get<0>(values[i]) << endl; 
                }
            }*/


            delete DEL;
        } 



        // EList_t* G = new Elist(*DEL, false);

        // Step 4: Reorganize the local entries of the sparse matrix
        // and broadcast along the fiber. 
        
        
        

        if(proc_rank == 0) {
            cout << "Initialization complete!" << endl; 
        }
    };

    ~Sparse15D() {
        // Destructor
    }
};

void benchmark15D(int M, int N, int K, int c, char* filename) {
    Sparse15D(M, N, K, c);
}

/*void setup15D(int M_loc, int N_loc, int K_loc, int c_loc, char* filename) {
    M = M_loc;
    N = N_loc;
    K = K_loc;
    c = c_loc;

    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

    int num_procs;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    p = (int) (num_procs / c);

    // STEP 1: Make sure the replication factor is valid for the number
    // of processes running
    if(num_procs % (c * c) != 0) {
        if(proc_rank == 0) {
            cout << "Error, for 1.5D algorithm, must have c^2 divide num_procs!" << endl;
            exit(1);
        }
    }

    // Step 2: split the processes into layers and establish a 2D
    // grid on each layer 
    rank_in_fiber = proc_rank % c;
    rank_in_layer = proc_rank / c;
    MPI_Comm_split(MPI_COMM_WORLD, rank_in_layer,   rank_in_fiber, &interlayer_communicator);
    MPI_Comm_split(MPI_COMM_WORLD, rank_in_fiber,   rank_in_layer, &intralayer_communicator);

}*/

