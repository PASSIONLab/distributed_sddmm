#include "CombBLAS/CombBLAS.h"
#include "common.h"
#include <string>
#include <memory>
#include <cmath>

using namespace std;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    {
    string input_file(argv[1]);
    string output_file(argv[2]);

    MPI_Comm WORLD;
    MPI_Comm_dup(MPI_COMM_WORLD, &WORLD);

    int proc_rank, num_procs;
    MPI_Comm_rank(WORLD, &proc_rank);
    MPI_Comm_size(WORLD, &num_procs);

    PSpMat_s32p64_Int * G; 
    uint64_t nnz;

    shared_ptr<CommGrid> simpleGrid;

    int sqrtp = (int) sqrt(num_procs);
    simpleGrid.reset(new CommGrid(WORLD, sqrtp, sqrtp));

    G = new PSpMat_s32p64_Int(simpleGrid);
    G->ParallelReadMM(input_file, true, maximum<double>());	

    nnz = G->getnnz();
    if(proc_rank == 0) {
        cout << "File reader read " << nnz << " nonzeros." << endl;
    }

    if(proc_rank == 0) {
        cout << "Starting random permutation!" << endl; 
    }

    FullyDistVec<int64_t, int64_t> p( G->getcommgrid());
    FullyDistVec<int64_t, int64_t> q( G->getcommgrid());
    p.iota(G->getnrow(), 0);
    q.iota(G->getncol(), 0);

    p.RandPerm();
    q.RandPerm();

    (*G)(p,q,true);// in-place permute to save memory


    if(proc_rank == 0) {
        cout << "Finished random permutation!" << endl;
    }

    G->ParallelWriteMM(output_file, true);
    }
    MPI_Finalize();
}
