#ifndef IO_UTILS
#define IO_UTILS

#include "common.h"
#include "CombBLAS/CombBLAS.h"

using namespace Eigen;
using namespace combblas;
using namespace std;

// Requirement: Only the processes participating in the sparse matrix
// read should enter either of these function calls.

// The random matrix is Erdos-Renyi with the specified number of nonzeros
// per row. All parameters except the first two are output parameters

void spmat_to_coord_arrays(PSpMat_s32p64_Int *G, spmat_local_t &output, VectorXd &Svalues) {
    // Everything after this point is just unpacking into a convenient format;
    // Also: should look into using the RowSplit(), colSplit() functions... 
    output.dist_nnz = G->getnnz();
    output.local_nnz = G->seq().getnnz();
    output.nrows = G->seq().getnrow();
    output.ncols = G->seq().getncol();

    output.distrows = G->getnrow();
    output.distcols = G->getncol();

    new (&(output.rCoords)) vector<int64_t>; 
    new (&(output.cCoords)) vector<int64_t>; 
    new (&Svalues) VectorXd(output.local_nnz);

    output.rCoords.resize(output.local_nnz);
    output.cCoords.resize(output.local_nnz);

    SpTuples<int64_t,int> tups(G->seq()); 
    tups.SortColBased();

    tuple<int64_t, int64_t, int>* values = tups.tuples;
    
    for(int i = 0; i < tups.getnnz(); i++) {
        output.rCoords[i] = get<0>(values[i]);
        output.cCoords[i] = get<1>(values[i]); 
        Svalues(i) = get<2>(values[i]); 
    }
}


// TODO: Add a parameter that can sort the nonzeros  
void generateRandomMatrix(int logM, 
    int nnz_per_row,
    shared_ptr<CommGrid> layerGrid,
    spmat_local_t &output,
    VectorXd &Svalues
) {

    PSpMat_s32p64_Int * G; 
    DistEdgeList<int64_t> * DEL = new DistEdgeList<int64_t>(layerGrid);

    double initiator[4] = {0.25, 0.25, 0.25, 0.25};
    unsigned long int scale      = logM;

    DEL->GenGraph500Data(initiator, scale, nnz_per_row);
    PermEdges(*DEL);
    RenameVertices(*DEL);	
    G = new PSpMat_s32p64_Int(*DEL, false);

    spmat_to_coord_arrays(G, output, Svalues);

    delete DEL;
    delete G;
}

/*
 * Currently, this assumes one-based indexing. 
 */
void loadMatrixFromFile(
    string &filename,
    shared_ptr<CommGrid> layerGrid,
    spmat_local_t &output,
    VectorXd &Svalues) {
    
    PSpMat_s32p64_Int *G = new PSpMat_s32p64_Int(layerGrid);
    G->ParallelReadMM(filename, true, maximum<double>());
    spmat_to_coord_arrays(G, output, Svalues);

    delete G;
}


#endif