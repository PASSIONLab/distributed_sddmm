#ifndef IO_UTILS
#define IO_UTILS

#include "common.h"
#include "CombBLAS/CombBLAS.h"

using namespace Eigen;

// Requirement: Only the processes participating in the sparse matrix
// read should enter either of these function calls.

// The random matrix is Erdos-Renyi with the specified number of nonzeros
// per row. All parameters except the first two are output parameters

// TODO: Add a parameter that can sort the nonzeros  
void generateRandomMatrix(int logM, int nnz_per_row
    vector<int64_t> *rCoords,
    vector<int64_t> *cCoords,
    VectorXd* Svalues 
) {

    PSpMat_s32p64_Int * G; 

    DistEdgeList<int64_t> * DEL = new DistEdgeList<int64_t>(grid->GetCommGridLayer());

    double initiator[4] = {0.25, 0.25, 0.25, 0.25};
    unsigned long int scale      = logM;

    DEL->GenGraph500Data(initiator, scale, nnz_per_row);
    PermEdges(*DEL);
    RenameVertices(*DEL);	
    G = new PSpMat_s32p64_Int(*DEL, false);

    int64_t total_nnz = G->getnnz(); 

    *local_nnz = G->seq().getnnz();
    localSrows = G->seq().getnrow();
    delete DEL;

    rCoords->resize(local_nnz);
    cCoords->resize(local_nnz);

    new (Svalues) VectorXd(local_nnz);

    SpTuples<int64_t,int> tups(G->seq()); 
    tups.SortColBased();

    tuple<int64_t, int64_t, int>* values = tups.tuples;  
    
    for(int i = 0; i < tups.getnnz(); i++) {
        (*rCoords)[i] = get<0>(values[i]);
        (*cCoords)[i] = get<1>(values[i]); 
    }

    // TODO: Fill Svalues here!
    delete G;
}


#endif