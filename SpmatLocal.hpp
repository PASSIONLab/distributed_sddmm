#pragma once

#include <iostream>
#include "common.h"
#include "CombBLAS/CombBLAS.h"

using namespace Eigen;
using namespace combblas;
using namespace std;

class SpmatLocal {
public:
	// This is redundant, but it makes coding more convenient.
	// These are unzipped versions 
    vector<uint64_t> rCoords;
    vector<uint64_t> cCoords;
	VectorXd Svalues;

    uint64_t local_nnz;
    uint64_t dist_nnz;

    uint64_t nrows_local;
    uint64_t ncols_local;

    uint64_t M;
    uint64_t N;

	bool initialized;	

	SpmatLocal() {
		initialized = false;
	}

	void initialize(PSpMat_s32p64_Int *G) {	
		dist_nnz = G->getnnz();
		local_nnz = G->seq().getnnz();
		nrows_local = G->seq().getnrow();
		ncols_local = G->seq().getncol();

		M = G->getnrow();
		N = G->getncol();

		new (&rCoords) vector<int64_t>; 
		new (&cCoords) vector<int64_t>; 
		new (&Svalues) VectorXd(local_nnz);

		rCoords.resize(local_nnz);
		cCoords.resize(local_nnz);

		SpTuples<int64_t,int> tups(G->seq()); 
		tups.SortColBased();

		tuple<int64_t, int64_t, int>* values = tups.tuples;
		
		for(int i = 0; i < tups.getnnz(); i++) {
			rCoords[i] = get<0>(values[i]);
			cCoords[i] = get<1>(values[i]); 
			Svalues(i) = get<2>(values[i]); 
		}
		initialized=true;
	}

	/*
	 * Replicates sparse matrix values and indices across layers
	 */
	void broadcast_synchronize(int rankInWorld, int rootInWorld, MPI_Comm world) {	
        MPI_Bcast(&dist_nnz, 1, MPI_UINT64_T, 0, world);
        MPI_Bcast(&local_nnz, 1, MPI_UINT64_T, 0, world);
        MPI_Bcast(&nrows_local, 1, MPI_UINT64_T, 0, world);
        MPI_Bcast(&ncols_local, 1, MPI_UINT64_T, 0, world);

        MPI_Bcast(&M, 1, MPI_UINT64_T, 0, world);
        MPI_Bcast(&N, 1, MPI_UINT64_T, 0, world);

		if(rankInWorld != rootInWorld) {
			rCoords.resize(local_nnz);
			cCoords.resize(local_nnz);
			Svalues.resize(local_nnz);
		}
 
        MPI_Bcast(rCoords.data(), local_nnz, MPI_UINT64_T, 0, world);
        MPI_Bcast(cCoords.data(), local_nnz, MPI_UINT64_T, 0, world);
        MPI_Bcast(Svalues.data(), local_nnz, MPI_UINT64_T, 0, world);
		initialized=true;
	}

	static void loadMatrix(bool readFromFile, 
			int logM, 
			int nnz_per_row,
			string filename,
			shared_ptr<CommGrid> layerGrid,
			SpmatLocal* S,
			SpmatLocal* ST
			) {
		int proc_rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);	
		PSpMat_s32p64_Int * G; 

		if(! readFromFile) {
			DistEdgeList<int64_t> * DEL = new DistEdgeList<int64_t>(layerGrid);

			double initiator[4] = {0.25, 0.25, 0.25, 0.25};
			unsigned long int scale      = logM;

			DEL->GenGraph500Data(initiator, scale, nnz_per_row);
			PermEdges(*DEL);
			RenameVertices(*DEL);	
			G = new PSpMat_s32p64_Int(*DEL, false);
			delete DEL;

			if(proc_rank == 0) {
				cout << "R-mat generator created " << S->dist_nnz << " nonzeros." << endl;
			}
		}
		else {
			PSpMat_s32p64_Int *G = new PSpMat_s32p64_Int(layerGrid);
			G->ParallelReadMM(filename, true, maximum<double>());	
			if(proc_rank == 0) {
				cout << "File reader read " << S->dist_nnz << " nonzeros." << endl;
			}
		}

		S->initialize(G);

		delete G;
	}
};
