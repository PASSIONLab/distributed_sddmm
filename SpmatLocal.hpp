#pragma once

#include <iostream>
#include <memory>
#include "common.h"
#include "CombBLAS/CombBLAS.h"

using namespace Eigen;
using namespace combblas;
using namespace std;

class SpmatLocal {
public:
	// This is redundant, but it makes coding more convenient.
	// These are unzipped versions of the sparse matrix G.
	vector<spcoord_t> coords;	
	VectorXd Svalues;

    vector<uint64_t> blockStarts;

	shared_ptr<PSpMat_s32p64_Int> G;

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
		(this->G).reset(G);

		dist_nnz = G->getnnz();
		local_nnz = G->seq().getnnz();
		nrows_local = G->seq().getnrow();
		ncols_local = G->seq().getncol();

		M = G->getnrow();
		N = G->getncol();

		new (&coords) vector<spcoord_t>; 
		new (&Svalues) VectorXd(local_nnz);

		coords.resize(local_nnz);

		SpTuples<int64_t,int> tups(G->seq()); 
		tups.SortColBased();

		tuple<int64_t, int64_t, int>* values = tups.tuples;
		
		for(int i = 0; i < tups.getnnz(); i++) {
			coords[i].r = get<0>(values[i]);
			coords[i].c = get<1>(values[i]); 
			Svalues(i) = get<2>(values[i]); 
		}
		initialized=true;
	}

	/*
	 * Replicates sparse matrix values and indices across layers. TODO: Should
	 * change this to shard the matrix across layers instead!
	 */
	void broadcast_synchronize(int rankInWorld, int rootInWorld, MPI_Comm world) {	
        MPI_Bcast(&dist_nnz, 1, MPI_UINT64_T, 0, world);
        MPI_Bcast(&local_nnz, 1, MPI_UINT64_T, 0, world);
        MPI_Bcast(&nrows_local, 1, MPI_UINT64_T, 0, world);
        MPI_Bcast(&ncols_local, 1, MPI_UINT64_T, 0, world);

        MPI_Bcast(&M, 1, MPI_UINT64_T, 0, world);
        MPI_Bcast(&N, 1, MPI_UINT64_T, 0, world);

		if(rankInWorld != rootInWorld) {
			coords.resize(local_nnz);
			Svalues.resize(local_nnz);
		}
 
        MPI_Bcast(coords.data(), local_nnz, SPCOORD, 0, world);
        MPI_Bcast(Svalues.data(), local_nnz, MPI_UINT64_T, 0, world);
		initialized=true;
	}

	static void loadMatrix(bool readFromFile, 
			int logM, 
			int nnz_per_row,
			string filename,
			shared_ptr<CommGrid> layerGrid,
			SpmatLocal* S
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
			int nnz = G->getnnz();

			if(proc_rank == 0) {
				cout << "R-mat generator created " << nnz << " nonzeros." << endl;
			}

		}
		else {
			G = new PSpMat_s32p64_Int(layerGrid);
			G->ParallelReadMM(filename, true, maximum<double>());	

			int nnz = G->getnnz();

			if(proc_rank == 0) {
				cout << "File reader read " << nnz << " nonzeros." << endl;
			}
		}

		S->initialize(G);	
	}

	/*
	 * This method assumes the tuples are sorted in a column major order,
	 * and it also changes the column coordinates 
	 */
	void divideIntoBlockCols(int blockWidth, int targetDivisions) {
        // Locate block starts within the local sparse matrix (i.e. divide a long
        // block row into subtiles) 
        int currentStart = 0;
        for(uint64_t i = 0; i < local_nnz; i++) {
            while(coords[i].c >= currentStart) {
                blockStarts.push_back(i);
                currentStart += blockWidth;
            }

            // This modding step helps indexing. 
            coords[i].c %= blockWidth;
        }

		assert(blockStarts.size() <= targetDivisions + 1);

        while(blockStarts.size() < targetDivisions + 1) {
            blockStarts.push_back(local_nnz);
        }
	}

	/*
	 * Sets up the coordinates and the permutation for a sparse transpose
	 */
	SpmatLocal* transpose() {
		// TODO: Fill this in!
		return nullptr;	
	}
};


