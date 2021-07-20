#pragma once

#include <iostream>
#include <memory>
#include <numeric>
#include <algorithm>
#include <iterator>
#include <algorithm>
#include "common.h"
#include "CombBLAS/CombBLAS.h"

using namespace Eigen;
using namespace combblas;
using namespace std;

bool sortbycolumns(spcoord_t &a, spcoord_t &b) {
    if(a.c == b.c) {
        return a.r < b.r;
    }
    else {
        return a.c < b.c;
    }
}

class NonzeroDistribution {
	int M, N;

	/*
	 * Returns the processor that is supposed to own a particular nonzero. 
	 */
	virtual void getOwner(int r, int c, int transpose) = 0;
}

class SpmatLocal {
public:
	// This is redundant, but it makes coding more convenient.
	// These are unzipped versions of the sparse matrix G.
	vector<spcoord_t> coords;	
	VectorXd Svalues;

    vector<uint64_t> blockStarts;

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
	 * This is a bad prefix sum function.
	 */
	void prefix_sum(vector<int> &values, vector<int> &offsets) {
		int sum = 0;
		for(int i = 0; i < values.size(); i++) {
			offsets.push_back(sum);
			sum += values[i];
		}
	}

	/*
	 * Redistributes nonzeros according to the provided distribution, optionally transposing the matrix
	 * in the process. Returns a new sparse matrix with the redistributed nonzeros. 
	 */
	SpmatLocal* redistribute_nonzeros(NonzeroDistribution* dist, bool transpose) {
		int num_procs;
		MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

		vector<int> sendcounts(num_procs, 0);
		vector<int> recvcounts(num_procs, 0);

		vector<int> offsets, bufindices;

		spcoord_t* sendbuf = new spcoord_t[coords.size()];

		for(int i = 0; i < coords.size(); i++) {
			sendcounts[dist->getOwner(coords[i].r, coords[i].c, transpose)]++;
		}
		prefix_sum(sendcounts, offsets);
		bufindices = offsets;

		for(int i = 0; i < coords.size(); i++) {
			int owner = dist->getOwner(coords[i].r, coords[i].c, transpose);
			sendbuf[bufindices[owner]] = coords[i];

			if(transpose) {
				sendbuf[bufindices[owner]]	
			}

			bufindices[owner]++;
		}

		// Broadcast the number of nonzeros that each processor is going to receive
		MPI_Alltoall(sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1, 
				MPI_INT, MPI_COMM_WORLD);

		vector<int> recvoffsets;
		prefix_sum(recvcounts, recvoffsets);

		// Use the sizing information to execute an AlltoAll 
		int total_received_coords = 
				std::accumulate(recvcounts.begin(), recvcounts.end(), 0);

		SpmatLocal* result = new SpmatLocal();
		result.coords.resize(total_received_coords);

		MPI_Alltoallv(sendbuf, sendcounts.data(), offsets.data(), 
				SPCOORD, result->coords.data(), recvcounts.data(), recvoffsets.data(), 
				MPI_INT, MPI_COMM_WORLD
				);

		std::sort(result->coords.begin(), result->coords.end(), sortbycolumns);		
		result->initialized = true;

		return result;
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
	void divideIntoBlockCols(int blockWidth, int targetDivisions, bool modIndex) {
		blockStarts.clear();
        // Locate block starts within the local sparse matrix (i.e. divide a long
        // block row into subtiles) 
        int currentStart = 0;
        for(uint64_t i = 0; i < local_nnz; i++) {
            while(coords[i].c >= currentStart) {
                blockStarts.push_back(i);
                currentStart += blockWidth;
            }

            // This modding step helps indexing. 

			if(modIndex) {
				coords[i].c %= blockWidth;
			}
        }

		assert(blockStarts.size() <= targetDivisions + 1);

        while(blockStarts.size() < targetDivisions + 1) {
            blockStarts.push_back(local_nnz);
        }
	}

};


