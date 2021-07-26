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

/**
 * Some notes about ParallelReadMM given a 2D grid:
 * - It re-indexes the local sparse matrices 
 * - The trailing processor along each dimension is slightly larger than
 *   the other processors. 
 */

class NonzeroDistribution {
public:
	int M, N;
	MPI_Comm world;
	/*
	 * Returns the processor that is supposed to own a particular nonzero. 
	 */
	virtual int getOwner(int r, int c, int transpose) = 0;
};

class SpmatLocal {
public:
	// This is redundant, but it makes coding more convenient.
	// These are unzipped versions of the sparse matrix G. 
	vector<spcoord_t> coords;	
    vector<uint64_t> blockStarts;

	/*
	 * Global properties of the distributed sparse matrix. 
	 */
    uint64_t M;
    uint64_t N;
    uint64_t dist_nnz;

	bool initialized;

	SpmatLocal() {
		initialized = false;
	}

	void unpack_tuples( SpTuples<int64_t,int> &tups, 
			vector<spcoord_t> &unpacked) {
		tuple<int64_t, int64_t, int>* values = tups.tuples;

		new (&unpacked) vector<spcoord_t>; 
		unpacked.resize(tups.getnnz());

		for(int i = 0; i < tups.getnnz(); i++) {
			unpacked[i].r = get<0>(values[i]);
			unpacked[i].c = get<1>(values[i]); 
			unpacked[i].value = get<2>(values[i]); 
		}
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
	 * in the process. Works either in-place, or returns an entirely new sparse matrix. 
	 *
	 * TODO: Write extra code to amortize away the process of a transpose. 
	 */
	SpmatLocal* redistribute_nonzeros(NonzeroDistribution* dist, bool transpose, bool in_place) {
		int num_procs, proc_rank;
		MPI_Comm_size(dist->world, &num_procs);	
		MPI_Comm_rank(dist->world, &proc_rank);

		dist->M = M;
		dist->N = N;

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
			int idx = bufindices[owner];
	
			sendbuf[idx].r = transpose ? coords[i].c : coords[i].r;
			sendbuf[idx].c = transpose ? coords[i].r : coords[i].c;	
			sendbuf[idx].value = coords[i].value;	

			bufindices[owner]++;
		}

		// Broadcast the number of nonzeros that each processor is going to receive
		MPI_Alltoall(sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1, 
				MPI_INT, dist->world);

		vector<int> recvoffsets;
		prefix_sum(recvcounts, recvoffsets);

		// Use the sizing information to execute an AlltoAll 
		int total_received_coords = 
				std::accumulate(recvcounts.begin(), recvcounts.end(), 0);

		SpmatLocal* result;

		if(in_place) {
			result = this;
		}
		else {
			result = new SpmatLocal();
		}

		result->M = transpose ? this->N : this->M; 
		result->N = transpose ? this->M : this->N; 
		result->dist_nnz = this->dist_nnz; 

		result->initialized = true;
		(result->coords).resize(total_received_coords);

		MPI_Alltoallv(sendbuf, sendcounts.data(), offsets.data(), 
				SPCOORD, (result->coords).data(), recvcounts.data(), recvoffsets.data(), 
				SPCOORD, dist->world 
				);

		std::sort((result->coords).begin(), (result->coords).end(), sortbycolumns);

		delete[] sendbuf;

		return result;
	}

	void loadTuples(bool readFromFile, 
			int logM, 
			int nnz_per_row,
			string filename) {

		MPI_Comm WORLD;
		MPI_Comm_dup(dist->world, &WORLD);

		int proc_rank, num_procs;
		MPI_Comm_rank(WORLD, &proc_rank);
		MPI_Comm_size(WORLD, &num_procs);

		shared_ptr<CommGrid> simpleGrid;
		simpleGrid.reset(new CommGrid(WORLD, num_procs, 1));

		PSpMat_s32p64_Int * G; 
		int nnz;

		if(readFromFile) {
			G = new PSpMat_s32p64_Int(simpleGrid);
			// This has a load-balancing problem...	
			G->ParallelReadMM(filename, true, maximum<double>());	

			nnz = G->getnnz();
			if(proc_rank == 0) {
				cout << "File reader read " << nnz << " nonzeros." << endl;
			}
		}
		else { // This uses the Graph500 R-mat generator 
			DistEdgeList<int64_t> * DEL = new DistEdgeList<int64_t>(simpleGrid);

			double initiator[4] = {0.25, 0.25, 0.25, 0.25};
			unsigned long int scale      = logM;

			DEL->GenGraph500Data(initiator, scale, nnz_per_row);
			PermEdges(*DEL);
			RenameVertices(*DEL);	
			G = new PSpMat_s32p64_Int(*DEL, false);

			delete DEL;

			nnz = G->getnnz();
			if(proc_rank == 0) {
				cout << "R-mat generator created " << nnz << " nonzeros." << endl;
			}	
		}	
	
		SpTuples<int64_t,int> tups(G->seq()); 

		unpack_tuples(tups, coords);
		
		this->M = G->getnrow();
		this->N = G->getncol();
		this->dist_nnz = nnz; 
		
		int rowIncrement = this->M / num_procs;
		for(int i = 0; i < coords.size(); i++) {
			coords[i].r += rowIncrement * proc_rank;
		}

		initialized = true;
		delete G;
	}

	/*
	 * This method assumes the tuples are sorted in a column major order,
	 * and it also changes the column coordinates. DO NOT call this function
	 * unless you're sure there is no more work to do with the current sparse
	 * matrix. 
	 */
	void divideIntoBlockCols(int blockWidth, int targetDivisions, bool modIndex) {
		blockStarts.clear();
        // Locate block starts within the local sparse matrix (i.e. divide a long
        // block row into subtiles) 
        int currentStart = 0;
        for(uint64_t i = 0; i < coords.size(); i++) {
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
            blockStarts.push_back(coords.size());
        }
	}
};

