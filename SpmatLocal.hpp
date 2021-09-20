#pragma once

#include <iostream>
#include <memory>
#include <numeric>
#include <algorithm>
#include <iterator>
#include <algorithm>
#include <mkl_spblas.h>
#include <mpi.h>
#include "common.h"
#include "CombBLAS/CombBLAS.h"

using namespace Eigen;
using namespace combblas;
using namespace std;

#define TAG_MULTIPLIER 10000

/**
 * Some notes about ParallelReadMM given a 2D grid:
 * - It re-indexes the local sparse matrices 
 * - The trailing processor along each dimension is slightly larger than
 *   the other processors. 
 */

/*
 * This class handles block distributions. 
 */
class NonzeroDistribution {
public:
	MPI_Comm world;

	int rows_in_block, cols_in_block;

	virtual int blockOwner(int row_block, int col_block) = 0;

	/*
	 * Returns the processor that is supposed to own a particular nonzero. 
	 */
	int getOwner(int r, int c, int transpose) {
		if(! transpose) {
			return blockOwner(r / rows_in_block, c / cols_in_block);	
		}
		else {
			return blockOwner(c / rows_in_block, r / cols_in_block);
		}
	}
};

class CSRHandle {
public:
	vector<double> values;
	vector<MKL_INT> col_idx;
	vector<MKL_INT> rowStart;
	sparse_matrix_t mkl_handle;
};

class CSRLocal {
public:
	MKL_INT rows, cols;

	bool transpose;
	int max_nnz;
	int num_coords; 

	int active;
	CSRHandle buffer[2];

	CSRLocal(MKL_INT rows, MKL_INT cols, MKL_INT max_nnz, spcoord_t* coords, int num_coords, bool transpose) {
		this->rows = rows;
		this->cols = cols;
		this->transpose = transpose;
		this->num_coords = num_coords;

		// TODO: Possible problem when num_coords == 0?

		if(transpose) {
			for(int i = 0; i < num_coords; i++) {
				int temp = coords[i].r;
				coords[i].r = coords[i].c;
				coords[i].c = temp;
			}
		}

		std::sort(coords, coords + num_coords, row_major); 
		active = 0;

		for(int t = 0; t < 2; t++) {
			buffer[t].values.resize(max_nnz);
			buffer[t].col_idx.resize(max_nnz);
			buffer[t].rowStart.resize(rows + 1);

			int row = 1;
			buffer[t].rowStart[0] = 0;
			buffer[t].rowStart[rows + 1] = num_coords;
			for(int i = 0; i < num_coords; i++) {
				while(coords[i].r >= row) {
					buffer[t].rowStart[row] = i;
					row++;
				}

				buffer[t].values[i] = coords[i].value;
				buffer[t].col_idx[i] = coords[i].c;
			}

			mkl_sparse_d_create_csr(&(buffer[t].mkl_handle), 
					SPARSE_INDEX_BASE_ZERO,
					rows,
					cols,
					buffer[t].rowStart.data(),
					buffer[t].rowStart.data() + 1,
					buffer[t].col_idx.data(),
					buffer[t].values.data()	
					);
		}
	}

	/*
	 * Note: Input tag should be no greater than 10,000
	 */
	void shiftSparseMatrix(int src, int dst, MPI_Comm comm, int nnz_to_receive, int tag) {
		CSRHandle* send = buffer + active;
		CSRHandle* recv = buffer + 1 - active;

		MPI_Request vRequestSend, cRequestSend, rRequestSend;
		MPI_Request vRequestReceive, cRequestReceive, rRequestReceive;
		MPI_Status stat;

		MPI_Isend(send->values.data(), num_coords, MPI_DOUBLE, dst, tag * TAG_MULTIPLIER, comm, &vRequestSend);
		MPI_Isend(send->col_idx.data(), num_coords, MPI_LONG, dst, tag * TAG_MULTIPLIER + 1, comm, &cRequestSend);
		MPI_Isend(send->rowStart.data(), rows, MPI_LONG, dst, tag * TAG_MULTIPLIER + 2, comm, &rRequestSend);

		MPI_Irecv(recv->values.data(), nnz_to_receive, MPI_DOUBLE, src, tag * TAG_MULTIPLIER, comm, &vRequestReceive);
		MPI_Irecv(recv->col_idx.data(), nnz_to_receive, MPI_LONG, src, tag * TAG_MULTIPLIER + 1, comm, &cRequestReceive);
		MPI_Irecv(recv->rowStart.data(), rows, MPI_LONG, src, tag * TAG_MULTIPLIER + 2, comm, &rRequestReceive);

		MPI_Wait(&vRequestSend, &stat);
		MPI_Wait(&cRequestSend, &stat);
		MPI_Wait(&rRequestSend, &stat);
		MPI_Wait(&vRequestReceive, &stat);
		MPI_Wait(&cRequestReceive, &stat);
		MPI_Wait(&rRequestReceive, &stat);

		active = 1 - active;
	}

	CSRHandle* getActive() {
		return buffer + active;
	}
};


class SpmatLocal {
public:
	// This is redundant, but it makes coding more convenient.
	// These are unzipped versions of the sparse matrix G. 
	vector<spcoord_t> coords;	

	/*
	 * Global properties of the distributed sparse matrix. 
	 */
    uint64_t M;
    uint64_t N;
    uint64_t dist_nnz;

	bool initialized;

	// These are more specialized parameters

    // A contiguous interval of coordinates that this processor is responsible for in its input;
    // need to duplicate this for the transpose. 

    int owned_coords_start, owned_coords_end, nnz_buffer_size;
	bool coordinate_ownership_initialized;
	bool csr_initialized;

    vector<uint64_t> blockStarts;
	vector<CSRLocal> csr_blocks;

	SpmatLocal() {
		initialized = false;
		coordinate_ownership_initialized = false;
		csr_initialized = false;
	}

	/*
	 * We do not support shifting multiple blocks of nonzeros owned by processors, only
	 * a single block. 
	 */
	void initializeCSRBlocks(int blockRows, int blockCols, int max_nnz, bool transpose) {
		if(blockStarts.size() == 0) {
			csr_blocks.emplace_back(blockRows, blockCols, max_nnz, coords.data(), coords.size(), transpose);
		}
		else {
			for(int i = 0; i < blockStarts.size(); i++) {
				int num_coords = blockStarts[i + 1] - blockStarts[i]; 	
				csr_blocks.emplace_back(blockRows, blockCols, num_coords, coords.data() + blockStarts[i], num_coords, transpose);	
			}
		}
		csr_initialized = true;
	}

	void own_all_coordinates() {
		owned_coords_start = 0;
		owned_coords_end = coords.size();
		nnz_buffer_size = coords.size();

		coordinate_ownership_initialized = true;
	}

	void shard_across_layers(int num_layers, int current_layer) {
        vector<int> ccount_in_layer;
        int share = divideAndRoundUp(coords.size(), num_layers);
        for(int i = 0; i < coords.size(); i += share) {
            if(share < coords.size() - i) {
                ccount_in_layer.push_back(share);
            }
            else {
                ccount_in_layer.push_back(coords.size() - i);
            }
        }
        owned_coords_start = share * current_layer;
        owned_coords_end = owned_coords_start + ccount_in_layer[current_layer];
        nnz_buffer_size = share;

		coordinate_ownership_initialized = true;
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

		std::sort((result->coords).begin(), (result->coords).end(), column_major);
		delete[] sendbuf;

		return result;
	}

	/*
	 * Distributes tuples arbitrarily among all processors.
	 */
	void loadTuples(bool readFromFile, 
			int logM, 
			int nnz_per_row,
			string filename) {

		MPI_Comm WORLD;
		MPI_Comm_dup(MPI_COMM_WORLD, &WORLD);

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

	// TODO: Enable setting values to a constant. 
	void setCoordValues(VectorXd &values) {
		assert(values.size() == coords.size());

		for(int i = 0; i < values.size(); i++) {
			coords[i].value = values[i];
		}
	}	

	void setCSRValues(VectorXd &values) {
		if(blockStarts.size() == 0) {
			for(int i = 0; i < values.size(); i++) {
				csr_blocks[0].getActive()->values[i] = values[i];
			}
		}
		else {
			int currentBlock = 0;
			for(int i = 0; i < values.size(); i++) {
				while(i >= blockStarts[i+1]) {
					currentBlock++;
				}
				csr_blocks[currentBlock].getActive()->values[i - blockStarts[currentBlock]] = values[i];
			}
		}
	}

	VectorXd getCoordValues() {
		VectorXd values = VectorXd::Constant(coords.size(), 0.0);

		for(int i = 0; i < coords.size(); i++) {
			values[i] = coords[i].value;
		}

		return values;
	}

	VectorXd getCSRValues() {
		VectorXd values;
		if(blockStarts.size() == 0) {
			values = VectorXd::Constant(csr_blocks[0].num_coords, 0.0);
			for(int i = 0; i < values.size(); i++) {
				values[i] = csr_blocks[0].getActive()->values[i]; 
			}
		}
		else {
			values = VectorXd::Constant(coords.size(), 0.0);
			int currentBlock = 0;
			for(int i = 0; i < values.size(); i++) {
				while(i >= blockStarts[i+1]) {
					currentBlock++;
				}
				values[i] = csr_blocks[currentBlock].getActive()->values[i - blockStarts[currentBlock]]; 
			}
		}
		return values;
	}

	void setValuesConstant(double cval) {
		for(int i = 0; i < coords.size(); i++) {
			coords[i].value = cval;
		}
	}
};

