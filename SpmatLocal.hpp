#pragma once

#include <iostream>
#include <memory>
#include <numeric>
#include <algorithm>
#include <iterator>
#include <algorithm>
#include <parallel/algorithm>
#include <mkl_spblas.h>
#include <mpi.h>
#include <string.h>
#include "common.h"
#include "CombBLAS/CombBLAS.h"

using namespace Eigen;
using namespace combblas;
using namespace std;

#define TAG_MULTIPLIER 10000

typedef enum {csr, coo, both} ShiftMode;

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
	vector<MKL_INT> row_idx;
	sparse_matrix_t mkl_handle;
};

class CSRLocal {
public:
	MKL_INT rows, cols;

	int max_nnz, num_coords;	

	bool transpose;	

	int active;
	CSRHandle* buffer;

	/*
	 * TODO: Need to check this function for memory leaks! 
	 */
	CSRLocal(MKL_INT rows, MKL_INT cols, MKL_INT max_nnz, spcoord_t* coords, int num_coords, bool transpose) {
		this->transpose = transpose;
		this->num_coords = num_coords;
		this->rows = rows;
		this->cols = cols;

		this->buffer = new CSRHandle[2];
	
		// This setup is really clunky, but I don't have time to fix it. 
		vector<MKL_INT> rArray(num_coords, 0.0);
		vector<MKL_INT> cArray(num_coords, 0.0);
		vector<double> vArray(num_coords, 0.0);	

		// Put a dummy value in if the number of coordinates is 0, so that everything doesn't
		// blow up
		if(num_coords == 0) {
			rArray.push_back(0);
			cArray.push_back(0);
			vArray.push_back(0.0);
		}


		#pragma omp parallel for
		for(int i = 0; i < num_coords; i++) {
			rArray[i] = coords[i].r;
			cArray[i] = coords[i].c;
			vArray[i] = coords[i].value;
		}

		sparse_operation_t op;
		if(transpose) {
			op = SPARSE_OPERATION_TRANSPOSE;
		}
		else {
			op = SPARSE_OPERATION_NON_TRANSPOSE;
		}

		sparse_matrix_t tempCOO, tempCSR;

		mkl_sparse_d_create_coo(&tempCOO, SPARSE_INDEX_BASE_ZERO, rows, cols, max(num_coords, 1), rArray.data(), cArray.data(), vArray.data());
		mkl_sparse_convert_csr(tempCOO, op, &tempCSR);

		mkl_sparse_destroy(tempCOO);
		vector<MKL_INT>().swap(rArray);		
		vector<MKL_INT>().swap(cArray);		
		vector<double>().swap(vArray);

		sparse_index_base_t indexing;
		MKL_INT *rows_start, *rows_end, *col_idx;
		double *values;

		mkl_sparse_d_export_csr(tempCSR, 
				&indexing, 
				&(this->rows), 
				&(this->cols),
				&rows_start,
				&rows_end,
				&col_idx,
				&values
				);

		int rv = 0;
		for(int i = 0; i < num_coords; i++) {
			while(rv < this->rows && i >= rows_start[rv + 1]) {
				rv++;
			}
			coords[i].r = rv;
			coords[i].c = col_idx[i];
			coords[i].value = values[i];
		}

		active = 0;

		assert(num_coords <= max_nnz);

		for(int t = 0; t < 2; t++) {
			buffer[t].values.resize(max_nnz == 0 ? 1 : max_nnz);
			buffer[t].col_idx.resize(max_nnz == 0 ? 1 : max_nnz);
			buffer[t].row_idx.resize(max_nnz == 0 ? 1 : max_nnz);
			buffer[t].rowStart.resize(this->rows + 1);

			// Copy over row indices
			#pragma omp parallel for
			for(int i = 0; i < num_coords; i++) {
				buffer[t].row_idx[i] = coords[i].r;
			}

			memcpy(buffer[t].values.data(), values, sizeof(double) * max(num_coords, 1));
			memcpy(buffer[t].col_idx.data(), col_idx, sizeof(MKL_INT) * max(num_coords, 1));
			memcpy(buffer[t].rowStart.data(), rows_start, sizeof(MKL_INT) * this->rows);

			buffer[t].rowStart[this->rows] = max(num_coords, 1);

			mkl_sparse_d_create_csr(&(buffer[t].mkl_handle), 
					SPARSE_INDEX_BASE_ZERO,
					this->rows,
					this->cols,
					buffer[t].rowStart.data(),
					buffer[t].rowStart.data() + 1,
					buffer[t].col_idx.data(),
					buffer[t].values.data()	
					);

			// This madness is just trying to get around the inspector routine
			if(num_coords == 0) {
				buffer[t].rowStart[this->rows] = 0; 
			}
		}

		mkl_sparse_destroy(tempCSR);
	}

	~CSRLocal() {
		for(int t = 0; t < 2; t++) {
			mkl_sparse_destroy(buffer[t].mkl_handle);
		}
		delete[] buffer;
	}

	/*
	 * Note: Input tag should be no greater than 10,000
	 */
	void shiftCSR(int src, int dst, MPI_Comm comm, int nnz_to_receive, int tag, 
			ShiftMode mode) {
		CSRHandle* send = buffer + active;
		CSRHandle* recv = buffer + 1 - active;

		int proc_rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);


		MPI_Status stat;
		int nnz_to_send = num_coords;
		int recv_verify;
        /*MPI_Sendrecv(&nnz_to_send, 1, MPI_INT,
                dst, 0,
                &recv_verify, 1, MPI_INT,
                src, 0,
                comm, &stat);

		assert(recv_verify == nnz_to_receive);*/

		MPI_Request vRequestSend, cRequestSend, ridxRequestSend, rRequestSend;
		MPI_Request vRequestReceive, cRequestReceive, ridxRequestReceive, rRequestReceive;

		MPI_Isend(send->values.data(), num_coords, MPI_DOUBLE, dst, tag * TAG_MULTIPLIER, comm, &vRequestSend);
		MPI_Isend(send->col_idx.data(), num_coords, MPI_LONG, dst, tag * TAG_MULTIPLIER + 1, comm, &cRequestSend);
		
		if(mode == csr || mode == both) {
			MPI_Isend(send->rowStart.data(), rows + 1, MPI_LONG, dst, tag * TAG_MULTIPLIER + 2, comm, &ridxRequestSend);
		}
		if(mode == coo || mode == both) {
			MPI_Isend(send->row_idx.data(), num_coords, MPI_LONG, dst, tag * TAG_MULTIPLIER + 3, comm, &rRequestSend);
		}	
		if(mode == csr || mode == both) {
			MPI_Irecv(recv->rowStart.data(), rows + 1, MPI_LONG, src, tag * TAG_MULTIPLIER + 2, comm, &ridxRequestReceive);
		}
		if(mode == coo || mode == both) {
			MPI_Irecv(recv->row_idx.data(), nnz_to_receive, MPI_LONG, src, tag * TAG_MULTIPLIER + 3, comm, &rRequestReceive);
		}

		MPI_Irecv(recv->values.data(), nnz_to_receive, MPI_DOUBLE, src, tag * TAG_MULTIPLIER, comm, &vRequestReceive);
		MPI_Irecv(recv->col_idx.data(), nnz_to_receive, MPI_LONG, src, tag * TAG_MULTIPLIER + 1, comm, &cRequestReceive);

		MPI_Wait(&vRequestSend, &stat);
		MPI_Wait(&vRequestReceive, &stat);

		MPI_Wait(&cRequestSend, &stat);
		MPI_Wait(&cRequestReceive, &stat);

		if(mode == csr || mode == both) {
			MPI_Wait(&ridxRequestSend, &stat);
			MPI_Wait(&ridxRequestReceive, &stat);
		}
		else if (mode == coo || mode == both) {
			MPI_Wait(&rRequestSend, &stat);
			MPI_Wait(&rRequestReceive, &stat);
		}

		num_coords = nnz_to_receive;
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

    int owned_coords_start, owned_coords_end;
	vector<int> layer_coords_start, layer_coords_sizes;

	bool coordinate_ownership_initialized;
	bool csr_initialized;

    vector<uint64_t> blockStarts;
	vector<CSRLocal*> csr_blocks;

	SpmatLocal() {
		initialized = false;
		coordinate_ownership_initialized = false;
		csr_initialized = false;
	}

	~SpmatLocal() {
		for(int i = 0; i < csr_blocks.size(); i++) {
			if(csr_blocks[i] != nullptr) {
				delete csr_blocks[i];
			}
		}
	}

	/*
	 * We do not support shifting multiple blocks of nonzeros owned by processors, only
	 * a single block. 
	 */
	void initializeCSRBlocks(int blockRows, int blockCols, int max_nnz, bool transpose) {
		if(max_nnz == -1) {
			for(int i = 0; i < blockStarts.size() - 1; i++) {
				int num_coords = blockStarts[i + 1] - blockStarts[i];

				if(num_coords > 0) {
					CSRLocal* block 
							= new CSRLocal(blockRows, blockCols, num_coords, coords.data() + blockStarts[i], num_coords, transpose);					
					csr_blocks.push_back(block);
				}
				else {
					csr_blocks.push_back(nullptr);
				}
			}	
		}
		else {
			int num_coords = blockStarts[1] - blockStarts[0];
			CSRLocal* block = new CSRLocal(blockRows, blockCols, max_nnz, coords.data(), num_coords, transpose);	
			csr_blocks.push_back(block);
		}

		csr_initialized = true;
	}

	void own_all_coordinates() {
		owned_coords_start = 0;
		owned_coords_end = coords.size();

		layer_coords_start.push_back(0);	
		layer_coords_start.push_back(coords.size());	
		layer_coords_sizes.push_back(coords.size());

		coordinate_ownership_initialized = true;
	}

	void shard_across_layers(int num_layers, int current_layer) {
		divideIntoSegments(coords.size(), num_layers, layer_coords_start, layer_coords_sizes);

		owned_coords_start = layer_coords_start[current_layer]; 
		owned_coords_end = layer_coords_start[current_layer + 1]; 

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

		#pragma omp parallel for
		for(int i = 0; i < coords.size(); i++) {
			int owner = dist->getOwner(coords[i].r, coords[i].c, transpose);
			#pragma omp atomic update
			sendcounts[owner]++;
		}
		prefix_sum(sendcounts, offsets);
		bufindices = offsets;

		#pragma omp parallel for
		for(int i = 0; i < coords.size(); i++) {
			int owner = dist->getOwner(coords[i].r, coords[i].c, transpose);

			int idx;
			#pragma omp atomic capture
			idx = bufindices[owner]++;
	
			sendbuf[idx].r = transpose ? coords[i].c : coords[i].r;
			sendbuf[idx].c = transpose ? coords[i].r : coords[i].c;	
			sendbuf[idx].value = coords[i].value;	
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

		// TODO: Parallelize the sort routine?
		//std::sort((result->coords).begin(), (result->coords).end(), column_major);
		__gnu_parallel::sort((result->coords).begin(), (result->coords).end(), column_major);
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
		uint64_t nnz;

		if(readFromFile) {
			G = new PSpMat_s32p64_Int(simpleGrid);
			G->ParallelReadMM(filename, true, maximum<double>());	

			// Apply a random permutation for load balance
			// Taken from CombBLAS (see Aydin's email) 
			//FullyDistVec<int64_t, array<char, MAXVERTNAME> > perm 
			//		= G->ReadGeneralizedTuples(filename, maximum<double>());

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

	void monolithBlockColumn() {
		blockStarts.clear();
		blockStarts.push_back(0);
		blockStarts.push_back(coords.size());
	}

	void setCSRValues(VectorXd &values) {
		for(int i = 0; i < blockStarts.size() - 1; i++) {
			if(csr_blocks[i] != nullptr) {
				memcpy(csr_blocks[i]->getActive()->values.data(), 	
						values.data() + blockStarts[i], 
						sizeof(double) * (blockStarts[i + 1] - blockStarts[i]));
			}
		}
	}

	VectorXd getCSRValues() {
		VectorXd values = VectorXd::Constant(blockStarts[blockStarts.size() -1], 0.0);

		for(int i = 0; i < blockStarts.size() - 1; i++) {
			if(csr_blocks[i] != nullptr) {
				memcpy(values.data() + blockStarts[i], 
						csr_blocks[i]->getActive()->values.data(), 
						sizeof(double) * (blockStarts[i + 1] - blockStarts[i]));
			}
		}

		return values;
	}

	void setValuesConstant(double cval) {
		for(int i = 0; i < blockStarts.size() - 1; i++) {
			if(csr_blocks[i] != nullptr) {
				// This may be too slow, we maybe should optimize this...
				#pragma omp parallel for
				for(int j = 0; j < blockStarts[i+1] - blockStarts[i]; j++) {
					csr_blocks[i]->getActive()->values[j] = cval;
				}
			}
		}
	}
};

