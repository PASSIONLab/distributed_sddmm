#pragma once

#include <iostream>
#include <mpi.h>
#include <vector>

using namespace std;

/*
 * This is a more fully featured 3D grid than the one that CombBLAS offers.
 */
class FlexibleGrid {
public:
	int i, j, k; 
	int adjacency;

	int global_rank, num_procs;

	int dim_list[3];	

	// Convenient copy of dim_array
	int nr, nc, nh;

	int permutation[3];

	MPI_Comm row_world, col_world, fiber_world;
	MPI_Comm rowcol_slice, rowfiber_slice, colfiber_slice;
	int rankInRow, rankInCol, rankInFiber;

	/*
	 * Adjacency is a parameter from 1 to 6 that specifies the ordering
	 * of MPI ranks. Ordering is from most adjacent to least adjacent. 
	 *
	 * 1. crf
	 * 2. cfr
	 * 3. rcf  *** ADJACENCY 3 IS USUALLY THE BEST CHOICE ***
	 * 4. rfc
	 * 5. fcr
	 * 6. frc
	 */
	FlexibleGrid(int nr, int nc, int nh, int adjacency) {
		MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
		MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
		assert(nr * nc * nh == num_procs);

		dim_list[0] = nr;
		dim_list[1] = nc;
		dim_list[2] = nh;
		this->nr = nr;
		this->nc = nc;
		this->nh = nh;
		this->adjacency = adjacency;

		switch (adjacency) {
			case 1:
				permutation[0] = 0; permutation[1] = 1; permutation[2] = 2;
				break;
			case 2:
				permutation[0] = 0; permutation[1] = 2; permutation[2] = 1;
				break;
			case 3:
				permutation[0] = 1; permutation[1] = 0; permutation[2] = 2;
				break;
			case 4:
				permutation[0] = 1; permutation[1] = 2; permutation[2] = 0;
				break;
			case 5:
				permutation[0] = 2; permutation[1] = 0; permutation[2] = 1;
				break;
			case 6:
				permutation[0] = 2; permutation[1] = 1; permutation[2] = 0;
				break;
		}

		get_ijk_indices(&i, &j, &k);
		assert(global_rank == get_global_rank(i, j, k));

		// Create subcommunicators for row, column, fiber worlds; we can 
		// ignore the permutation here since we're just chunking up the grid. 
		MPI_Comm_split(MPI_COMM_WORLD, i + k * nr, j, &row_world);
		MPI_Comm_split(MPI_COMM_WORLD, j + k * nc, i, &col_world);
		MPI_Comm_split(MPI_COMM_WORLD, i + j * nr, k, &fiber_world);

		// Create subcommunicators for all slices. TODO: We should really use
		// the permutation to order the processes within a slice. 
		MPI_Comm_split(MPI_COMM_WORLD, k, i + j * nr, &rowcol_slice);
		MPI_Comm_split(MPI_COMM_WORLD, j, i + k * nr, &rowfiber_slice);
		MPI_Comm_split(MPI_COMM_WORLD, i, j + k * nc, &colfiber_slice);

		MPI_Comm_rank(row_world, &rankInRow);
		MPI_Comm_rank(col_world, &rankInCol);
		MPI_Comm_rank(fiber_world, &rankInFiber);

	}

	~FlexibleGrid() {
		MPI_Comm_free(&row_world);
		MPI_Comm_free(&col_world);
		MPI_Comm_free(&fiber_world);
		MPI_Comm_free(&rowcol_slice);
		MPI_Comm_free(&rowfiber_slice);
		MPI_Comm_free(&colfiber_slice);
	}

	void get_ijk_indices(int global_rank, int* i, int* j, int* k) {
		int ijk_temp[3];
		ijk_temp[permutation[0]] = global_rank % dim_list[permutation[0]];
		ijk_temp[permutation[1]] = (global_rank / dim_list[permutation[0]])
			% dim_list[permutation[1]];
		ijk_temp[permutation[2]] = (global_rank / (dim_list[permutation[0]] * dim_list[permutation[1]]))
			% dim_list[permutation[2]];

		*i = ijk_temp[0];
		*j = ijk_temp[1];
		*k = ijk_temp[2];
	}

	void get_ijk_indices(int* i, int* j, int* k) {
		get_ijk_indices(global_rank, i, j, k);
	}

	int get_global_rank(int i, int j, int k) {
		int global_rank = 0;

		int ijk_temp[3];
		ijk_temp[0] = i;
		ijk_temp[1] = j;
		ijk_temp[2] = k;

		global_rank += ijk_temp[permutation[0]];
		global_rank += ijk_temp[permutation[1]] * dim_list[permutation[0]];
		global_rank += ijk_temp[permutation[2]] * dim_list[permutation[0]] * dim_list[permutation[1]]; 

		return global_rank;
	}

	void print_rank_information() {
		cout << "Global Rank: " << global_rank <<
		"i, j, k: (" << i << ", " << j << ", " << k << ")" << endl;
	}

	template<typename T>
	void prettyPrint(vector<T> &input) {
		for(int k = 0; k < nh; k++) {
			cout << "========= Layer " << k << " ==========" << endl;

			for(int i = 0; i < nr; i++) {
				for(int j = 0; j < nc; j++) {
					cout << input[get_global_rank(i, j, k)] << "\t";
				}
				cout << endl;
			}

			cout << "============================" << endl; 

		}
	}

	void gather_and_pretty_print(string title, int msg) {
		vector<int> buff(num_procs, 0.0);
		MPI_Gather(&msg, 1, MPI_INT, buff.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

		if(global_rank == 0) {
			cout << title << endl;
			prettyPrint(buff);
		}
	}

	void self_test() {
		gather_and_pretty_print("Global Ranks:", global_rank);
		gather_and_pretty_print("i Values:", i);
		gather_and_pretty_print("j Values:", j);
		gather_and_pretty_print("k Values:", k);

		int buf;
		buf = i;
		MPI_Bcast(&buf, 1, MPI_INT, 0, row_world);
		gather_and_pretty_print("Row Broadcast:", buf);

		buf = j;
		MPI_Bcast(&buf, 1, MPI_INT, 0, col_world);
		gather_and_pretty_print("Col Broadcast:", buf);


		buf = i + nr * j;
		MPI_Bcast(&buf, 1, MPI_INT, 0, fiber_world);
		gather_and_pretty_print("Fiber Broadcast:", buf);

		buf = k;
		MPI_Bcast(&buf, 1, MPI_INT, 0, rowcol_slice);
		gather_and_pretty_print("Row Column Slice Broadcast:", buf);

		buf = i;
		MPI_Bcast(&buf, 1, MPI_INT, 0, colfiber_slice);
		gather_and_pretty_print("Column Fiber Slice Broadcast:", buf);


		buf = j;
		MPI_Bcast(&buf, 1, MPI_INT, 0, rowfiber_slice);
		gather_and_pretty_print("Row Fiber Slice Broadcast:", buf);
	}
};
