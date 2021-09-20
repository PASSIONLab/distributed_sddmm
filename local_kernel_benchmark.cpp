#include <iostream>
#include <mkl_spblas.h>
#include <Eigen/Dense>
#include <mpi.h>
#include <omp.h>

#include "SpmatLocal.hpp"
#include "common.h"

using namespace std;
using namespace Eigen;

void generateRowStart(MKL_INT* rows, MKL_INT* rowStart, int numCoords, int numRows) {
	#pragma omp parallel for
	for(int i = 0; i < numRows; i++) {
		rowStart[i] = 0;
	}

	rowStart[0] = 0;
	rowStart[numRows] = numCoords;

	/*#pragma omp parallel for
	for(int i = 1; i < numCoords; i++) {
		if(rows[i] != rows[i - 1]) {
			rowStart[rows[i]] = i;
		}
	}

	#pragma omp parallel for
	for(int i = 0; i < numRows; i++) {
		int idx = i + 1;
		if(rowStart[i] != -1) {
			while(rowStart[idx] == -1) {
				rowStart[idx] = rowStart[i];
				idx++;
			}
		}
	}*/
}

void sddmm(double* ptrB, double* ptrC, double* values, MKL_INT* rows, MKL_INT* cols, int num_coords, int R) {
	#pragma omp parallel for
	for(int t = 0; t < num_coords; t++) {
		double* Brow = ptrB + rows[t];
		double* Crow = ptrC + cols[t];
		double value = 0.0;
		#pragma ivdep
		for(int k = 0; k < R; k++) {
			value += Brow[k] * Crow[k];	
		}
		values[t] = value;
	}
}

void spmm(double* ptrB, double* ptrC, sparse_matrix_t &A, int num_coords, int R) {
	struct matrix_descr descr;
	descr.type = SPARSE_MATRIX_TYPE_GENERAL;

	mkl_sparse_d_mm (
			SPARSE_OPERATION_NON_TRANSPOSE,	
			1.0, 
			A, 
			descr,	
			SPARSE_LAYOUT_ROW_MAJOR,	
			ptrB, 
			R, 
			R,  // ldb
			1.0, 
			ptrC, 
			R); // ldc	
}

void benchmark(int logM, int nnz_per_row, vector<int> &rValues, double min_time, bool benchmark_sddmm) {
	SpmatLocal erdos_renyi;
	erdos_renyi.loadTuples(false, logM, nnz_per_row, "");	
	std::sort(erdos_renyi.coords.begin(), erdos_renyi.coords.end(), sortbycolumns);

	int num_coords = erdos_renyi.coords.size();

	MKL_INT* rows = new MKL_INT[num_coords];
	MKL_INT* cols = new MKL_INT[num_coords];
	double* values  = new double[num_coords];

	for(int i = 0; i < num_coords; i++) {
		rows[i] = erdos_renyi.coords[i].r;
		cols[i] = erdos_renyi.coords[i].c;
		values[i] = 1.0; 
	}

	sparse_matrix_t A_coo, A, A_manual_convert;
	mkl_sparse_d_create_coo (
			&A_coo, 
			SPARSE_INDEX_BASE_ZERO,	
			erdos_renyi.M, 
			erdos_renyi.N, 
			erdos_renyi.coords.size(), 
			rows, 
			cols, 
			values);

	//A = A_coo;
	mkl_sparse_convert_csr(A_coo, SPARSE_OPERATION_NON_TRANSPOSE, &A);

	MKL_INT* rowStart = new MKL_INT[erdos_renyi.M + 1];

	for(int i = 0; i < rValues.size(); i++) {
		int R = rValues[i];
		DenseMatrix B = DenseMatrix::Constant(erdos_renyi.N, R, 1.0);
		DenseMatrix C = DenseMatrix::Constant(erdos_renyi.M, R, 1.0);

		double* ptrB = B.data();
		double* ptrC = C.data();

		my_timer_t t = start_clock();
		int num_trials = 0;

		//#pragma omp parallel 
		//{
			//#pragma omp single 
			//{
				do {
					num_trials++;	

					/*mkl_sparse_d_create_csr(
							&A_manual_convert,
							SPARSE_INDEX_BASE_ZERO,
							erdos_renyi.M,
							erdos_renyi.N,
							rowStart,
							rowStart+1,
							cols,
							values
					);*/

					if(benchmark_sddmm) {
						sddmm(ptrB, ptrC, values, rows, cols, num_coords, R);
					}
					else {
						spmm(ptrB, ptrC, A, num_coords, R);
					}


				} while(stop_clock_get_elapsed(t) < min_time);
			//}
		//}

		double elapsed = stop_clock_get_elapsed(t);
		double throughput = erdos_renyi.coords.size() * 2 * R * num_trials / elapsed;
		throughput /= 1.0e9;

		cout << erdos_renyi.M << "\t" 
				<< erdos_renyi.N << "\t" 
				<< erdos_renyi.coords.size() << "\t"
				<< R << "\t"
				<< throughput << "\t"
				<< num_trials	
				<< endl;
	}
}

void print_header() {
	cout << "M\tN\tNNZ\tR\tGFLOPs\tTrials" << endl;
	cout << "==========================================" << endl;
}

void print_footer() {
	cout << "==========================================" << endl;
}

int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);	

	vector<int> logMValues {13, 14, 15, 16};
	vector<int> nnz_per_row_values {8, 16, 32, 64, 128};
	vector<int> rValues {8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};

	double min_time = 5.0;

	print_header();
	for(int i = 0; i < logMValues.size(); i++) {
		for(int j = 0; j < nnz_per_row_values.size(); j++) {
			cout << "SDDMM Benchmark" << endl;
			benchmark(logMValues[i], 
					nnz_per_row_values[j], 
					rValues,
					min_time,
					true
					);

			cout << "SpMM Benchmark" << endl;
			benchmark(logMValues[i], 
					nnz_per_row_values[j], 
					rValues,
					min_time,
					false	
					);
		}
	}
	print_footer();

	MPI_Finalize();
}

