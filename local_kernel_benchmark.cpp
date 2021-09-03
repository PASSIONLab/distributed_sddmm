#include <iostream>
#include <mkl_spblas.h>
#include <Eigen/Dense>
#include "SpmatLocal.hpp"
#include <mpi.h>

using namespace std;
using namespace Eigen;

typedef Matrix<double, Dynamic, Dynamic, RowMajor> DenseMatrix;

int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);	
	cout << "Starting MKL SpGEMM Benchmark!" << endl;

	// Plug into the Graph500 generator to generate sparse matrices
	// with varying counts of nonzeros

	SpmatLocal erdos_renyi;
	erdos_renyi.loadTuples(false, 5, 3, "");

	sparse_matrix_t A;

	int m = 5;
	int n = m;
	int nnz = 5;
	int R = 25;

	DenseMatrix B = DenseMatrix::Constant(n, R, 1.0);
	DenseMatrix C = DenseMatrix::Constant(m, R, 1.0);

	MKL_INT row_start[] = {0, 1, 2, 3, 4, 5};
	MKL_INT col_index[] = {0, 1, 2, 3, 4};
	double values[] = {0.0, 1.0, 2.0, 3.0, 4.0};

	struct matrix_descr descr;
	descr.type = SPARSE_MATRIX_TYPE_GENERAL;



	mkl_sparse_d_create_csr(
			&A, 
			SPARSE_INDEX_BASE_ZERO,
			m,
			n,
			row_start,
			row_start + 1,
			col_index,
			values
	);

	mkl_sparse_d_mm (
			SPARSE_OPERATION_NON_TRANSPOSE,	
			1.0, 
			A, 
			descr,	
			SPARSE_LAYOUT_ROW_MAJOR,	
			B.data(), 
			R, 
			R,  // ldb
			1.0, 
			C.data(), 
			R); // ldc

	cout << "Benchmark complete!" << endl;

	MPI_Finalize();
}