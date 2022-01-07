#include <petscmat.h>
#include <algorithm>
#include <fstream>
#include "omp.h"
#include "../json.hpp"

using json = nlohmann::json;
using namespace std;

static char help[] = "help yourself!";

int main (int argc, char **argv)
{
	Mat A, B, C;
	PetscViewer fd;
	PetscInt m, n;
	MatInfo info;
	double nnz;
	int niters;

	PetscInitialize(&argc, &argv, (char*)0, help);

	int nthds, np;
	MPI_Comm_size(MPI_COMM_WORLD, &np);
	int rank;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	#pragma omp parallel
	{
		nthds = omp_get_num_threads();
	}

	std::string output_file(argv[4]);

	niters = atoi(argv[2]);
	//PetscPrintf(PETSC_COMM_WORLD, "np %d nthds %d\n", np, nthds);

	double read_mat_beg = MPI_Wtime();
	
	//PetscPrintf(PETSC_COMM_WORLD, "reading matrix %s (A)\n", argv[1]);
	PetscViewerBinaryOpen(PETSC_COMM_WORLD, argv[1] ,FILE_MODE_READ, &fd);
	MatCreate(PETSC_COMM_WORLD, &A);
	MatSetType(A, MATMPIAIJ);
	MatLoad(A, fd);
	PetscViewerDestroy(&fd);

	MatGetSize(A, &m, &n);
	MatGetInfo(A, MAT_GLOBAL_SUM, &info);

	long long int sparse_nnz = (long long int) info.nz_used;
	//PetscPrintf(PETSC_COMM_WORLD, "A matrix size %d %d %lld\n",
	//			m, n, sparse_nnz);

	/*
	PetscInt a_local_m, a_local_n;
	MatGetLocalSize(A, &a_local_m, &a_local_n);

	PetscPrintf(PETSC_COMM_WORLD, "A matrix local size %d %d %lld\n",
				a_local_m, a_local_n, (long long int)info.nz_used);
	*/

	/*PetscPrintf(PETSC_COMM_WORLD, "reading matrix %s (B)\n", argv[2]);
	PetscViewerBinaryOpen(PETSC_COMM_WORLD, argv[2] ,FILE_MODE_READ, &fd);
	MatCreate(PETSC_COMM_WORLD, &B);
	MatSetType(B, MATMPIAIJ);
	MatLoad(B, fd);
	PetscViewerDestroy(&fd);*/

	PetscInt r = atoi(argv[3]);

	/*MatCreate(PETSC_COMM_WORLD, &B);
	MatSetType(B, MATMPIDENSE);
	MatSetSizes(B, PETSC_DECIDE, PETSC_DECIDE, n, r);*/
	MatCreateDense(PETSC_COMM_WORLD, 
			PETSC_DECIDE, 
			PETSC_DECIDE, 
			n, 
			r,
			NULL,
			&B);

	MatGetSize(B, &m, &n);
	MatGetInfo(B, MAT_GLOBAL_SUM, &info);
	//PetscPrintf(PETSC_COMM_WORLD, "B matrix size %d %d %lld\n",
	//			m, n, (long long int)info.nz_used);

	double read_mat_end = MPI_Wtime();


	PetscInt b_local_m, b_local_n;
	MatGetLocalSize(B, &b_local_m, &b_local_n);
	/*PetscPrintf(PETSC_COMM_WORLD, "B matrix local size %d %d %lld\n",
				b_local_m, b_local_n, (long long int)info.nz_used);*/


	PetscInt arrLength = b_local_m * r; 
	PetscScalar* arr;
	MatDenseGetArray(B, &arr);

	for(int i = 0; i < arrLength; i++) {
		arr[i] = 1.0;
	}

	MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);
	
	//PetscPrintf(PETSC_COMM_WORLD, "Performing SpMM\n");

	// Warmup to initialize matrix
	MatMatMult(A, B, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &C);

	int i;
	double start_time = MPI_Wtime();
	for (i = 0; i < niters; ++i)
	{
		MatMatMult(A, B, MAT_REUSE_MATRIX, PETSC_DEFAULT, &C);
	}
	double end_time = MPI_Wtime();

	/*
	PetscPrintf(PETSC_COMM_WORLD, "IO %lf spmm %lf\n",
				read_mat_end-read_mat_beg,
				(end_time-start_time));
	*/

	MatGetSize(C, &m, &n);
	MatGetInfo(C, MAT_GLOBAL_SUM, &info);
	/*	
	PetscPrintf(PETSC_COMM_WORLD, "C matrix size %d %d %lld\n",
				m, n, (long long int)info.nz_used);
	*/

	/* PetscPrintf(PETSC_COMM_WORLD, "Writing the output matrix\n"); */
	/* PetscViewerBinaryOpen(PETSC_COMM_WORLD, argv[3], FILE_MODE_WRITE, &fd); */
	/* MatView(C, fd); */

	json j_obj, algorithm_info;
	double elapsed = end_time - start_time;

	j_obj["elapsed"] = elapsed; 
	j_obj["overall_throughput"] = (long long int) 2 * r * sparse_nnz * niters 
			/ (elapsed * 1e9); 
	j_obj["num_trials"] = niters / 2;
	j_obj["fused"] = false;
	j_obj["alg_name"] = "petsc";

	json alg_info = {
		{"alg_name", "petsc"},
		{"m", m},
		{"m", n},
		{"nnz", sparse_nnz},
		{"r", r},
		{"p", np},
		{"c", 1}};

    j_obj["alg_info"] = alg_info;
	string res =  j_obj.dump() + ",\n";

	if(rank == 0) {
		ofstream fout;
			fout.open(output_file, std::ios_base::app 
		);

		fout << res;
		fout.close();
	}

	MatDestroy(&A);
	MatDestroy(&B);
	MatDestroy(&C);
	PetscFinalize();

	return 0;	
}