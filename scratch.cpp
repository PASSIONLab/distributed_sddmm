class ZeroProcess : public NonzeroDistribution {
public:
    ZeroProcess(int M, int N) { 
        rows_in_block = M; 
        cols_in_block = N;
    }

	int blockOwner(int row_block, int col_block) {
        return 0; 
    }
};


void verify_operation(SpmatLocal &spmat, Distributed_Sparse* d_ops) {
    int proc_rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    DenseMatrix A = d_ops->like_A_matrix(0.0);    
    DenseMatrix B = d_ops->like_B_matrix(0.0);

    VectorXd S = d_ops->like_S_values(1.0); 
    VectorXd ST = d_ops->like_ST_values(1.0);
    VectorXd result = d_ops->like_S_values(0.0);

    d_ops->dummyInitialize(A, Amat);
    d_ops->dummyInitialize(B, Bmat);
    d_ops->initial_synchronize(&A, nullptr, nullptr);

    //d_ops->print_nonzero_distribution(A, B);
    //d_ops->sddmmA(A, B, S, result);

    double A_fingerprint = A.squaredNorm();
    MPI_Allreduce(MPI_IN_PLACE, &A_fingerprint, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double sddmm_fingerprint = result.squaredNorm(); 
    MPI_Allreduce(MPI_IN_PLACE, &sddmm_fingerprint, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    d_ops->dummyInitialize(A, Amat);
    d_ops->dummyInitialize(B, Bmat);
    d_ops->initial_synchronize(&A, nullptr, nullptr);
    //d_ops->spmmA(A, B, S);

    double spmmA_fingerprint = A.squaredNorm();
    MPI_Allreduce(MPI_IN_PLACE, &spmmA_fingerprint, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    d_ops->dummyInitialize(A, Amat);
    d_ops->dummyInitialize(B, Bmat);

    d_ops->initial_synchronize(nullptr, &B, nullptr);
    //d_ops->spmmB(A, B, ST);

    double spmmB_fingerprint = B.squaredNorm(); 
    MPI_Allreduce(MPI_IN_PLACE, &spmmB_fingerprint, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    DenseMatrix fused_result = d_ops->like_A_matrix(0.0);

    //d_ops->print_nonzero_distribution(A, B); 

    if(proc_rank == 0) {
        cout << "A Fingerprint: " << A_fingerprint << endl;
        cout << "SDDMM Fingerprint: " << sddmm_fingerprint << endl;
        cout << "SpMMA Fingerprint: " << spmmA_fingerprint << endl;
        cout << "SpMMB Fingerprint: " << spmmB_fingerprint << endl; 
    } 
}