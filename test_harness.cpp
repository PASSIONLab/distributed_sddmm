#include "15D_sparse.hpp"
#include "25D_sparse.hpp"
#include "sparse_kernels.h"
#include "common.h"
#include "als_conjugate_gradients.h"

using namespace std;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    string fname(argv[1]);

    StandardKernel local_ops;
    Sparse15D* d_ops = new Sparse15D(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), &local_ops);
    d_ops->reset_performance_timers();
    //Sparse15D* d_ops = new Sparse15D(fname, atoi(argv[2]), atoi(argv[3]), &local_ops);

    //d_ops->setVerbose(true);

    Distributed_ALS* x = new Distributed_ALS(d_ops, d_ops->grid->GetLayerWorld(), true);
    x->run_cg(5);
    d_ops->print_performance_statistics();


    MPI_Finalize();
}