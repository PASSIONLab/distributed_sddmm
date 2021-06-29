#include "15D_mdense_bcast.hpp"
#include "15D_mdense_shift.hpp"
#include "15D_mdense_shift_striped.hpp"
#include "25D_mdense_nostage.hpp"

#include "sparse_kernels.h"
#include "common.h"
#include "als_conjugate_gradients.h"

using namespace std;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    string fname(argv[1]);

    StandardKernel local_ops;
    //FusedStandardKernel fused_local_ops;

    Sparse15D_MDense_Shift_Striped* d_ops = new Sparse15D_MDense_Shift_Striped(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), &local_ops);
    // Sparse15D_MDense_Shift* d_ops = new Sparse15D_MDense_Shift(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), &local_ops);

    //Sparse25D_MDense_Nostage* d_ops = new Sparse25D_MDense_Nostage(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), &local_ops);

    //Sparse25D* d_ops = new Sparse25D(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), &local_ops);

    srand((unsigned int) time(0) + d_ops->proc_rank);

    Distributed_ALS* x = new Distributed_ALS(d_ops, d_ops->grid->GetLayerWorld(), true, false);

    //Distributed_ALS* x = new Distributed_ALS(d_ops, d_ops->grid->GetLayerWorld(), true, false);

    d_ops->reset_performance_timers();
    x->run_cg(2);
    d_ops->print_performance_statistics();

    MPI_Finalize();
}