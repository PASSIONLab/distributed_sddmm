#include "benchmark_dist.hpp"
#include <string>

using namespace std;

/*
 * The main file is currently set up to benchmark the
 * algorithms on Erdos-Reyi random matrices. The parameters
 * are, in order,
 *
 * logM
 * edgeFactor
 * algorithm name
 * R-value
 * replication factor
 * fused?
 */

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    initialize_mpi_datatypes();

    string fname(argv[1]);
    string algorithm_name(argv[2]);
    int R = atoi(argv[3]);
    int c = atoi(argv[4]);
    string output_file(argv[5]);
    string app(argv[6]);

    SpmatLocal S;
    S.loadTuples(true, -1, -1, fname);

    if(algorithm_name == "15d") {
        /*benchmark_algorithm(&S, 
                "15d_sparse",
                output_file,
                true,
                R,
                c,
                app);*/

        benchmark_algorithm(&S, 
                "15d_sparse",
                output_file,
                false,
                R,
                c,
                app);

        /*benchmark_algorithm(&S, 
                "15d_fusion2",
                output_file,
                true,
                R,
                c,
                app);*/

        /*benchmark_algorithm(&S, 
                "15d_fusion1",
                output_file,
                true,
                R,
                c,
                app);*/

        /*benchmark_algorithm(&S, 
                "15d_fusion1",
                output_file,
                false,
                R,
                c,
                app);*/
    }
    else if(algorithm_name == "25d") {
        /*benchmark_algorithm(&S, 
                "25d_sparse_replicate",
                output_file,
                false,
                R,
                c,
                app);*/

        /*benchmark_algorithm(&S, 
                "25d_dense_replicate",
                output_file,
                true,
                R,
                c,
                app);*/

        benchmark_algorithm(&S, 
                "25d_dense_replicate",
                output_file,
                false,
                R,
                c,
                app);
    }
    else {
        assert(false);
    }

    MPI_Finalize();
}