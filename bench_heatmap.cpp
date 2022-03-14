#include "benchmark_dist.hpp"
#include <string>

using namespace std;

/*
 * The main file is currently set up to benchmark the
 * algorithms on Erdos-Reyi random matrices. The parameters
 * are, in order,
 *
 * edgeFactor
 * algorithm name
 * replication factor
 * fused?
 */

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    initialize_mpi_datatypes();

    int logM = atoi(argv[1]);
    int edgeFactor = atoi(argv[2]);
    string algorithm_name(argv[3]);
    int c = atoi(argv[4]);
    string output_file(argv[5]);

    string app = "vanilla";
    string dummy = "";

    SpmatLocal S;
    S.loadTuples(false, logM, edgeFactor, dummy);

    int rvalues[] = {64, 128, 192, 256, 320, 384, 448};

    for(int i = 0; i < 7; i++) {
        int R = rvalues[i];

        if(algorithm_name == "15d") {
            benchmark_algorithm(&S, 
                    "15d_fusion1",
                    output_file,
                    true,
                    R,
                    c,
                    app);

            benchmark_algorithm(&S, 
                    "15d_fusion2",
                    output_file,
                    true,
                    R,
                    c,
                    app);

            /*benchmark_algorithm(&S, 
                    "15d_fusion1",
                    output_file,
                    false,
                    R,
                    c,
                    app);*/

            benchmark_algorithm(&S, 
                    "15d_sparse",
                    output_file,
                    true,
                    R,
                    c,
                    app);

            /*benchmark_algorithm(&S, 
                    "15d_sparse",
                    output_file,
                    false,
                    R,
                    c,
                    app);*/
        }
        else if(algorithm_name == "25d") {
            benchmark_algorithm(&S, 
                    "25d_sparse_replicate",
                    output_file,
                    false,
                    R,
                    c,
                    app);

            benchmark_algorithm(&S, 
                    "25d_dense_replicate",
                    output_file,
                    true,
                    R,
                    c,
                    app);

            /*benchmark_algorithm(&S, 
                    "25d_dense_replicate",
                    output_file,
                    false,
                    R,
                    c,
                    app);*/
        }
        else {
            assert(false);
        }
    } 

    MPI_Finalize();
}