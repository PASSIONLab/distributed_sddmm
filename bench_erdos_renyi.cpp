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

    int logM = atoi(argv[1]);
    int edgeFactor = atoi(argv[2]);
    string algorithm_name(argv[3]);
    int R = atoi(argv[4]);
    int c = atoi(argv[5]);
    string fused_string(argv[6]);
    string output_file(argv[7]);

    bool fused;
    if(fused_string == "fused") {
        fused = true;
    }
    else if(fused_string == "unfused") {
        fused = false;
    }
    else {
        assert(false);
    }

    string dummy = "";

    SpmatLocal S;
    S.loadTuples(false, logM, edgeFactor, dummy);
    benchmark_algorithm(&S, 
            algorithm_name,
            output_file,
            fused,
            R,
            c);

    MPI_Finalize();
}