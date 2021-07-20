#include <chrono>
#include "common.h"

using namespace std;

my_timer_t start_clock() {
    return std::chrono::steady_clock::now();
}

double stop_clock_get_elapsed(my_timer_t &start) {
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;
    return diff.count();
}

int pMod(int num, int denom) {
    return ((num % denom) + denom) % denom;
}

int divideAndRoundUp(int num, int denom) {
    if (num % denom > 0) {
        return num / denom + 1;
    }
    else {
        return num / denom;
    }
} 

MPI_Datatype SPCOORD;

void initialize_mpi_datatypes() {
    const int nitems = 3;
    int blocklengths[3] = {1, 1, 1};
    MPI_Datatype types[3] = {MPI_UINT64_T, MPI_UINT64_T, MPI_DOUBLE}; 
    MPI_Aint offsets[3];
    offsets[0] = offsetof(spcoord_t, r);
    offsets[1] = offsetof(spcoord_t, c);
    offsets[2] = offsetof(spcoord_t, value);
    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &SPCOORD);
    MPI_Type_commit(&SPCOORD);
}	

