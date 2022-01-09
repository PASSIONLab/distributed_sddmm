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


string spcoord_t::string_rep() {
    return std::to_string(r) + " " + std::to_string(c) + " " + std::to_string(value);
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

bool column_major(spcoord_t a, spcoord_t b) {
    if(a.c == b.c) {
        return a.r < b.r;
    }
    else {
        return a.c < b.c;
    }
}


bool row_major(spcoord_t &a, spcoord_t &b) {
    if(a.r == b.r) {
        return a.c < b.c;
    }
    else {
        return a.r < b.r;
    }
}

void divideIntoSegments(int total, int num_segments, 
        vector<int> &segment_starts, 
        vector<int> &segment_sizes) {

    int share_size = divideAndRoundUp(total, num_segments);
    segment_starts.clear();

    for(int i = 0; i < num_segments; i++) {
        segment_starts.push_back(std::min(share_size * i, total));
    }
    segment_starts.push_back(total);

    for(int i = 0; i < num_segments; i++) {
        segment_sizes.push_back(segment_starts[i+1] - segment_starts[i]);
    }

}