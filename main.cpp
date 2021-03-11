#include <iostream>
#include <cstdio>
#include "mm.h"
#include <vector>
#include <utility>
#include <algorithm>
#include <bcl/bcl.hpp>
#include <bcl/containers/DMatrix.hpp>
#include <immintrin.h>
#include <string.h>
#include <chrono>
#include <ctime>
#include "hypergraph_partition.h"
#include <string>
#include <fstream>
#include <unordered_set>

using namespace std;

inline bool file_exists (const std::string& name) {
    ifstream f(name.c_str());
    return f.good();
}

// Sort by rows...
struct coord_sort_key 
{
    inline bool operator() (const pair<int, int>& p1, 
                            const pair<int, int>& p2)
    {
        // If rows are equal, sort by column
        if(p1.first == p2.first) {
            return p1.second < p2.second;
        }
        else {
            return p1.first < p2.first;
        }
    }
};

inline double vectorized_dot_product(double* A, double* B, size_t r) {
        __m512d lane1;

        #pragma GCC unroll 20
        for(int j = 0; j < r; j+=8) {
            __m512d Avec1 = _mm512_loadu_pd(A + j);
            __m512d Bvec1 = _mm512_loadu_pd(B + j);

            lane1 = _mm512_fmadd_pd(Avec1, Bvec1, lane1);
        }
        return (_mm512_reduce_add_pd(lane1));
}

std::vector<double> fetch_row(BCL::DMatrix<double> &A, size_t abs_row) {
    size_t tile_num =  abs_row / A.tile_shape()[0];
    size_t row_num  = abs_row % A.tile_shape()[0];
    // printf("Getting row %lu, which should be row %lu within tile %lu:\n",
    //        i, row_num, tile_num);
    return(A.get_tile_row(tile_num, 0, row_num));
}

void serial_kernel(vector<pair<size_t, size_t>> &coordinates, 
    double* A,
    double* B,
    size_t r,
    double* result) {   

    // We assume that the local coordinates are sorted by row so we can
    // just call this kernel repeatedly

    for(int i = 0; i < coordinates.size(); i++) {
        double* Arow = A + r * coordinates[i].first;
        double* Brow = B + r * coordinates[i].second; 
        result[i] = vectorized_dot_product(Arow, Brow, r); 
    }
}

void SDDMM_column_dist( vector<pair<size_t, size_t>> &coordinates, 
            BCL::DMatrix<double> &A,
            double* B_local, 
            double* result 
            ) {
    
    // We assume that the local coordinates are sorted by row so we can
    // just call this kernel repeatedly

    int currentRow = -1;

    // vector<double> Atile = A.get_tile(BCL::rank(), 0);
    vector<double> Btile = B.get_tile(BCL::rank(), 0);
    vector<double> Arow;

    for(int i = 0; i < coordinates.size(); i++) {
        if ((int) coordinates[i].first != currentRow) {
            if((int) coordinates[i].first < currentRow) {
                cout << "Not ordered correctly " << coordinates[i].first << endl;
            }
            Arow = fetch_row(A, coordinates[i].first); // Should do asynchronously 
            if(coordinates[i].second / B.tile_shape()[0] != BCL::rank()) {
                cout << "Error, nonzero in the wrong bucket: " << coordinates[i].second << endl;
            }
            currentRow = coordinates[i].first;
        }

        // double* Arow = Atile.data() + A.shape()[1] * (coordinates[i].first % A.tile_shape()[0]);
        double* Brow = B_local + A.shape()[1] * coordinates[i].second;
        result[i] = vectorized_dot_product(Arow.data(), Brow, A.shape()[1]);
    }
}

int main(int argc, char** argv) {
    if(argc < 3) {
        cout << "Usage: Provide filename of matrix market format, r value." << endl;
        return 1; 
    }

    struct mmdata spMat; 

    if (initialize_mm(argv[1], &spMat)) {
        cout << "Error reading supplied matrix." << endl;
        return 1;
    } 

    cout    << "Reading Sparse Matrix with " << spMat.M << " rows,"
            << spMat.N << " columns, and " << spMat.NNZ << " nonzeros." << endl;

    size_t r = (size_t) atoi(argv[2]);

    // Start BCL once we've read the sparse matrix.
    size_t segment_size = 256; // Segment size in megabytes
	BCL::init(segment_size);
    // In case symmetric, materialize every entry for convenience

    vector<pair<size_t, size_t>> coordinates;

    for(int i = 0; i < spMat.NNZ; i++) {
        coordinates.emplace_back((size_t) spMat.x[i], (size_t) spMat.y[i]);
        if(spMat.symmetricity == 1) {
            if(spMat.x[i] != spMat.y[i]) {            
                    coordinates.emplace_back((size_t) spMat.y[i], (size_t) spMat.x[i]); 
            }
        }
    }

        // This fully materializes the sparse matrix (since otherwise, we only get the
        // entries above the diagonal if symmetric)

    sort(coordinates.begin(), coordinates.end(), coord_sort_key());

    string partition_filepath =  string(argv[1]) 
                       + "." 
                       + to_string(BCL::nprocs()) 
                       + ".partitioning";

    if(BCL::rank() == 0) {
        if(! file_exists(partition_filepath)) {
            partition(coordinates, spMat.M, spMat.N, BCL::nprocs(), partition_filepath);
        }
    }

    // Now filter down to only the coordinates that we care about
    unordered_set<int> my_column_embeds; 
    vector<pair<size_t, size_t>> local_coordinates;
    ifstream f(partition_filepath);
    int maxLen;
    double partition_time;
    f >> maxLen >> partition_time;

    for(int i = 0; i < spMat.N; i++) {
        size_t col_idx, processor, local_idx;
        f >> coord_idx >> processor >> local_idx;

        if (processor == BCL::rank()) {
            my_column_embeds.insert(coord_idx); 
        }
    }

    // Initialize the distributed and local dense matrices
	BCL::DMatrix<double> A({(size_t) spMat.M, BCL::BlockRow());
    srand48(BCL::rank());
    A.apply_inplace([](double a) { return drand48(); });
    double* B_local = new double[maxLen * r];
    for(int i = 0; i < maxLen * r; i++) {
        B_local[i] = drand48(); 
    }

    if (BCL::rank() == 0) {
        cout << "Matrix A: " << endl;
        A.print_info();
    }

    BCL::barrier();

    /*double* result = new double[coordinates.size()];

    int num_trials = 30;

    if(strcmp(argv[3], "serial") == 0) {
        vector<double> Atile = A.get_tile(BCL::rank(), 0);
        vector<double> Btile = B.get_tile(BCL::rank(), 0);

        cout << "Starting serial kernel..." << endl;
        auto t_start = std::chrono::steady_clock::now();
        for(int i = 0; i < num_trials; i++) {
            serial_kernel(coordinates, Atile.data(), Btile.data(), r, result);
        }
        auto t_end = std::chrono::steady_clock::now();
        double millis_per_kernel = std::chrono::duration<double, std::milli>(t_end-t_start).count() / num_trials;

        cout << "Kernel took " << millis_per_kernel << " milliseconds" << endl;
        cout    << "Throughput is " 
                << total_NNZ * r * 2.0 / (millis_per_kernel / 1000) * 1e-9 
                << " GFLOPs" << endl;
    }
    else {

        if(BCL::rank() == 0) {
            cout << "Starting parallel kernel..." << endl;
        }

        // Warmup
        SDDMM_column_dist(coordinates, A, B, result); 

        auto t_start = std::chrono::steady_clock::now();
        for(int i = 0; i < num_trials; i++) {
            SDDMM_column_dist(coordinates, A, B, result); 
        }
        auto t_end = std::chrono::steady_clock::now();
        double millis_per_kernel = std::chrono::duration<double, std::milli>(t_end-t_start).count() / num_trials;
        BCL::barrier();
        if(BCL::rank() == 0) {
            cout << "Kernel took " << millis_per_kernel << " milliseconds" << endl;
            cout    << "Throughput is " 
                    << total_NNZ * r * 2.0 / (millis_per_kernel / 1000) * 1e-9 
                    << " GFLOPs" << endl;
        }
    }

    delete result;
    // freemm(&spMat); Slight memory free issue here, will fix later 
    */

    BCL::finalize();


}