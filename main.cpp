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
#include <unordered_map>
#include <mpi.h>

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

inline  BCL::future<std::vector<double, BCL::bcl_allocator<double>>>
fetch_row(BCL::DMatrix<double> &A, size_t abs_row) {
    size_t tile_num =  abs_row / A.tile_shape()[0];
    size_t row_num  = abs_row % A.tile_shape()[0];
    return A.arget_tile_row(tile_num, 0, row_num);
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
            vector<int> active_rows,
            BCL::DMatrix<double> &A,
            double* B_local, 
            double* result,
            bool printstatistics
            ) {
    
    // We assume that the local coordinates are sorted by row so we can
    // just call this kernel repeatedly


    int num_row_fetches = 0;
    // vector<double> Atile = A.get_tile(BCL::rank(), 0);
    vector<BCL::future<std::vector<double, BCL::bcl_allocator<double>>>> rows;
    rows.reserve(active_rows.size());

    for(int i = 0; i < active_rows.size(); i++) {
        rows.push_back(fetch_row(A, active_rows[i]));
    }

    vector<double, BCL::bcl_allocator<double>> Arow;

    int currentRow = -1;
    for(int i = 0; i < coordinates.size(); i++) {
        if ((int) coordinates[i].first != currentRow) {
            if((int) coordinates[i].first < currentRow) {
                cout << "Not ordered correctly " << coordinates[i].first << endl;
            }

            currentRow = (int) coordinates[i].first;
            Arow = rows[num_row_fetches].get();
            num_row_fetches++;
        }

        double* Brow = B_local + A.shape()[1] * coordinates[i].second;
        result[i] = vectorized_dot_product(Arow.data(), Brow, A.shape()[1]);
    }

    if(printstatistics) {
        int buffer = num_row_fetches;
        num_row_fetches = BCL::allreduce(buffer, std::plus<int>{});

        cout << "Rank " << BCL::rank() << " " << coordinates.size() << endl;

        if(BCL::rank() == 0 ) {
            cout << "Total number of row fetches: " << num_row_fetches << endl;
        }
    }

}

int main(int argc, char** argv) {
    if(argc < 3) {
        if(BCL::rank() == 0) {
            cout << "Usage: Provide filename of matrix market format, r value." << endl;
        }
        return 1; 
    }

    struct mmdata spMat; 

    if (initialize_mm(argv[1], &spMat)) {
        if(BCL::rank() == 0) {
            cout << "Error reading supplied matrix." << endl;
        }
        return 1;
    } 

    size_t r = (size_t) atoi(argv[2]);

    // Start BCL once we've read the sparse matrix.
    size_t segment_size = 512; // Segment size in megabytes
	BCL::init(segment_size);
    // In case symmetric, materialize every entry for convenience

    if(BCL::rank() == 0) {
        cout    << "Reading Sparse Matrix with " << spMat.M << " rows,"
                << spMat.N << " columns, and " << spMat.NNZ << " nonzeros." << endl;
    }

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
        cout << "Total Number of Processes: " << BCL::nprocs() << endl;
        if(! file_exists(partition_filepath)) {
            partition(coordinates, spMat.M, spMat.N, BCL::nprocs(), partition_filepath);
        }
    }

    BCL::barrier();

    // Now filter down to only the coordinates that we care about
    unordered_map<int, int> my_column_embeds; 
    vector<pair<size_t, size_t>> local_coordinates;
    ifstream f(partition_filepath);
    // f.clear();
    // f.seekg(0, ios::beg);

    int maxLen;
    double partition_time;
    f >> maxLen >> partition_time;

	BCL::DMatrix<double> A({(size_t) spMat.M, r}, BCL::BlockRow());

    // TODO: THIS ONLY WORKS IF EVERYTHING IS SYMMETRIC!!
    // These are for the case when you don't want to do a hypergraph partition
    size_t tile_start = BCL::rank() * A.tile_shape()[0];
    size_t tile_end = min(tile_start + (size_t) A.tile_shape()[0], (size_t) spMat.M);

    for(size_t i = 0; i < spMat.N; i++) {
        if(strcmp(argv[4], "hypergraph") == 0) {
            int col_idx, processor, local_idx;
            f >> col_idx >> processor >> local_idx;

            if ((size_t) processor == BCL::rank()) {
                my_column_embeds.emplace(col_idx, local_idx); 
            }
        }
        else if(strcmp(argv[4], "simple") == 0) {
            if(tile_start <= i && i < tile_end) {
                my_column_embeds.emplace(i, i - tile_start);
            }
        }
    }

    // cout << argv[4] << endl;

    f.close();

    // Initialize the distributed and local dense matrices
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


    for(int i = 0; i < coordinates.size(); i++) {
        auto it = my_column_embeds.find(coordinates[i].second);
        if(it != my_column_embeds.end()) {
            local_coordinates.emplace_back(coordinates[i].first, (size_t) it->second);
        }
    }


    sort(local_coordinates.begin(), local_coordinates.end(), coord_sort_key());
    // Isolate only the active rows

    vector<int> active_rows;
    int currentRow = -1;
    for(auto it = local_coordinates.begin(); it != local_coordinates.end(); it++) {
        if((int) (*it).first != currentRow ) {
            currentRow = (int) (*it).first;
            active_rows.push_back(currentRow);
        }
    }

    cout << "Rank " << BCL::rank() << " nonzeros:\t " << local_coordinates.size() << endl; 
    
    if(BCL::rank() == 0) {
        cout << "Total Nonzeros (Fully Materialized in Symmetric Case): " << coordinates.size() << endl;
    } 

    BCL::barrier();

    double* result = new double[local_coordinates.size()];

    int num_trials = 40;

    if(strcmp(argv[3], "serial") == 0) {
        vector<double> Atile = A.get_tile(BCL::rank(), 0);

        cout << "Starting serial kernel..." << endl;
        auto t_start = std::chrono::steady_clock::now();
        for(int i = 0; i < num_trials; i++) {
            serial_kernel(local_coordinates, Atile.data(), B_local, r, result);
        }
        auto t_end = std::chrono::steady_clock::now();
        double millis_per_kernel = std::chrono::duration<double, std::milli>(t_end-t_start).count() / num_trials;

        cout << "Kernel took " << millis_per_kernel << " milliseconds" << endl;
        cout    << "Throughput is " 
                << coordinates.size() * r * 2.0 / (millis_per_kernel / 1000) * 1e-9 
                << " GFLOPs" << endl;
    }
    else {

        if(BCL::rank() == 0) {
            cout << "Starting parallel kernel..." << endl;
        }

        // Warmup (and print statistics)
        SDDMM_column_dist(local_coordinates, active_rows, A, B_local, result, true); 

        auto t_start = std::chrono::steady_clock::now();
        for(int i = 0; i < num_trials; i++) {
            SDDMM_column_dist(local_coordinates, active_rows, A, B_local, result, false); 
        }
        auto t_end = std::chrono::steady_clock::now();
        double millis_per_kernel = std::chrono::duration<double, std::milli>(t_end-t_start).count() / num_trials;
        BCL::barrier();
        if(BCL::rank() == 0) {
            cout << "Kernel took " << millis_per_kernel << " milliseconds" << endl;
            cout    << "Throughput is " 
                    << coordinates.size() * r * 2.0 / (millis_per_kernel / 1000) * 1e-9 
                    << " GFLOPs" << endl;
        }
    }

    delete result;
    // freemm(&spMat); Slight memory free issue here, will fix later 

    BCL::finalize();
}