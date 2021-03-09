#include <iostream>
#include <cstdio>
#include "mm.h"
#include <vector>
#include <utility>
#include <algorithm>
#include <bcl/bcl.hpp>
#include <bcl/containers/DMatrix.hpp>

using namespace std;

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

std::vector<double> fetch_row(BCL::DMatrix<double> &A, size_t abs_row) {
    size_t tile_num =  abs_row / A.tile_shape()[0];
    size_t row_num  = abs_row % A.tile_shape()[0];
    // printf("Getting row %lu, which should be row %lu within tile %lu:\n",
    //        i, row_num, tile_num);
    return(A.get_tile_row(tile_num, 0, row_num));
}

void SDDMM_column_dist( vector<pair<size_t, size_t>> &coordinates, 
            BCL::DMatrix<double> &A,
            BCL::DMatrix<double> &B,
            double* result 
            ) {
    
    // We assume that the local coordinates are sorted by row so we can
    // just call this kernel repeatedly

    int currentRow = -1;

    vector<double> Btile = B.get_tile(BCL::rank(), 0);

    for(int i = 0; i < coordinates.size(); i++) {
        vector<double> Arow;
        if (coordinates[i].first != currentRow) {
            if(coordinates[i].first > currentRow) {
                cout << "Not ordered correctly." << endl;
            }
            Arow = fetch_row(A, coordinates[i].first); // Could do asynchronously, perhaps?
            double* Brow = Btile.data() + B.shape()[1] * coordinates[i].second;
            result[i] = cblas_ddot(B.shape()[1], Arow.data(), 1, Brow, 1);
        }
    }
}

int main(int argc, char** argv) {
    cout << "Starting kernel" << endl;

    if(argc < 3) {
        cout << "Usage: Provide filename of matrix market format, r value." << endl;
        return 1; 
    }

    struct mmdata spMat;

    if (initialize_mm(argv[1], &spMat)) {
        cout << "Error reading supplied matrix." << endl;
        return 1;
    } 
    size_t r = (size_t) atoi(argv[2]);

    // Start BCL once we've read the sparse matrix.
    size_t segment_size = 256;
	BCL::init(segment_size);

	BCL::DMatrix<double> A({(size_t) spMat.M, r}, BCL::BlockRow());
	BCL::DMatrix<double> B({(size_t) spMat.N, r}, BCL::BlockRow());

    BCL::barrier();

    // In case symmetric, materialize every entry for convenience

    vector<pair<size_t, size_t>> coordinates;

    // Block into column panels. Modify here if the distribution changes.
    size_t tile_start = BCL::rank() * B.tile_shape()[0];
    size_t tile_end = min(tile_start + (size_t) B.tile_shape()[0], (size_t) spMat.N);

    int NNZ = 0;
    int total_NNZ = 0; // Sanity check across all the processes
    if(spMat.symmetricity == 1) {
        for(int i = 0; i < spMat.NNZ; i++) {
            if(tile_start <= spMat.y[i] && spMat.y[i] < tile_end) {
                coordinates.emplace_back((size_t) spMat.x[i] - 1, (size_t) spMat.y[i] - 1); 
                // newX.push_back(spMat.x[i]);
                // newY.push_back(spMat.y[i]);
                NNZ++;
            }
            total_NNZ++;
            if(spMat.x[i] != spMat.y[i]) {
                if(tile_start <= spMat.x[i] && spMat.x[i] < tile_end) {
                    coordinates.emplace_back((size_t) spMat.y[i] - 1, (size_t) spMat.x[i] - 1); 
                    // newY.push_back(spMat.x[i]);
                    // newX.push_back(spMat.y[i]);
                    NNZ++;
                }
                total_NNZ++;
            }
        }

        // This fully materializes the sparse matrix (since otherwise, we only get the
        // entries above the diagonal if symmetric)
    }

    double* result = new double[coordinates.size()];

    SDDMM_column_dist(coordinates, A, B, result);
 

    /*if(BCL::rank() == 0) {
        cout << "Total number of nonzeros is " << total_NNZ << endl;
    }
    cout << "Rank " << BCL::rank() << " has " << NNZ << endl;
    */

    BCL::finalize();

    delete result;
    freemm(&spMat);
}