#include <iostream>
#include <mpi.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <utility>

#include "mm.h"

using namespace std;

// Put an array of coordinates into row-major order
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

/**
 * Reads a 2D slice of a sparse matrix into memory, with the slice boundaries
 * givne by [rowStart, rowEnd) and [colStart, colEnd)
 */
vector<pair<size_t, size_t>> read_sparse_matrix_fraction(int proc_rank,
                                char* filename, 
                                size_t rowStart, 
                                size_t rowEnd, 
                                size_t colStart, 
                                size_t colEnd) {

    cout << "Beginning Sparse Matrix Read..." << endl;
    struct mmdata spMat; 

    if (initialize_mm(filename, &spMat)) {
        if(proc_rank == 0) {
            cout << "Error reading supplied matrix." << endl;
        }
        exit(1); 
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
    if(proc_rank == 0) {
        cout    << "Read sparse matrix with " << spMat.M << " rows,"
                << spMat.N << " columns, and " << coordinates.size() << " nonzeros." << endl;
    }

    // TODO: Implement a random permutation here to load balance everything

    // Now filter to only the coordinates that we care about. 
     
    vector<pair<size_t, size_t>> local_coordinates;

    for(int i = 0; i < coordinates.size(); i++) {
        if(rowStart <= coordinates[i].first 
            && coordinates[i].first < rowEnd
            && colStart <= coordinates[i].second
            && coordinates[i].second < colEnd 
            ) {
            local_coordinates.emplace_back(coordinates[i].first, coordinates[i].second);
        }
    }

    sort(local_coordinates.begin(), local_coordinates.end(), coord_sort_key()); 
    return local_coordinates;
}