#include <iostream>
#include <vector>
#include <utility>
#include <cblas.h>
#include <bcl/containers/DMatrix.hpp>


using namespace std;

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
