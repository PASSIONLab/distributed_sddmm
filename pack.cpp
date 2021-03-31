#include <cassert>
#include <algorithm>
#include <iostream>
#include "pack.h"

using namespace std;

void pack_helper(  
            double* in_mat, 
            double* out_mat,
            int coordX,
            int coordY,
            int dimR,
            int dimC,
            tile_spec_t* tile_specs,
            int max_tiling_levels,
            int level,
            int m,
            int n,     // This deals specifically with the output matrix
            int unpack
          ) {
              
    assert(! ((tile_specs[level].r == -1) && (tile_specs[level].c == -1)));
    
    int M = tile_specs[level].r;
    int N = tile_specs[level].c;

    if(tile_specs[level].r == -1) {
        M = dimR;
    }

    if(tile_specs[level].c == -1) {
        N = dimC;
    }

    if (level == max_tiling_levels - 1) {
        for(int i = 0; i < dimR; i++) {
            for(int j = 0; j < dimC; j++) {
                // Lowest level of unpacking
                if(tile_specs[level].colMajor) {
                    if(unpack) {
                        in_mat[coordX + i + (coordY + j) * m] = out_mat[j * dimR + i];      
                    }
                    else {
                        out_mat[j * dimR + i] = in_mat[coordX + i + (coordY + j) * m]; // Toggle from m to n to switch from row to column major input
                    }
                }
                else {
                    if(unpack) {
                        in_mat[coordX + i + (coordY + j) * m] = out_mat[i * dimC + j];
                    }
                    else {
                        out_mat[i * dimC + j] = in_mat[coordX + i + (coordY + j) * m]; 
                    }
                }
                // End lowest level of unpacking
            }
        }
    }
    else {
        assert(M > 0);
        assert(N > 0);

        if(tile_specs[level].colMajor == 1) {
            for(int j = 0; j < dimC; j+= N) {
                for(int i = 0; i < dimR; i+= M) {
                    int rWidth = min(M, dimR - i);
                    int cWidth = min(N, dimC - j);

                    pack_helper(in_mat, 
                        out_mat, 
                        coordX + i, 
                        coordY + j, 
                        rWidth, 
                        cWidth, 
                        tile_specs, 
                        max_tiling_levels, 
                        level + 1, 
                        m, 
                        n, 
                        unpack);
                    out_mat += rWidth * cWidth;

                }
            }
        }
        else {
            for(int i = 0; i < dimR; i+= M) {
                for(int j = 0; j < dimC; j+= N) {
                    int rWidth = min(M, dimR - i);
                    int cWidth = min(N, dimC - j);

                    pack_helper(in_mat, 
                        out_mat, 
                        coordX + i, 
                        coordY + j, 
                        rWidth, 
                        cWidth, 
                        tile_specs, 
                        max_tiling_levels, 
                        level + 1, 
                        m, 
                        n, 
                        unpack);

                    out_mat += rWidth * cWidth;
                }
            }
        }
    }
}

// TODO: Should really rename these from in_mat, out_mat
void pack(  double* in_mat, 
            double* out_mat, 
            tile_spec_t* tile_specs, 
            int max_tiling_levels, 
            int m, 
            int n
            ) {
    pack_helper(in_mat, 
                out_mat, 
                0, 0, 
                m, n, 
                tile_specs, 
                max_tiling_levels, 
                0, 
                m, n, 
                0);
}

// TODO: Should really rename these from in_mat, out_mat
void unpack(double* in_mat, 
            double* out_mat, 
            tile_spec_t* tile_specs, 
            int max_tiling_levels, 
            int m, 
            int n
            ) {
    pack_helper(in_mat, 
                out_mat, 
                0, 0, 
                m, n, 
                tile_specs, 
                max_tiling_levels, 
                0, 
                m, n, 
                1);
}