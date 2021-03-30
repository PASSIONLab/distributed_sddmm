#ifndef PACK_H
#define PACK_H

/*
 * Head and associated source file can pack a matrix into blocks; actually,
 * they can perform multiple levels of recursive blocking (a la GoToBLAS),
 * but we're only using a subset of the functionality. Tile_spec_t sets
 * parameters for every level of blocking.
 */

typedef struct tile_spec_t {
    int r;
    int c;
    int colMajor; // 1 if ColMajor, 0 otherwise,
                  // ignored if either row or column is -1
} tile_spec_t;

void pack(  double* in_mat, 
            double* out_mat, 
            tile_spec_t* tile_specs, 
            int max_tiling_levels, 
            int m, 
            int n
            );


void unpack(  double* in_mat, 
            double* out_mat, 
            tile_spec_t* tile_specs, 
            int max_tiling_levels, 
            int m, 
            int n
            );

#endif