#include <vector>
#include <utility>
#include <immintrin.h>
#include <omp.h>
#include <iostream>

using namespace std;

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

size_t kernel(vector<pair<size_t, size_t>> &coordinates, 
    double* A,
    double* B,
    size_t r,
    double* result,
    int start,
    int end) {   

    // We assume that the local coordinates are sorted by row so we can
    // just call this kernel repeatedly

    size_t processed = 0;

    // #pragma omp parallel for
    for(int i = start; i < end; i++) {
        processed++;
        double* Arow = A + r * coordinates[i].first;
        double* Brow = B + r * coordinates[i].second; 
        result[i] = vectorized_dot_product(Arow, Brow, r); 
    }
    return processed;
}