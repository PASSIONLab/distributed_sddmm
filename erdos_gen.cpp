#include <iostream>
#include <random>
#include <fstream>

using namespace std;

void generateMatrix(int M, int N, char* filename, int nnz_per_row) {
    std::ofstream ofs (filename, std::ofstream::out);

    ofs << "%%MatrixMarket matrix coordinate real general\n";
    ofs << M << " " << N << " " << nnz_per_row * M << "\n";

    //int nnz_per_row = frac_nonzero * M;
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < nnz_per_row; j++) {
            ofs << (i + 1) << " " << (rand() % N) + 1 << " " << 1 << "\n";
        }
    }

    ofs.close();
}

