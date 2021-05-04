#include <iostream>
#include <random>
#include <fstream>
#include <stdio.h>


using namespace std;

void generateMatrix(int M, int N, char* filename, int nnz_per_row) {
    FILE * fp;
    fp = fopen (filename, "w+");
    std::mt19937 g1 (rand());
    //std::ofstream ofs (filename, std::ofstream::out);

    fprintf(fp, "%%MatrixMarket matrix coordinate real general\n");
    fprintf(fp, "%d %d %d\n", M, N, nnz_per_row);
    //ofs << M << " " << N << " " << nnz_per_row * M << "\n";

    //int nnz_per_row = frac_nonzero * M;

    for(int i = 0; i < M; i++) {
        for(int j = 0; j < nnz_per_row; j++) {
            // ofs << (i + 1) << " " << (rand() % N) + 1 << " " << 1 << "\n";
            fprintf(fp, "%d %d %d\n", (i + 1), g1() % N, 1);
        }
    }
    fclose(fp);
    //ofs.close();
}

