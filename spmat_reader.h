#include <vector>
#include <utility>

using namespace std;

vector<pair<size_t, size_t>> read_sparse_matrix_fraction(int proc_rank,
                                char* filename, 
                                size_t rowStart, 
                                size_t rowEnd, 
                                size_t colStart, 
                                size_t colEnd);