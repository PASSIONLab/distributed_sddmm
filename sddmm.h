#include <vector>
#include <utility>


using namespace std;

size_t kernel(int64_t* rCoords,
    int64_t* cCoords,
    double* A,
    double* B,
    size_t r,
    double* result,
    int start,
    int end);