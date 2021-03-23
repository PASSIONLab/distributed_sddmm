#include <vector>
#include <utility>


using namespace std;

void serial_kernel(vector<pair<size_t, size_t>> &coordinates, 
    double* A,
    double* B,
    size_t r,
    double* result);