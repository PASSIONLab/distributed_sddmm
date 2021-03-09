#include <vector>
#include <utility>
#include <bcl/bcl.hpp>
#include <bcl/containers/DMatrix.hpp>



using namespace std;

void SDDMM_column_dist( vector<pair<size_t, size_t>> &, 
            BCL::DMatrix<double> &,
            BCL::DMatrix<double> &,
            double* 
            );