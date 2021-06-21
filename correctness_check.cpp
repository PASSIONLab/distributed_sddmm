#include <iostream>
#include "15D_sparse.hpp"
#include "25D_sparse.hpp"
#include "pack.h"

using namespace std;

int main(int argc, char** argv) {
    tile_spec_t Atile[2] = {{rowAwidth, colWidth,  1}, {-1, 0, 0}};
    tile_spec_t Btile[2] = {{colWidth,  rowBwidth, 1}, {-1, 0, 0}}; 

}
