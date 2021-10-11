#include <iostream>
#include <vector>
#include "json.hpp"
#include "common.h"
#include <mpi.h>

using namespace std;
using json = nlohmann::json;

void test_Isend(int compCount, int sendCount) {
    VectorXd x(compCount);
    VectorXd y(compCount);



    VectorXd u(sendCount);
    VectorXd v(sendCount); 
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    test_Isend(50000, 50000);
    MPI_Finalize();
}

