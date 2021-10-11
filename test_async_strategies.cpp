#include <iostream>
#include <vector>
#include "json.hpp"
#include "common.h"
#include <mpi.h>

using namespace std;
using json = nlohmann::json;

/*
 * You must run these tests with exactly two MPI processes.
 */

void test_Isend(uint64_t compCount, uint64_t sendCount) {
    int proc_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

    VectorXd x(compCount);
    VectorXd y(compCount);

    VectorXd u(sendCount);

    MPI_Status stat;
    MPI_Request sendreq, recvreq;

    if(proc_rank == 0) {
        MPI_Isend(u.data(), sendCount, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &sendreq);
    }

    if(proc_rank == 1) {
        MPI_Irecv(u.data(), sendCount, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &recvreq);
    }

    auto t = start_clock();
    VectorXd z = x.array() * y.array();
    double znorm = z.norm();

    if(proc_rank == 0) {
        cout << znorm << endl;
    }

    if(proc_rank == 0) {
        cout << "Computation time: " << stop_clock_get_elapsed(t) << endl;
    }

    t = start_clock();
    if(proc_rank == 0) {
        MPI_Wait(&sendreq, &stat);
    }

    if(proc_rank == 1) {
        MPI_Wait(&recvreq, &stat);
        cout << "Additional time spent in communication: " << stop_clock_get_elapsed(t) << endl;
    }

}

void test_RMA(int compCount, int sendCount) {
    int proc_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

    VectorXd x(compCount);
    VectorXd y(compCount);

    VectorXd u(sendCount);

    MPI_Status stat;
    MPI_Request req;

    MPI_Info info;
    MPI_Info_create(&info);

    MPI_Win window;
    double* ptr;
    MPI_Win_allocate( sizeof(double) * sendCount, 8, info, MPI_COMM_WORLD, &ptr, &window );

    MPI_Win_fence(0, window);

    if(proc_rank == 1) {
        MPI_Raccumulate(u.data(), sendCount, MPI_DOUBLE, 0, 0, 
                sendCount, MPI_DOUBLE, MPI_SUM, window, &req);
    }

    auto t = start_clock();
    VectorXd z = x.array() * y.array();
    double znorm = z.norm();

    if(proc_rank == 0) {
        cout << znorm << endl;
    }

    if(proc_rank == 1) {
        cout << "Computation time: " << stop_clock_get_elapsed(t) << endl;
    }

    if(proc_rank == 1) {
        t = start_clock();
        MPI_Wait(&req, &stat);
        cout << "Additional time spent in communication: " << stop_clock_get_elapsed(t) << endl;
    }
    MPI_Win_fence(0, window);
    MPI_Win_free(&window);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    test_Isend(500000000/*00000*/, 500000000);
    //test_RMA(500000000/*00000*/, 50000000);

    MPI_Finalize();
}

