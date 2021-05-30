#include <iostream>
#include <Eigen/Dense>

#include "common.h"

using namespace std;
using namespace Eigen;

// The simplest possible program: solves a random dense linear system using
// conjugate gradients, and compares it to an exact solution. 

// Demmel, Applied Numerical Linear Algebra: Conjugate Gradient Algorithm
// is on page 312 

int main(int argc, char** argv) {
    cout << "Starting serial conjugate gradient algorithm!" << endl; 

    int rows = 50;
    int cols = 20;

    DenseMatrix X(rows, cols);
    VectorXd b(rows);
    VectorXd z(cols);

    X.setRandom(rows, cols);
    b.setRandom(rows);

    VectorXd x = VectorXd::Zero(rows);
    DenseMatrix A = 0.01 * MatrixXd::Identity(rows, rows) + (X * X.transpose());// .selfadjointView<Eigen::Upper>();

    double tol = 1e-10;
    int max_iter = 10000;
    int k = 0;

    // Conjugate gradient algorithm taken directly from Wikipedia 
    VectorXd r = b;
    VectorXd p = r;
    double rsold = r.dot(r);

    bool max_iterations_exceeded = true; 
    while(k < b.size()) {
        k++;
        VectorXd Ap = A * p;
        
        double alpha = rsold / p.dot(Ap);
        x += alpha * p;
        r -= alpha * Ap;
        double rsnew = r.dot(r);

        
        if(sqrt(rsnew) < tol) {
            max_iterations_exceeded = false;
            break;
        }

        p = r + (rsnew / rsold) * p;
        rsold = rsnew;
    }

    if(max_iterations_exceeded) {
        cout << "CG algorithm exceeded maximum number of iterations!" << endl;
    }
    else {
        cout << "CG algorithm converged in " << k << " iterations!" << endl;
    }

    VectorXd solution = A.colPivHouseholderQr().solve(b);

    // cout << A * r - b << endl;
    cout << A * solution - b << endl;

}
