#include <iostream>
#include <cstdio>
#include <vector>
#include "patoh/patoh.h"

using namespace std;

void partition(vector<pair<size_t, size_t>> &coords, long int nRows, long int nCols) {
    cout << "Starting hypergraph partitioning..." << endl;
    // This assumes that the coordinates arrive in row-sorted order 
    PaToH_Parameters args;
    PPaToH_Parameters pargs = &args;
    PaToH_Initialize_Parameters(pargs, PATOH_CONPART, PATOH_SUGPARAM_SPEED);

    vector<int> pins;
    vector<int> xpins;

    pins.push_back(0);
    for(int i = 0; i < coords.size(); i++) {
        if(i > 0 && coords[i].first != coords[i-1].first) {
            if(coords[i].first < coords[i-1].first) {
                cout << "Error, coordinates not sorted in row order!" << endl;
            }

            pins.push_back(i); 
        }
        xpins.push_back((int) coords[i].second);
    }
    pins.push_back(coords.size());

    cout << "Set up pins, xpins!" << endl;

    int _c = nCols;
    int _n = nRows;
    int _nconst = 0;
    int useFixCells = 0;
    int* cwghts = NULL;
    int* nwghts = NULL;

    int* partvec = new int[_c];
    int* partweights = new int[pargs->_k * _nconst]; 
    float* targetweights = NULL;

    pargs->_k = 5; // For now, test partitioning into just 5 parts 

    cout << "Finished setting up partitioning parameters!" << endl;


    PaToH_Alloc(pargs, _c, _n, _nconst, cwghts, nwghts, xpins.data(), pins.data());

    int cut;

    PaToH_Part(pargs, _c, _n, _nconst, useFixCells,
        cwghts, nwghts, xpins.data(), pins.data(), targetweights,
        partvec, partweights, &cut);

    // PaToH_Free();
    cout << "Partitioned Hypergraph!" << endl;

}
