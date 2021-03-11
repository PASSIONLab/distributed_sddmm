#include <iostream>
#include <cstdio>
#include <vector>
#include "patoh/patoh.h"

using namespace std;

void partition(vector<pair<size_t, size_t>> &coords, int nRows, int nCols) {
    // This assumes that the coordinates arrive in row-sorted order 
    PaToH_Parameters pargs;
    PaToH_Initialize_Parameters(&pargs, PATOH_CONPART, PATOH_SUGPARAM_SPEED);

    vector<int> pins;
    vector<int> xpins;

    pins.push_back(0);
    for(int i = 0; i < coords.size(); i++) {
        if(i > 0 && coords[i].first != coords[i-1].first) {
            pins.push_back(i); 
        }
        xpins.push_back(coords[i].second);
    }
    pins.push_back(coords.size());

    int _c = nCols;
    int _n = nRows;
    int _nconst = 0;
    int useFixCells = 0;
    int* cwghts = NULL;
    int* nwghts = NULL;

    int* partvec = new int[_c];
    int* partweights = NULL;
    int* targetweights = NULL;

    int cut;

    PaToH_Part(pargs, _c, _n, _nconst, useFixCells,
        cwghts, nwghts, t *xpins, pins.data(), targetweights,
        partvec, partweights, &cut);


    PaToH_Free();
}
