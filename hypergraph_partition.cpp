#include <iostream>
#include <cstdio>
#include <vector>
#include <unordered_set>
#include <fstream>
#include "patoh/patoh.h"
#include <chrono>
#include <ctime>

using namespace std;

void partition(vector<pair<size_t, size_t>> &coords, long int nRows, long int nCols, int num_procs, string filename) {
    cout << "Starting hypergraph partitioning..." << endl;
    // This assumes that the coordinates arrive in row-sorted order 
    PaToH_Parameters args;
    PPaToH_Parameters pargs = &args;
    PaToH_Initialize_Parameters(pargs, PATOH_CONPART, PATOH_SUGPARAM_SPEED);

    int *_pins; 
    int *_xpins;

    vector<int> pins;
    vector<int> xpins;

    unordered_set<size_t> nonzero_rows;   

    xpins.push_back(0);
    for(int i = 0; i < coords.size(); i++) {
        // nonzero_rows.insert(coords[i].first);
        if(i > 0 && coords[i].first != coords[i-1].first) {
            if(coords[i].first < coords[i-1].first) {
                cout << "Error, coordinates not sorted in row order!" << endl;
            }

            xpins.push_back(i);
        }
        pins.push_back((int) coords[i].second);
        if(coords[i].second > nCols)
            cout << "Error!" << endl;
    }

    xpins.push_back(coords.size());

    // cout << nonzero_rows.size() << endl;
    // cout << xpins.size() << endl;

    int _c = nCols;
    int _n = xpins.size() - 1;
    int _nconst = 1;
    int useFixCells = 0;
    int* cwghts = new int[_c];

    for(int i = 0; i < _c; i++) {
        cwghts[i] = 1;
    }

    // Every vertex has equal weight here

    int* nwghts = new int[_n];
    for(int i = 0; i < _n; i++) {
        nwghts[i] = 1;
    }

    /*PaToH_Read_Hypergraph("../patoh/ken-11.u", &_c, &_n, &_nconst, &cwghts, &nwghts,
        &xpins, &pins);*/

    _pins = pins.data();
    _xpins = xpins.data();

    pargs->_k = num_procs; // For now, test partitioning into just 5 parts 

    cout << "Set up pins, xpins!" << endl;

    int* partvec = new int[_c];
    int* partweights = new int[pargs->_k * _nconst]; 
    float* targetweights = NULL;

    cout << "Finished setting up partitioning parameters!" << endl;

    PaToH_Alloc(pargs, _c, _n, _nconst, cwghts, nwghts, _xpins, _pins);


    int cut;

    auto t_start = std::chrono::steady_clock::now();

    if(num_procs > 1) {
        PaToH_Part(pargs, _c, _n, _nconst, useFixCells,
            cwghts, nwghts, _xpins, _pins, targetweights,
            partvec, partweights, &cut);
    }
    else {
        for(int i = 0; i < _c; i++) {
            partvec[i] = 0;
        }
        cut = 0;
    }
    auto t_end = std::chrono::steady_clock::now();
    double seconds = std::chrono::duration<double, std::milli>(t_end-t_start).count() / 1000.0;
 

    cout << "Hypergraph Partitioning took " << seconds << " seconds." << endl;
    printf("%d-way cutsize is: %d\n", args._k, cut);

    // Write information about the hypergraph partitioning
    // to an output file

    vector<vector<int>> processor_assignments;
    for(int i = 0; i < num_procs; i++) {
        processor_assignments.emplace_back();
    }
    for(int i = 0; i < _c; i++) {
        processor_assignments[partvec[i]].push_back(i);
    }

    int maxLen = -1;
    for(int i = 0; i < num_procs; i++) {
        maxLen = max(maxLen, (int) processor_assignments[i].size());
    }

    ofstream ofs (filename, std::ofstream::out);

    // First line: Max Length of any bucket and seconds taken for hypergraph partitioning 
    ofs << maxLen << " " << seconds << "\n";


    // Subsequent line: Column coordinate, which processor, which
    // row within that processor

    for(int i = 0; i < num_procs; i++) {
        for(int j = 0; j < processor_assignments[i].size(); j++) {
            ofs << processor_assignments[i][j] << " " << i << " " << j << "\n";
        }
    }
    ofs.flush();
    ofs.close();

    // PaToH_Free();

}
