#include <iostream>
#include <cstdio>
#include <vector>
#include <unordered_set>
#include <fstream>
#include <chrono>
#include <cstring>
#include <ctime>
#include "metis.h"

using namespace std;


void graph_partition(vector<pair<size_t, size_t>> &coords, long int nRows, long int nCols, int num_procs, string filename) {
    cout << "Starting graph partitioning..." << endl;

    // Precondition: The coordinates arrive in row-sorted order 

    idx_t* nnz_per_row = (idx_t*) calloc(nRows, sizeof(idx_t));
    idx_t* row_starts = (idx_t*) calloc(nRows + 1, sizeof(idx_t));
    idx_t* adjacencies = new idx_t[coords.size()];

    for(int i = 0; i < nRows; i++) {
        nnz_per_row[coords[i].first]++;
        adjacencies[i] = coords[i].second;
    }

    
    for(int i = 1; i < nRows + 1; i++) {
        row_starts[i] = row_starts[i - 1] + nnz_per_row[i - 1];
    }


    idx_t* vwghts = new idx_t[nCols];

    // We are going to set the cell weights by the number of nonzeros in a particular column;
    // This will give us load balance 
    memset(vwghts, 0, sizeof(idx_t) * nCols);

    for(int i = 0; i < coords.size(); i++) {
        vwghts[coords[i].second]++;
    }

    idx_t nvtxs = nCols;
    idx_t ncon = 1; 
    idx_t* xadj = row_starts;
    idx_t* adjncy = adjacencies;
    idx_t* vsize = NULL;
    idx_t* adjwgt = NULL;
    idx_t nparts = num_procs;
    real_t* tpwgts = NULL;
    real_t* ubvec = NULL;

    cout << "Set up pins, xpins!" << endl;

    idx_t* part = new idx_t[nCols];
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_VOL;

    cout << "Finished setting up partitioning parameters!" << endl;

    idx_t cut;

    auto t_start = std::chrono::steady_clock::now();

    if(num_procs > 1) {
        METIS_PartGraphKway(&nvtxs, &ncon, xadj, adjncy,
        vwghts, vsize, adjwgt, &nparts, tpwgts,
        ubvec, options, &cut, part);
    }
    else {
        for(int i = 0; i < nCols; i++) {
            part[i] = 0;
        }
        cut = 0;
    }
    auto t_end = std::chrono::steady_clock::now();
    double seconds = std::chrono::duration<double, std::milli>(t_end-t_start).count() / 1000.0;
 
    cout << "Hypergraph Partitioning took " << seconds << " seconds." << endl;
    printf("%d-way cutsize is: %d\n", num_procs, cut);

    // Write information about the hypergraph partitioning
    // to an output file

    vector<vector<int>> processor_assignments;
    for(int i = 0; i < num_procs; i++) {
        processor_assignments.emplace_back();
    }
    for(int i = 0; i < nCols; i++) {
        processor_assignments[part[i]].push_back(i);
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

}