#pragma once

#include <cmath>
#include <iostream>
#include <utility>
#include <vector>
#include <memory>
#include <Eigen/Dense>
#include <string.h>
#include <mpi.h>

#include "CombBLAS/CombBLAS.h"
#include "sparse_kernels.h"
#include "common.h"
#include "als_conjugate_gradients.h"
#include "distributed_sparse.h"

using namespace std;
using namespace combblas;
using namespace Eigen;

class ShardedBlockCyclicColumn: public NonzeroDistribution {
public:
    int p, c;
    shared_ptr<FlexibleGrid> grid;

    ShardedBlockCyclicColumn(int M, int N, int p, int c, shared_ptr<FlexibleGrid> &grid) { 
        world = MPI_COMM_WORLD;
        this->p = p;
        this->c = c;
        this->grid = grid;
        rows_in_block = divideAndRoundUp(M, p) * c; 
        cols_in_block = divideAndRoundUp(N, p); 
    }

	int blockOwner(int row_block, int col_block) {
        int rowRank = row_block;
        int layerRank = col_block % c;

        return grid->get_global_rank(rowRank, layerRank, 0);
    }
};

/*
 * Unlike its non-striped counterpart, this algorithm uses reductions of smaller
 * messages instead of one large AllReduce 
 */
class Sparse15D_Dense_Shift : public Distributed_Sparse {
public:
    int fusionApproach;

    DenseMatrix accumulation_buffer;

    Sparse15D_Dense_Shift(SpmatLocal* S_input, int R, int c, int fusionApproach, KernelImplementation* k) 
        : Distributed_Sparse(k) 
    {
        this->fusionApproach = fusionApproach;
        this->c = c;

        if(p % c != 0) {
            if(proc_rank == 0) {
                cout << "Error, for 1.5D algorithm, must have c divide num_procs!" << endl;
                exit(1);
            }
        }

        algorithm_name = "1.5D Block Row Replicated S Striped AB Cyclic Shift";
        proc_grid_names = {"# Rows", "# Layers"};

        perf_counter_keys = 
                {"Replication Time",
                "Cyclic Shift Time",
                "Computation Time" 
                };

        grid.reset(new FlexibleGrid(p/c, c, 1, 1));

        r_split = false;

        this->M = S_input->M;
        this->N = S_input->N;

        ShardedBlockCyclicColumn standard_dist(M, N, p, c, grid);
        ShardedBlockCyclicColumn transpose_dist(N, M, p, c, grid);

        // Copies the nonzeros of the sparse matrix locally (so we can do whatever
        // we want with them; this does incur a memory overhead)
        S.reset(S_input->redistribute_nonzeros(&standard_dist, false, false));
        ST.reset(S->redistribute_nonzeros(&transpose_dist, true, false));

        localArows = divideAndRoundUp(this->M, p);
        localBrows = divideAndRoundUp(this->N, p);

        setRValue(R);

        #pragma omp parallel for
        for(int i = 0; i < S->coords.size(); i++) {
            S->coords[i].r %= localArows * c;
        }
        S->divideIntoBlockCols(localBrows, p, true); 

        #pragma omp parallel for
        for(int i = 0; i < ST->coords.size(); i++) {
            ST->coords[i].r %= localBrows * c;
        } 
        ST->divideIntoBlockCols(localArows, p, true); 

        S->own_all_coordinates();
        ST->own_all_coordinates();

        assert(fusionApproach == 1 || fusionApproach == 2);

        bool local_tpose;
        if(fusionApproach == 2) {
            local_tpose = false;
        }
        else {
            local_tpose = true;
        }

        S->initializeCSRBlocks(localArows * c, localBrows, -1, local_tpose);
        vector<spcoord_t>().swap(S->coords);
        ST->initializeCSRBlocks(localBrows * c, localArows, -1, local_tpose);
        vector<spcoord_t>().swap(ST->coords);
        check_initialized();
    }

    void setRValue(int R) {
        this->R = R;

        localAcols = R;
        localBcols = R; 

        aSubmatrices.clear();
        bSubmatrices.clear();

        // TODO: I'm pretty sure that this is broken...
        aSubmatrices.emplace_back(localArows * (c * grid->i + grid->j), 0, localArows, localAcols);
        bSubmatrices.emplace_back(localBrows * (c * grid->i + grid->j), 0, localBrows, localBcols);
    }
 
    void initial_shift(DenseMatrix *localA, DenseMatrix *localB, KernelMode mode) {
        // Empty on purpose
    }


    void de_shift(DenseMatrix *localA, DenseMatrix *localB, KernelMode mode) {
        // Empty on purpose
    }

    void fusedSpMM(DenseMatrix &localA, 
            DenseMatrix &localB, 
            VectorXd &Svalues, 
            VectorXd &sddmm_buffer,
            MatMode mode) {

        if(fusionApproach == 1) {
            Distributed_Sparse::fusedSpMM(localA, 
                localB, 
                Svalues, 
                sddmm_buffer,
                mode);
            return;
        }

        DenseMatrix *Arole, *Brole;
        SpmatLocal* choice;

        if(mode == Amat) {
            //assert(Svalues.size() == S->coords.size());
            Arole = &localA;
            Brole = &localB;
            choice = S.get();
        } 
        else if(mode == Bmat) {
            //assert(Svalues.size() == ST->coords.size());
            Arole = &localB;
            Brole = &localA;
            choice = ST.get(); 
        }
        else {
            assert(false);
        }

        BufferPair bBuf(Brole); 

        DenseMatrix broadcast_buffer;
		accumulation_buffer = DenseMatrix::Constant(Arole->rows() * c, R, 0.0); 
        choice->setValuesConstant(0.0);

        if(c > 1) {
            broadcast_buffer = DenseMatrix::Constant(Arole->rows() * c, R, 0.0); 
            auto t = start_clock();
            MPI_Allgather(Arole->data(), Arole->size(), MPI_DOUBLE,
                            broadcast_buffer.data(), Arole->size(), MPI_DOUBLE, grid->row_world);
            stop_clock_and_add(t, "Replication Time");
        }

        for(int i = 0; i < p / c; i++) {
            int block_id = pMod((grid->rankInCol - i) * c + grid->rankInRow, p);

            auto t = start_clock();
            kernel->triple_function(
                k_sddmmA,
                *choice,
                c > 1 ? broadcast_buffer : *Arole,
                *(bBuf.getActive()),
                block_id,
                0);
 
            kernel->triple_function(
                k_spmmA,
                *choice,
                accumulation_buffer,
                *(bBuf.getActive()),
                block_id,
                0);

            stop_clock_and_add(t, "Computation Time");

            t = start_clock();
            if(p > 1) {
                shiftDenseMatrix(bBuf, grid->col_world, pMod(grid->rankInCol + 1, p / c), 55);
                MPI_Barrier(MPI_COMM_WORLD);
            }
            stop_clock_and_add(t, "Cyclic Shift Time");
        }

        vector<int> recvCounts;
        for(int i = 0; i < c; i++) {
            recvCounts.push_back(Arole->rows() * R);
        }
    
        auto t = start_clock();
        bBuf.sync_active();
        stop_clock_and_add(t, "Computation Time");

        if(c > 1) {
            t = start_clock();
            MPI_Reduce_scatter(accumulation_buffer.data(), 
                    Arole->data(), recvCounts.data(),
                        MPI_DOUBLE, MPI_SUM, grid->row_world);
            stop_clock_and_add(t, "Replication Time");
        }
        else {
            auto t = start_clock();
            *Arole = accumulation_buffer;
            stop_clock_and_add(t, "Computation Time");
        }
        // TODO: Doesn't affect the applications, but this fused method
        // currently doesn't fill the SDDMM buffers.
    }

    VectorXd like_S_values(double value) {
        if(fusionApproach == 1) {
            return VectorXd::Constant(ST->owned_coords_end - ST->owned_coords_start, value); 
        }
        else {
            return VectorXd::Constant(S->owned_coords_end - S->owned_coords_start, value); 
        }
    }

    VectorXd like_ST_values(double value) {
        if(fusionApproach == 1) {
            return VectorXd::Constant(S->owned_coords_end - S->owned_coords_start, value); 
        }
        else {
            return VectorXd::Constant(ST->owned_coords_end - ST->owned_coords_start, value); 
        }
    }

    /*
     * Set the mode to take an SDDMM, SpMM with A as the output matrix, or 
     * SpMM with B as the output matrix. 
     */
    void algorithm(         DenseMatrix &localA, 
                            DenseMatrix &localB, 
                            VectorXd &SValues, 
                            VectorXd *sddmm_result_ptr, 
                            KernelMode mode,
                            bool initial_replicate 
                            ) {

        DenseMatrix *Arole, *Brole;
        SpmatLocal* choice;

        bool invert = fusionApproach == 1;
        if((mode == k_spmmA || mode == k_sddmmA) == invert) {
            Arole = &localB;
            Brole = &localA;
            choice = ST.get();
        } 
        else if((mode == k_spmmB || mode == k_sddmmB) == invert) {
            Arole = &localA;
            Brole = &localB;
            choice = S.get(); 
        }
        else {
            assert(false);
        }

        BufferPair bBuf(Brole); 

		// Temporary buffer that holds the results of the local ops; this buffer
		// is sharded and then reduced to the local portions of the matrix. 
        if(initial_replicate) {
            if(c > 1) {
                auto t = start_clock();
                accumulation_buffer = DenseMatrix::Constant(Arole->rows() * c, R, 0.0); 
                MPI_Allgather(Arole->data(), Arole->size(), MPI_DOUBLE,
                                accumulation_buffer.data(), Arole->size(), MPI_DOUBLE, grid->row_world);
                stop_clock_and_add(t, "Replication Time");  
            }
        }

        auto t = start_clock();
        if(mode == k_sddmmA || mode == k_sddmmB) {
            choice->setValuesConstant(0.0);
        }
        else {
            choice->setCSRValues(SValues);
        }
        stop_clock_and_add(t, "Computation Time"); 
 
        for(int i = 0; i < p / c; i++) {
            int block_id = pMod((grid->rankInCol - i) * c + grid->rankInRow, p);

            //assert(S->blockStarts[block_id] <= S->coords.size());
            //assert(S->blockStarts[block_id + 1] <= S->coords.size());

            KernelMode mode_temp;
            if(fusionApproach == 1) {
                mode_temp = mode == k_spmmA ? k_spmmB : mode;
            }
            else if(fusionApproach == 2) {
                mode_temp = mode == k_spmmB ? k_spmmA : mode;
            }
            else {
                assert(false);
            }

            auto t = start_clock();
            kernel->triple_function(
                mode_temp, 
                *choice,
                c > 1 ? accumulation_buffer : *Arole,
                *(bBuf.getActive()),
                block_id,
                0);
            stop_clock_and_add(t, "Computation Time"); 

            t = start_clock();
            if(p > 1) {
                shiftDenseMatrix(bBuf, grid->col_world, pMod(grid->rankInCol + 1, p / c), 55);
                MPI_Barrier(MPI_COMM_WORLD);
            }
            stop_clock_and_add(t, "Cyclic Shift Time");
        }        

        t = start_clock();
        bBuf.sync_active();
        stop_clock_and_add(t, "Computation Time");

        if(mode == k_sddmmA || mode == k_sddmmB) {
            auto t = start_clock();
            *sddmm_result_ptr = SValues.cwiseProduct(choice->getCSRValues());
            stop_clock_and_add(t, "Computation Time"); 
        }

        if(fusionApproach == 2 && (mode == k_spmmA || mode == k_spmmB)) {
            vector<int> recvCounts;
            for(int i = 0; i < c; i++) {
                recvCounts.push_back(Arole->rows() * R);
            }

            if(c > 1) {
                t = start_clock();
                MPI_Reduce_scatter(accumulation_buffer.data(), 
                        Arole->data(), recvCounts.data(),
                            MPI_DOUBLE, MPI_SUM, grid->row_world);
                stop_clock_and_add(t, "Replication Time");
            }
        }
    }
};
