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


class ShardedBlockRow : public NonzeroDistribution {
public:
    int p, c;
    shared_ptr<FlexibleGrid> grid;

    DenseMatrix accumulation_buffer;

    ShardedBlockRow(int M, int N, int p, int c, shared_ptr<FlexibleGrid> &grid) { 
        world = MPI_COMM_WORLD;
        this->p = p;
        this->c = c;
        this->grid = grid;
        rows_in_block = divideAndRoundUp(M, p); 
        cols_in_block = N; 
    }

	int blockOwner(int row_block, int col_block) {
        int rowRank = row_block / c;
        int layerRank = row_block % c;

        return grid->get_global_rank(rowRank, layerRank, 0);
    }
};


class Sparse15D_Sparse_Shift : public Distributed_Sparse {
public:
    DenseMatrix accumulation_buffer;
    int blockAwidth, blockBwidth;

    vector<int> nnz_in_row_axis, nnz_in_row_axis_tpose;

    Sparse15D_Sparse_Shift(SpmatLocal* S_input, int R, int c, KernelImplementation* k) 
        : Distributed_Sparse(k) 
    {
        this->c = c;

        if(p % c != 0) {
            if(proc_rank == 0) {
                cout << "Error, for 1.5D algorithm, must have c divide num_procs!" << endl;
                exit(1);
            }
        }

        algorithm_name = "1.5D Sparse Shifting Dense Replicating Algorithm";
        proc_grid_names = {"# Rows", "# Layers"};

        perf_counter_keys = 
                {"Replication Time",
                "Cyclic Shift Time",
                "Computation Time" 
                };

        grid.reset(new FlexibleGrid(p/c, c, 1, 1));

        r_split = true;

        A_R_split_world = grid->col_world;
        B_R_split_world = grid->col_world;

        this->M = S_input->M;
        this->N = S_input->N;

        ShardedBlockRow standard_dist(M, N, p, c, grid);
        ShardedBlockRow transpose_dist(N, M, p, c, grid);

        // Copies the nonzeros of the sparse matrix locally (so we can do whatever
        // we want with them; this does incur a memory overhead)
        S.reset(S_input->redistribute_nonzeros(&standard_dist, false, false));
        ST.reset(S->redistribute_nonzeros(&transpose_dist, true, false));

        blockAwidth = divideAndRoundUp(this->M, p);
        blockBwidth = divideAndRoundUp(this->N, p);

        localArows = blockAwidth * p / c;
        localBrows = blockBwidth * p / c;

        setRValue(R);

        #pragma omp parallel for
        for(int i = 0; i < S->coords.size(); i++) {
            S->coords[i].r %= blockAwidth; 
        }

        #pragma omp parallel for
        for(int i = 0; i < ST->coords.size(); i++) {
            ST->coords[i].r %= blockBwidth; 
        }

        nnz_in_row_axis.resize(p / c);
        nnz_in_row_axis_tpose.resize(p / c);

        int my_nnz = S->coords.size();
        int my_nnz_tpose = ST->coords.size();
        MPI_Allgather(&my_nnz, 1, MPI_INT, nnz_in_row_axis.data(), 
                1, MPI_INT, grid->col_world);

        MPI_Allgather(&my_nnz_tpose, 1, MPI_INT, nnz_in_row_axis_tpose.data(), 
                1, MPI_INT, grid->col_world);

        int max_nnz = *(std::max_element(nnz_in_row_axis.begin(), nnz_in_row_axis.end()));
        int max_nnz_tpose = *(std::max_element(nnz_in_row_axis_tpose.begin(), nnz_in_row_axis_tpose.end()));

        S->own_all_coordinates();
        ST->own_all_coordinates();

        S->monolithBlockColumn();
        ST->monolithBlockColumn();

        S->initializeCSRBlocks(blockAwidth, localArows, max_nnz, false);
        vector<spcoord_t>().swap(S->coords);
        ST->initializeCSRBlocks(blockBwidth, localBrows, max_nnz_tpose, false);
        vector<spcoord_t>().swap(ST->coords);
        check_initialized();
    }

    void setRValue(int R) {
        this->R = R;

        localAcols = R * c / p;
        localBcols = R * c / p; 

        if(localAcols * p / c != R) {
            cout << "Error, R must be divisible by p / c!" << endl;
        }

        aSubmatrices.clear();
        bSubmatrices.clear();

        for(int i = 0; i < p / c; i++) {
            aSubmatrices.emplace_back(blockAwidth * (grid->j + c * i), 
                    localAcols * grid->i, blockAwidth, localAcols);
            bSubmatrices.emplace_back(blockBwidth * (grid->j + c * i), 
                    localBcols * grid->i, blockBwidth, localBcols);
        }
    }

    void initial_shift(DenseMatrix *localA, DenseMatrix *localB, KernelMode mode) {
        // Empty on purpose
    }

    void de_shift(DenseMatrix *localA, DenseMatrix *localB, KernelMode mode) {
        // Empty on purpose
    }


    void algorithm(         DenseMatrix &localA, 
                            DenseMatrix &localB, 
                            VectorXd &SValues, 
                            VectorXd *sddmm_result_ptr, 
                            KernelMode mode,
                            bool initial_replicate 
                            ) {
        DenseMatrix *Arole, *Brole;
        SpmatLocal* choice;
        int arBwidth, brBwidth;
        vector<int> nnz_in_axis;

        if(mode == k_spmmA || mode == k_sddmmA) {
            Arole = &localA;
            Brole = &localB;
            choice = S.get();
            arBwidth = blockAwidth;
            brBwidth = blockBwidth;
            nnz_in_axis = nnz_in_row_axis;
        } 
        else if(mode == k_spmmB || mode == k_sddmmB) {
            Arole = &localB;
            Brole = &localA;
            choice = ST.get();
            arBwidth = blockBwidth;
            brBwidth = blockAwidth;
            nnz_in_axis = nnz_in_row_axis_tpose;
        }
        else {
            assert(false);
        }

		// Temporary buffer that holds the results of the local ops; this buffer
		// is sharded and then reduced to the local portions of the matrix. 
        if(initial_replicate) {
            auto t = start_clock();
        
            if(c > 1) {
                int cols = Brole->cols();
                accumulation_buffer = DenseMatrix::Constant(Brole->rows() * c, cols, 0.0); 
                for(int i = 0; i < p / c; i++) {
                    MPI_Allgather(Brole->data() + brBwidth * cols * i, brBwidth * cols, MPI_DOUBLE,
                                    accumulation_buffer.data() + brBwidth * cols * c * i, brBwidth * cols, MPI_DOUBLE, grid->row_world);
                }
            }
            stop_clock_and_add(t, "Replication Time");  
        }

        auto t = start_clock();
        if(mode == k_sddmmA || mode == k_sddmmB) {
            choice->setValuesConstant(0.0);
        }
        else {
            choice->setCSRValues(SValues);
        }
        stop_clock_and_add(t, "Computation Time"); 

        DenseMatrix tmp(arBwidth, Arole->cols());

        for(int i = 0; i < p / c; i++) {
            auto t = start_clock();
            int block_id = pMod(grid->i - i, p / c);

            if(mode == k_sddmmA || mode == k_sddmmB) {
                tmp = Arole->middleRows(block_id * arBwidth, arBwidth);
            }
            else {
                tmp *= 0.0;
            }

            kernel->triple_function(
                mode == k_spmmB ? k_spmmA : mode,
                *choice,
                tmp,
                c > 1 ? accumulation_buffer : *Brole,
                0,
                0); // TODO: Need to modify this offset! 

            if(mode == k_spmmA || mode == k_spmmB) {
                Arole->middleRows(block_id * arBwidth, arBwidth) = tmp; 
            }
            stop_clock_and_add(t, "Computation Time"); 

            if(p > 1) {
                t = start_clock();
                int src = pMod(grid->i - 1, p / c);
                int dst = pMod(grid->i + 1, p / c);

                if(mode==k_sddmmA || mode==k_sddmmB) {
                    choice->csr_blocks[0]->shiftCSR(src, dst, grid->col_world, nnz_in_axis[pMod(grid->i - i - 1, p / c)], 72, coo);
                    choice->blockStarts[1] = choice->csr_blocks[0]->num_coords;
                }
                else {
                    choice->csr_blocks[0]->shiftCSR(src, dst, grid->col_world, nnz_in_axis[pMod(grid->i - i - 1, p / c)], 72, csr);
                    choice->blockStarts[1] = choice->csr_blocks[0]->num_coords;
                }

                MPI_Barrier(MPI_COMM_WORLD);
                stop_clock_and_add(t, "Cyclic Shift Time");
            }
        }        

        if(mode == k_sddmmA || mode == k_sddmmB) {
            auto t = start_clock();
            *sddmm_result_ptr = SValues.cwiseProduct(choice->getCSRValues());
            stop_clock_and_add(t, "Computation Time"); 
        }
    }
};
