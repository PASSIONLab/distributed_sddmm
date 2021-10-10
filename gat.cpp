#include <vector>
#include <utility>
#include <omp.h>
#include <iostream>
#include "sparse_kernels.h"
#include "SpmatLocal.hpp"
#include "common.h"

using namespace std;

// Slight variation of standard SDDMM used in a GAT 
/*class GATKernel : public StandardKernel {
public:
    size_t sddmm_local(
        SpmatLocal &S,
        DenseMatrix &A,
        DenseMatrix &B,
        int block,
        int offset) { 

        return 0.0;
    }
};*/

class GATLayer {
public:
    int input_features;
    int features_per_head;
    int num_heads;

    vector<DenseMatrix> wMats;
    VectorXd a1;
    VectorXd a2; 

    GATLayer(int input_features, int features_per_head, int num_heads) {
        this->input_features = input_features;
        this->features_per_head = features_per_head;
        this->num_heads = num_heads;
    }
};

/*
 * Multihead graph attention network.
 *
 * Work in Progress: While it does not affect timing performance,
 * implementation of the backward pass requires that the Wmat 
 * values be correctly synchronized. 
 */
class GAT {
public:

    Distributed_Sparse* d_ops;
    vector<GATLayer> layers;
    vector<DenseMatrix> buffers; 

    GAT(vector<GATLayer> &l_input, Distributed_Sparse* d_ops) {
        assert(l_input.size() > 0);
        
        this->d_ops = d_ops;
        layers = l_input;
        d_ops->setRValue(layers[0].input_features);
        buffers.push_back(d_ops->like_B_matrix(0.0));

        for(int i = 0; i < layers.size(); i++) {
            d_ops->setRValue(layers[i].features_per_head * layers[i].num_heads);
            buffers.push_back(d_ops->like_A_matrix(0.0));

            d_ops->setRValue(layers[i].features_per_head);

            for(int i = 0; i < layers[i].num_heads; i++) {
                layers[i].wMats.push_back(DenseMatrix::Constant(buffers[i].cols(), d_ops->localAcols));
            }
        }

    }

    // Computes the j'th self-attention head of the i'th layer
    void computeSelfAttentionHead(int i, int j) {        
        d_ops->setRValue(features_per_head);

        VectorXd Svalues = d_ops->like_S_values(1.0);
        VectorXd sddmm_buffer = d_ops->like_S_values(1.0);
        DenseMatrix A = buffers[i] * layers[i].wMats[j];
        DenseMatrix B = A; 
        d_ops->de_shift(B, nullptr, k_spmmA);
        algorithm(A, B, Svalues, &sddmm_buffer, k_sddmmA, true);
        A.setZero();
        // TODO: Normalize the rows, apply leaky ReLU function here! 
        algorithm(A, B, sddmm_buffer, nullptr, k_spmmA, false);
        buffers[i+1].middleCols(j * A.cols(), A.cols()) = A;

        // TODO: Sigmoid activation function here!
    }

    void forwardPass() {
        for(int i = 0; i < layers.size(); i++) {
            for(int j = 0; j < layers[i].num_heads; j++) {
                computeSelfAttentionHead(i, j);
            }
        }
    }
};

