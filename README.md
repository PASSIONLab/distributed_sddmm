# Half-and-Half  (HnH)
__Half-and-Half (HnH)__ is a C++ library for extremely 
large parallel distributed sparse-times-dense matrix multiplication (SpMM) 
and sampled-dense-dense matrix multiplication (SDDMM) on computing clusters. 
It relies on MPI for inter-process communication and OpenMP for intra-node parallelism.
HnH uses one-and-a-__half__ dimensional
(1.5D) and two-and-a-__half__ dimensional (2.5D) sparse-times-dense algorithms
to reduce communication bandwidth, especially when dense matrix inputs are tall-skinny. 
It provides a simple, unified interface for 1.5D
dense shifting, 1.5D sparse shifting, 2.5D dense shifting, and 2.5D sparse shifting
SDDMM / SpMM algorithms that hides implementation details from users.

When executing an SDDMM followed by an SpMM operation, HnH can save even more communication by
using one of two distinct strategies: replication reuse (every algorithm) or kernel
overlap (only 1.5D dense shifting algorithms).

HnH can use any replacement for its local SpMM and SDDMM kernels that you provide,
allowing it to generalize beyond the standard definitions of SDDMM and SpMM.

## Citation info

The algorithms implemented in this repository are described in the following publication

Vivek Bharadwaj, Aydin Buluç, James Demmel. Distributed-Memory Sparse Kernels for Machine Learning. In Proceedings of 36th IEEE International Parallel & Distributed Processing Symposium, 2022.

Preprint available at https://arxiv.org/abs/2203.07673

## How do I use it?
Here are the steps:

1. Load a sparse matrix.
2. Select a local kernel implementation and an algorithm. 
3. Retrieve the input buffers adapted to the algorithm and fill them. 
4. Execute an SDDMM, SpMM, or both on the input buffers. 

Here's a demo: 
```c++

// 1. Load a sparse matrix from the given filename 
//    (matrix format format) 
string fname(argv[1]);
SpmatLocal S;
S.loadTuples(true, -1, -1, fname);

// 2. Use the standard definition of SDDMM / SpMM with
//    a 1.5D sparse shifting algorithm  
StandardKernel local_ops;
Sparse15D_Sparse_Shift* d_ops =
    new Sparse15D_Sparse_Shift(&S,
        atoi(argv[2]), 
        atoi(argv[3]), 
        &local_ops);

// 3. Retrieve and fill IO buffers 

// 4. Execute an SDDMM
d_ops->sddmmA(A, B, S, result);
```
The result of the SDDMM computation is stored in ``result".

## Who is this for?
HnH is useful when the main computation in your application is an SDDMM / SpMM.
Use it when the input matrices in your problem exceed the memory capacity of a single
node, or you want to reduce runtime on a parallel cluster. 

## External Dependencies 
HnH relies on:
- CMake >= 3.14
- GCC >= 8.3.0: It has not been tested yet with the Intel C++ Compiler.
- MPI
- OpenMP
- Intel MKL >= 2018
- CombBLAS: The Combinatorial BLAS, for sparse matrix IO and random sparse matrix generation. 
- Eigen: For local dense matrix algebra

The first five dependencies are your responsibility, and CMake should locate them
automatically. Run `. install_dependencies.sh` script to download Eigen and build
CombBLAS. If you are running on Cori, run `. modules.sh` to load the modules with 
the correct dependencies and set the programming environment correctly. 

## Included Dependencies
HnH includes Niels Lohmann's JSON C++ library header to neatly print out statistics
when benchmarking. 

## Building
Follow these steps in the repository root:

1. `mkdir build`
2. `cd build`
3. `cmake ..`
4. `make -j4`

Link your code to the resulting output library.

