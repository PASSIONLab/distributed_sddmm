# Distributed CPU SDDMM Benchmarks

## Dependencies
You will need:

- CMake version 3.14 or above
- GCC version 8.3.0 or above
- A clone of BCL that supports Cray-SHMEM (a public branch is available,
but it is not the main branch). Place the cloned BCL repo 
in a folder ``BCL" next to this README.

PaToH is included in this repo, so no need to configure or build.
This repo also includes scripts for reading / writing matrix
market files.

## Building
Follow these steps:

1. mkdir build
2. cd build
3. cmake ..
4. make

The output file is distributed_sddmm.

## Usage
Right now, this can only on a KNL processor (or another processor
that supports AVX512 instructions), since we call our own implementation
of the dot product to avoid function-call overhead. 
Benchmark the computation like this:

```
srun ./distributed_sddmm comp20k_A.mtx 128 parallel hypergraph
```

Specify the number of tasks / nodes you want in the srun command.

The first argument is the path to your matrix market file (can be 
symmetric or general). The second is the r-value (short matrix) 
dimension. The third parameter is deprecated (previously used to toggle
serial vs parallel, but no longer needed), always set it to be
``parallel". The fourth parameter specifies whether to reorder using
a hypergraph partitioner before executing the benchmark. If so, a call
to PaToH (included in the repo) is executed. Options for the fourth 
parameter are either "simple" or "hypergraph"; simple does not do
any reordering, hypergraph will perform reordering.

Testing...

