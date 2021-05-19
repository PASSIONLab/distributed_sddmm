#!/bin/bash
#SBATCH -N 256
#SBATCH -C knl
#SBATCH -q regular
#SBATCH -J 25D_sparse_sddmm_testing
#SBATCH --mail-user=vivek_bharadwaj@berkeley.edu
#SBATCH --mail-type=ALL
#SBATCH -t 02:00:00

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

# Test increasing embedding sizes with several different replication
# factors

LOGM=24
NNZPERROW=32

echo "Testing with a single MPI Rank Per Node..."
for c in 1 4 16 64
do
    srun -N 256 -n 256 -c 272 --cpu_bind=cores ./25D_sparse $LOGM $NNZPERROW 512 $c
done

# Test the impact of multiple MPI ranks on a single node
# (do multiple simultaneous broadcasts clog up the network?)
echo "Testing with 64 MPI ranks per node..."
for c in 1 4 16 64
do
    srun -N 4 -n 256 -c 4 --cpu_bind=cores ./25D_sparse $LOGM $NNZPERROW 512 $c
done

echo "Testing with 32 MPI ranks per node..."
for c in 1 4 16 64
do
    srun -N 8 -n 256 -c 8 --cpu_bind=cores ./25D_sparse $LOGM $NNZPERROW 512 $c
done

echo "Testing with 16 MPI ranks per node..."
for c in 1 4 16 64
do
    srun -N 16 -n 256 -c 16 --cpu_bind=cores ./25D_sparse $LOGM $NNZPERROW 512 $c
done

echo "Testing with 8 MPI ranks per node..."
for c in 1 4 16 64
do
    srun -N 32 -n 256 -c 32 --cpu_bind=cores ./25D_sparse $LOGM $NNZPERROW 512 $c
done

echo "Testing with 4 MPI ranks per node..."
for c in 1 4 16 64
do
    srun -N 64 -n 256 -c 64 --cpu_bind=cores ./25D_sparse $LOGM $NNZPERROW 512 $c
done

echo "Testing with 2 MPI ranks per node..."
for c in 1 4 16 64
do
    srun -N 128 -n 256 -c 128 --cpu_bind=cores ./25D_sparse $LOGM $NNZPERROW 512 $c
done

