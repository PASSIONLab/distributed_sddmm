#!/bin/bash
#SBATCH -N 256
#SBATCH -C knl
#SBATCH -q regular
#SBATCH -J 25D_sparse_sddmm_testing
#SBATCH --mail-user=vivek_bharadwaj@berkeley.edu
#SBATCH --mail-type=ALL
#SBATCH -t 00:30:00

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

# Test increasing embedding sizes with several different replication
# factors

echo "Testing with a single MPI Rank Per Node..."
srun -N 256 -n 256 -c 272 --cpu_bind=cores ./25D_sparse 25 64 128 1
srun -N 256 -n 256 -c 272 --cpu_bind=cores ./25D_sparse 25 64 128 4
srun -N 256 -n 256 -c 272 --cpu_bind=cores ./25D_sparse 25 64 128 16

srun -N 256 -n 256 -c 272 --cpu_bind=cores ./25D_sparse 25 64 512 1
srun -N 256 -n 256 -c 272 --cpu_bind=cores ./25D_sparse 25 64 512 4
srun -N 256 -n 256 -c 272 --cpu_bind=cores ./25D_sparse 25 64 512 16
srun -N 256 -n 256 -c 272 --cpu_bind=cores ./25D_sparse 25 64 512 64

srun -N 256 -n 256 -c 272 --cpu_bind=cores ./25D_sparse 25 64 1024 1
srun -N 256 -n 256 -c 272 --cpu_bind=cores ./25D_sparse 25 64 1024 4
srun -N 256 -n 256 -c 272 --cpu_bind=cores ./25D_sparse 25 64 1024 16
srun -N 256 -n 256 -c 272 --cpu_bind=cores ./25D_sparse 25 64 1024 64
srun -N 256 -n 256 -c 272 --cpu_bind=cores ./25D_sparse 25 64 1024 128

# Test the impact of multiple MPI ranks on a single node
# (do multiple simultaneous broadcasts clog up the network?)
echo "Testing with 64 MPI ranks per node..."
srun -N 4 -n 256 -c 4 --cpu_bind=cores ./25D_sparse 25 64 512 1
srun -N 4 -n 256 -c 4 --cpu_bind=cores ./25D_sparse 25 64 512 4
srun -N 4 -n 256 -c 4 --cpu_bind=cores ./25D_sparse 25 64 512 8
srun -N 4 -n 256 -c 4 --cpu_bind=cores ./25D_sparse 25 64 512 16
srun -N 4 -n 256 -c 4 --cpu_bind=cores ./25D_sparse 25 64 512 64

echo "Testing with 32 MPI ranks per node..."
srun -N 8 -n 256 -c 8 --cpu_bind=cores ./25D_sparse 25 64 512 1
srun -N 8 -n 256 -c 8 --cpu_bind=cores ./25D_sparse 25 64 512 4
srun -N 8 -n 256 -c 8 --cpu_bind=cores ./25D_sparse 25 64 512 8
srun -N 8 -n 256 -c 8 --cpu_bind=cores ./25D_sparse 25 64 512 16
srun -N 8 -n 256 -c 8 --cpu_bind=cores ./25D_sparse 25 64 512 64

echo "Testing with 16 MPI ranks per node..."
srun -N 16 -n 256 -c 16 --cpu_bind=cores ./25D_sparse 25 64 512 1
srun -N 16 -n 256 -c 16 --cpu_bind=cores ./25D_sparse 25 64 512 4
srun -N 16 -n 256 -c 16 --cpu_bind=cores ./25D_sparse 25 64 512 8
srun -N 16 -n 256 -c 16 --cpu_bind=cores ./25D_sparse 25 64 512 16
srun -N 16 -n 256 -c 16 --cpu_bind=cores ./25D_sparse 25 64 512 64


echo "Testing with 8 MPI ranks per node..."
srun -N 32 -n 256 -c 32 --cpu_bind=cores ./25D_sparse 25 64 512 1
srun -N 32 -n 256 -c 32 --cpu_bind=cores ./25D_sparse 25 64 512 4
srun -N 32 -n 256 -c 32 --cpu_bind=cores ./25D_sparse 25 64 512 8
srun -N 32 -n 256 -c 32 --cpu_bind=cores ./25D_sparse 25 64 512 16
srun -N 32 -n 256 -c 32 --cpu_bind=cores ./25D_sparse 25 64 512 64

echo "Testing with 4 MPI ranks per node..."
srun -N 64 -n 256 -c 64 --cpu_bind=cores ./25D_sparse 25 64 512 1
srun -N 64 -n 256 -c 64 --cpu_bind=cores ./25D_sparse 25 64 512 4
srun -N 64 -n 256 -c 64 --cpu_bind=cores ./25D_sparse 25 64 512 8
srun -N 64 -n 256 -c 64 --cpu_bind=cores ./25D_sparse 25 64 512 16
srun -N 64 -n 256 -c 64 --cpu_bind=cores ./25D_sparse 25 64 512 64

echo "Testing with 2 MPI ranks per node..."
srun -N 128 -n 256 -c 128 --cpu_bind=cores ./25D_sparse 25 64 512 1
srun -N 128 -n 256 -c 128 --cpu_bind=cores ./25D_sparse 25 64 512 4
srun -N 128 -n 256 -c 128 --cpu_bind=cores ./25D_sparse 25 64 512 8
srun -N 128 -n 256 -c 128 --cpu_bind=cores ./25D_sparse 25 64 512 16
srun -N 128 -n 256 -c 128 --cpu_bind=cores ./25D_sparse 25 64 512 64

