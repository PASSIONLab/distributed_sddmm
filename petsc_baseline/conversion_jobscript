#!/bin/bash
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -q regular 
#SBATCH -J conversion_jobscript
#SBATCH --mail-user=vivek_bharadwaj@berkeley.edu
#SBATCH --mail-type=ALL
#SBATCH -t 02:00:00

#OpenMP settings:
export OMP_NUM_THREADS=32
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

#run the application:
#srun -n 1 python ConvertMtxToPetsc.py $SCRATCH/dist_sddmm/amazon_large_randomized.mtx
#srun -n 1 python ConvertMtxToPetsc.py $SCRATCH/dist_sddmm/uk-2002-permuted.mtx
#srun -n 1 python ConvertMtxToPetsc.py $SCRATCH/dist_sddmm/eukarya-permuted.mtx
srun -n 1 python ConvertMtxToPetsc.py $SCRATCH/dist_sddmm/twitter7-permuted.mtx
