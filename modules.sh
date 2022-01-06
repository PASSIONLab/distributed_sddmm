module load cmake/3.21.3

module swap PrgEnv-intel PrgEnv-gnu
module load cray-shmem
module load tbb
module load intel/18.0.1.163
module load cray-petsc-64/3.13.3.0 

# export MKLROOT=/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl
#module load upcxx
#export gasnet_prefix=$(upcxx-meta GASNET_INSTALL)
# module swap gcc/8.3.0 gcc/10.1.0
