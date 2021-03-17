module load cmake
module load openmpi

module swap PrgEnv-intel PrgEnv-gnu
module load cray-shmem
module load tbb
export MKLROOT=/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl

module load upcxx
export gasnet_prefix=$(upcxx-meta GASNET_INSTALL)

# module swap gcc/8.3.0 gcc/10.1.0
