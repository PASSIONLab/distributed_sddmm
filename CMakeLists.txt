cmake_minimum_required(VERSION 3.14)
# set (CMAKE_CXX_COMPILER tau_cxx.sh)
project(distributed_sddmmm)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

find_package(MPI REQUIRED)
find_package(BLAS REQUIRED)
find_package(OpenMP REQUIRED)

# These link to the copy of CombBLAS in DiBELLA
include_directories(
	/global/homes/v/vbharadw/diBELLA.2D/CombBLAS/install/include)

link_directories(
    /global/homes/v/vbharadw/diBELLA.2D/CombBLAS/install/lib)

add_executable(distributed_sddmm main.cpp mm.cpp spmat_reader.cpp sddmm.cpp 15D_sparse.cpp erdos_gen.cpp pack.cpp) 

target_compile_options(distributed_sddmm PRIVATE -O3 -march=knl -O3) 
target_link_libraries(distributed_sddmm PRIVATE MPI::MPI_CXX ${BLAS_LIBRARIES} OpenMP::OpenMP_CXX CombBLAS)
