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

# Link to Eigen
include_directories(
    /global/homes/v/vbharadw/distributed_sddmm/eigen
)

link_directories(
    /global/homes/v/vbharadw/diBELLA.2D/CombBLAS/install/lib)

add_executable(test_harness test_harness.cpp sparse_kernels.cpp common.cpp als_conjugate_gradients.cpp) 

foreach(TARGET test_harness)
    target_compile_options(${TARGET} PRIVATE -O3 -march=knl -O3) 
    target_link_libraries(${TARGET} PRIVATE MPI::MPI_CXX ${BLAS_LIBRARIES} OpenMP::OpenMP_CXX CombBLAS GraphGenlib)
endforeach()
