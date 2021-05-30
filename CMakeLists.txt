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

add_executable(15D_sparse 15D_sparse.cpp sddmm.cpp) 
add_executable(25D_sparse 25D_sparse.cpp sddmm.cpp) 
add_executable(serial_conjgrad serial_conjgrad.cpp) 

foreach(TARGET 15D_sparse 25D_sparse serial_conjgrad)
    target_compile_options(${TARGET} PRIVATE -O3 -march=knl -O3) 
    target_link_libraries(${TARGET} PRIVATE MPI::MPI_CXX ${BLAS_LIBRARIES} OpenMP::OpenMP_CXX CombBLAS GraphGenlib)
endforeach()
