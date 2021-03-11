cmake_minimum_required(VERSION 3.14)
project(distributed_sddmmm)

# set(CMAKE_CXX_COMPILER "upcxx")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

find_package(MPI REQUIRED)
find_package(BLAS REQUIRED)

link_directories(patoh)
add_executable(distributed_sddmm main.cpp mm.cpp hypergraph_partition.cpp)
# -DSHMEM
# -DGASNET_EX
target_compile_options(distributed_sddmm PRIVATE -O3 -march=knl -DSHMEM -O3)
target_include_directories(distributed_sddmm PRIVATE bcl/ patoh)
target_link_libraries(distributed_sddmm PRIVATE MPI::MPI_CXX ${BLAS_LIBRARIES} patoh)

