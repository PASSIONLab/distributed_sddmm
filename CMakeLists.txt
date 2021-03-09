cmake_minimum_required(VERSION 3.14)
project(distributed_sddmmm)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

find_package(MPI REQUIRED)
find_package(BLAS REQUIRED)

add_executable(distributed_sddmm main.cpp mm.cpp dist_sddmm_kernel.cpp)
target_include_directories(distributed_sddmm PRIVATE bcl/)
target_link_libraries(distributed_sddmm PRIVATE MPI::MPI_CXX ${BLAS_LIBRARIES})

