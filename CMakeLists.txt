cmake_minimum_required(VERSION 3.14)
project(distributed_sddmmm)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

find_package(MPI REQUIRED)
find_package(BLAS REQUIRED)

add_executable(distributed_sddmm main.cpp mm.cpp spmat_reader.cpp sddmm.cpp 2D.cpp pack.cpp) 

target_compile_options(distributed_sddmm PRIVATE -O3 -march=knl -O3) 
target_link_libraries(distributed_sddmm PRIVATE MPI::MPI_CXX ${BLAS_LIBRARIES})
