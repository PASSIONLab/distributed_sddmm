cmake_minimum_required(VERSION 3.14)

project(distributed_sddmmm)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

find_package(MPI REQUIRED)
find_package(BLAS REQUIRED)
find_package(OpenMP REQUIRED)
# find_package(MKL REQUIRED)

# These link to the copy of CombBLAS in DiBELLA
include_directories(
	/global/homes/v/vbharadw/diBELLA.2D/CombBLAS/install/include)

# Link to Eigen
include_directories(
    /global/homes/v/vbharadw/distributed_sddmm/eigen
)

link_directories(
    /global/homes/v/vbharadw/diBELLA.2D/CombBLAS/install/lib $ENV{MKLROOT}/lib/intel64 
    )

add_executable(main main.cpp benchmark_dist.cpp sparse_kernels.cpp common.cpp) 
add_executable(local_kernel_benchmark local_kernel_benchmark.cpp common.cpp)

foreach(TARGET test_harness local_kernel_benchmark)
    target_compile_options(${TARGET} PRIVATE -O3 -march=knl -DMKL_ILP64  -m64  -I$ENV{MKLROOT}/include) 
    target_link_options(${TARGET} PRIVATE -L$ENV{MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl)
    target_link_libraries(${TARGET} PRIVATE MPI::MPI_CXX ${BLAS_LIBRARIES} OpenMP::OpenMP_CXX CombBLAS GraphGenlib )
endforeach()
