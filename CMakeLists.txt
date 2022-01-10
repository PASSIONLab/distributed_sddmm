cmake_minimum_required(VERSION 3.16)

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
    /global/homes/v/vbharadw/diBELLA.2D/CombBLAS/install/lib $ENV{MKLROOT}/lib/intel64 
    )

add_executable(bench_er bench_erdos_renyi.cpp benchmark_dist.cpp sparse_kernels.cpp common.cpp als_conjugate_gradients.cpp) 
add_executable(bench_file bench_file.cpp benchmark_dist.cpp sparse_kernels.cpp common.cpp als_conjugate_gradients.cpp) 
add_executable(local_kernel_benchmark sparse_kernels.cpp local_kernel_benchmark.cpp common.cpp)
add_executable(scratch scratch.cpp sparse_kernels.cpp common.cpp als_conjugate_gradients.cpp)
add_executable(petsc_bench petsc_baseline/spmm_test.cpp) 
add_executable(random_permute random_permute.cpp) 

#add_executable(test_async test_async_strategies.cpp common.cpp) 
#add_executable(par_iotest ParIOTest.cpp) 

foreach(TARGET bench_file bench_er local_kernel_benchmark scratch)
    target_compile_options(${TARGET} PRIVATE -O3 -march=knl -DMKL_ILP64  -m64  -I$ENV{MKLROOT}/include) 
    target_link_options(${TARGET} PRIVATE -L$ENV{MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl)
    target_link_libraries(${TARGET} PRIVATE MPI::MPI_CXX ${BLAS_LIBRARIES} OpenMP::OpenMP_CXX CombBLAS GraphGenlib )
    target_precompile_headers(${TARGET} PRIVATE json.hpp <Eigen/Dense>)
endforeach()


foreach(TARGET random_permute)
    target_compile_options(${TARGET} PRIVATE -O3 -march=haswell -DMKL_ILP64  -m64  -I$ENV{MKLROOT}/include) 
    target_link_options(${TARGET} PRIVATE -L$ENV{MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl)
    target_link_libraries(${TARGET} PRIVATE MPI::MPI_CXX ${BLAS_LIBRARIES} OpenMP::OpenMP_CXX CombBLAS GraphGenlib )
    target_precompile_headers(${TARGET} PRIVATE json.hpp <Eigen/Dense>)
endforeach()


#foreach(TARGET par_iotest)
#    target_compile_options(${TARGET} PRIVATE -O3 -march=haswell -DMKL_ILP64  -m64  -I$ENV{MKLROOT}/include) 
#    target_link_options(${TARGET} PRIVATE -L$ENV{MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl)
#    target_link_libraries(${TARGET} PRIVATE MPI::MPI_CXX ${BLAS_LIBRARIES} OpenMP::OpenMP_CXX CombBLAS GraphGenlib )
#    target_precompile_headers(${TARGET} PRIVATE json.hpp <Eigen/Dense>)
#endforeach()

