cmake_minimum_required(VERSION 3.14)
project(distributed_sddmmm)

# set(CMAKE_CXX_COMPILER "upcxx")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

find_package(MPI REQUIRED)
find_package(BLAS REQUIRED)

link_directories(patoh)
link_directories(metis/build/Linux-x86_64/libmetis)

add_executable(distributed_sddmm main.cpp mm.cpp hypergraph_partition.cpp graph_partition.cpp)

# Below 
#include(gasnet_cmake.cmak)
#set (GASNET_FLAGS ${GASNET_CXXCPPFLAGS} ${GASNET_CXXFLAGS})
#set (GASNET_BACKEND_LIB_FLAGS ${GASNET_LDFLAGS} ${GASNET_LIBS})
#set (BACKEND_FLAGS -DGASNET_EX ${GASNET_FLAGS} ${GASNET_BACKEND_LIB_FLAGS})

target_compile_options(distributed_sddmm PRIVATE -O3 -march=knl -O3 -DSHMEM) 
target_include_directories(distributed_sddmm PRIVATE bcl/ patoh metis/include)
target_link_libraries(distributed_sddmm PRIVATE MPI::MPI_CXX ${BLAS_LIBRARIES} patoh metis)

# message(${GASNET_BACKEND_LIB_FLAGS})
# target_link_options(distributed_sddmm PRIVATE ${GASNET_BACKEND_LIB_FLAGS}) 

