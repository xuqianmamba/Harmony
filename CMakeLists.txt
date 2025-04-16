cmake_minimum_required(VERSION 3.21)
project(TriBase)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_C_COMPILER /es01/software/spack_external/compiler/gcc/13.3/bin/gcc)
set(CMAKE_CXX_COMPILER /es01/software/spack_external/compiler/gcc/13.3/bin/g++)
#set(CMAKE_C_COMPILER /es01/home/lvxg/gcc-12.2.0/bin/gcc)
#set(CMAKE_CXX_COMPILER /es01/home/lvxg/gcc-12.2.0/bin/g++)
# set(CMAKE_ASM_COMPILER "$HOME/binutils-2.40/bin/as")

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Standard)
elseif (NOT CMAKE_BUILD_TYPE MATCHES "Debug|Release|Standard")
    message(FATAL_ERROR "Invalid build type: ${CMAKE_BUILD_TYPE}")
endif()
#link_directories("/es01/home/lvxg/vdb/gcc-12.1.0/lib64")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib) 
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

find_package(OpenMP REQUIRED)
# find_package(GTest REQUIRED)
find_package(MPI REQUIRED)

# Print the value of OpenMP_CXX_FLAGS
message(STATUS "OpenMP_CXX_FLAGS = ${OpenMP_CXX_FLAGS}")

set(MKL_ROOT "/es01/software/spack_external/compiler/intel/2024.2/intel/mkl/2024.2")
include_directories(${MKL_ROOT}/include)
link_directories(${MKL_ROOT}/lib/intel64)

option(ENABLE_PROFILING "Enable profiling" OFF)

if (ENABLE_PROFILING)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
endif()

# set(W_CXX_FLAGS " -Wall -Wextra")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${W_CXX_FLAGS} -fopenmp ${OpenMP_CXX_FLAGS} -mavx2")
elseif (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "Standard")
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${W_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -Ofast -mavx512f -march=native -mtune=native -DNDEBUG")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${W_CXX_FLAGS} -fopenmp ${OpenMP_CXX_FLAGS} -Ofast -mavx2 -march=native -mtune=native")
# elseif (CMAKE_BUILD_TYPE STREQUAL "MPI")
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${W_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -Ofast -mavx512f -march=native -mtune=native")
#     set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
endif()



if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_definitions(NSTATS)
endif()
include_directories(/es01/home/lvxg/vdb/VectorDB/eigen3/Eigen)
# include_directories(/es01/home/lvxg/vdb/VectorDB)
# include_directories(/es01/home/lvxg/vdb/gcc-12.1.0/include)
include_directories(${CMAKE_SOURCE_DIR})
# target_include_directories(TriBase PRIVATE /es01/home/lvxg/vdb/VectorDB/eigen-3.4.0/Eigen)
include_directories(${CMAKE_SOURCE_DIR}/third_party)
include_directories(${CMAKE_SOURCE_DIR}/third_party/faiss)
include_directories(${CMAKE_SOURCE_DIR}/hnswlib)
add_subdirectory(third_party/faiss)
add_subdirectory(src)