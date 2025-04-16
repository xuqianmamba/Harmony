include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(SRC_FILES
    Clustering.cpp
    Index.cpp
    IVF.cpp
    node.cpp
    # utils.cpp
)

add_library(Tribase STATIC ${SRC_FILES})

target_link_libraries(Tribase PUBLIC OpenMP::OpenMP_CXX
                                    mkl_intel_lp64
                                    mkl_sequential
                                    mkl_core
                                    MPI::MPI_CXX)

add_executable(main main.cpp)
target_link_libraries(main Tribase faiss)

add_executable(query query.cpp)
target_link_libraries(query Tribase faiss)

# add_subdirectory(tests)

# OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(Tribase PUBLIC OpenMP::OpenMP_CXX)
endif()

# MPI
find_package(MPI REQUIRED)

target_link_libraries(Tribase PUBLIC MPI::MPI_CXX)