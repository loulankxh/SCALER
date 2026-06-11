cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

project(
    ScaleGANN
    LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")

# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fopenmp -Xcompiler=-mavx2 -Xcompiler=-mfma -Xcompiler=-msse4.2")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma -msse2 -ftree-vectorize -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free -fopenmp -fopenmp-simd -funroll-loops -Wfatal-errors -DUSE_AVX2")

find_package(Boost REQUIRED COMPONENTS program_options)
find_package(fmt REQUIRED)
find_package(benchmark REQUIRED)

# Try to find MKL
set(MKL_ROOT $ENV{CONDA_PREFIX} CACHE PATH "MKL Root")
find_library(MKL_RT_LIB NAMES mkl_rt HINTS ${MKL_ROOT}/lib)

include_directories(
    ${PROJECT_SOURCE_DIR}/DiskANN/include
    ${PROJECT_SOURCE_DIR}/src
    ${Boost_INCLUDE_DIRS}
)

# RAFT/CAGRA include paths (usually in conda include if installed via rapidsai)
if(DEFINED ENV{CONDA_PREFIX})
    include_directories($ENV{CONDA_PREFIX}/include)
    include_directories($ENV{CONDA_PREFIX}/include/rapids)
    include_directories($ENV{CONDA_PREFIX}/include/rapids/libcudacxx)
endif()

set(DISKANN_SRC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/DiskANN/src)
set(DISKANN_APP_PATH ${CMAKE_CURRENT_SOURCE_DIR}/DiskANN/apps)
set(SCALEGANN_SRC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(SCALEGANN_APP_PATH ${CMAKE_CURRENT_SOURCE_DIR}/apps)


set(DISKANN_SOURCES
    abstract_data_store.cpp ann_exception.cpp disk_utils.cpp 
    distance.cpp index.cpp in_mem_graph_store.cpp in_mem_data_store.cpp
    linux_aligned_file_reader.cpp math_utils.cpp natural_number_map.cpp
    in_mem_data_store.cpp in_mem_graph_store.cpp
    natural_number_set.cpp memory_mapper.cpp partition.cpp pq.cpp
    pq_flash_index.cpp scratch.cpp logger.cpp utils.cpp filter_utils.cpp index_factory.cpp abstract_index.cpp pq_l2_distance.cpp pq_data_store.cpp)
list(TRANSFORM DISKANN_SOURCES PREPEND ${DISKANN_SRC_PATH}/)
set(DISKANN_PARTITION
    diskann/partition.cpp)
list(TRANSFORM DISKANN_PARTITION  PREPEND ${SCALEGANN_APP_PATH}/)
set(DISKANN_BUILD
    diskann/build.cpp)
list(TRANSFORM DISKANN_BUILD  PREPEND ${SCALEGANN_APP_PATH}/)
set(DISKANN_MERGE
    diskann/merge.cpp)
list(TRANSFORM DISKANN_MERGE  PREPEND ${SCALEGANN_APP_PATH}/)
set(DISKANN_COMPUTE_GT
    utils/compute_groundtruth.cpp)
list(TRANSFORM DISKANN_COMPUTE_GT  PREPEND ${DISKANN_APP_PATH}/)
set(DISKANN_MEMORY_INDEX
    build_memory_index.cpp)
list(TRANSFORM DISKANN_MEMORY_INDEX  PREPEND ${DISKANN_APP_PATH}/)



set(DISK_PARTITION_SOURCES
    partition/partition.cpp partition/disk_partition.cpp partition/kmeans.cpp partition/kmeans.cu
    partition/kmeans_cpu.cpp
    utils/indexIO.cpp utils/datasetIO.cpp utils/distance.cpp)
list(TRANSFORM DISK_PARTITION_SOURCES  PREPEND ${SCALEGANN_SRC_PATH}/)
set(DISK_PARTITION_APP
    executeDiskPartition.cpp)
list(TRANSFORM DISK_PARTITION_APP  PREPEND ${SCALEGANN_APP_PATH}/)
set(DISK_DIRECT_PARTITION_APP
    executeDiskDirectPartition.cpp)
list(TRANSFORM DISK_DIRECT_PARTITION_APP  PREPEND ${SCALEGANN_APP_PATH}/)

set(DISK_MERGE_SOURCES 
    merge/disk_merge.cpp
    utils/indexIO.cpp utils/datasetIO.cpp)
list(TRANSFORM DISK_MERGE_SOURCES  PREPEND ${SCALEGANN_SRC_PATH}/)
set(DISK_MERGE_APP
    executeDiskMerge.cpp)
list(TRANSFORM DISK_MERGE_APP  PREPEND ${SCALEGANN_APP_PATH}/)
set(DISK_PIPELINED_MERGE_APP
    executeDiskPipelinedMerge.cpp)
list(TRANSFORM DISK_PIPELINED_MERGE_APP  PREPEND ${SCALEGANN_APP_PATH}/)

# set(MERGE_SOURCES 
#     merge/merge.cpp merge/merge.cu
#     utils/indexIO.cpp utils/datasetIO.cpp utils/distance.cpp
#     taskScheduler/executeMemoryMerge.cpp taskScheduler/gpuManagement.cpp)
# list(TRANSFORM MERGE_SOURCES  PREPEND ${SCALEGANN_SRC_PATH}/)


set(SEARCH_SOURCES 
    search/search.cpp search/disk_search.cpp
    utils/indexIO.cpp utils/datasetIO.cpp utils/distance.cpp)
list(TRANSFORM SEARCH_SOURCES  PREPEND ${SCALEGANN_SRC_PATH}/)
set(SEARCH_SCALEGANN_APP
    searchScaleGANN.cpp)
list(TRANSFORM SEARCH_SCALEGANN_APP  PREPEND ${SCALEGANN_APP_PATH}/)
set(SEARCH_CAGRA_APP
    searchCAGRA.cpp)
list(TRANSFORM SEARCH_CAGRA_APP  PREPEND ${SCALEGANN_APP_PATH}/)
set(SEARCH_GGNN_APP
    searchGGNN.cpp)
list(TRANSFORM SEARCH_GGNN_APP  PREPEND ${SCALEGANN_APP_PATH}/)
set(SEARCH_DISKANN_APP
    searchDiskANN.cpp)
list(TRANSFORM SEARCH_DISKANN_APP  PREPEND ${SCALEGANN_APP_PATH}/)




set(COMMON_LIBS
    cudart
    dl
    benchmark::benchmark
    pthread
    fmt::fmt
    ${MKL_RT_LIB}
    m
    Boost::program_options
)

add_executable(diskannPartition ${DISKANN_PARTITION} ${DISK_PARTITION_SOURCES} ${DISKANN_SOURCES})
add_executable(diskannBuild ${DISKANN_BUILD} ${DISKANN_SOURCES})
add_executable(diskannMerge ${DISKANN_MERGE} ${DISK_MERGE_SOURCES} ${DISKANN_SOURCES})
add_executable(diskannComputeGT ${DISKANN_COMPUTE_GT} ${DISKANN_SOURCES})
add_executable(diskannMemoryIndex ${DISKANN_MEMORY_INDEX} ${DISKANN_SOURCES})

add_executable(executeDiskPartition ${DISK_PARTITION_APP} ${DISK_PARTITION_SOURCES} ${DISKANN_SOURCES})
add_executable(executeDiskDirectPartition ${DISK_DIRECT_PARTITION_APP} ${DISK_PARTITION_SOURCES} ${DISKANN_SOURCES})
set(SCHEDULER_SOURCES
    taskScheduler/TaskBroker.cpp taskScheduler/gpuManagement.cpp)
list(TRANSFORM SCHEDULER_SOURCES PREPEND ${SCALEGANN_SRC_PATH}/)

add_library(scheduler STATIC ${SCHEDULER_SOURCES})
target_link_libraries(scheduler PRIVATE ${COMMON_LIBS})

# ... existing code ...

add_executable(executeDiskMerge ${DISK_MERGE_APP} ${DISK_MERGE_SOURCES} ${DISKANN_SOURCES})
add_executable(executeDiskPipelinedMerge ${DISK_PIPELINED_MERGE_APP} ${DISK_MERGE_SOURCES} ${DISKANN_SOURCES})
# add_executable(executeMerge ${MERGE_SOURCES} ${DISKANN_SOURCES})

target_link_libraries(executeDiskMerge PRIVATE scheduler)
add_executable(searchScaleGANN ${SEARCH_SCALEGANN_APP} ${SEARCH_SOURCES})
add_executable(searchCAGRA ${SEARCH_CAGRA_APP} ${SEARCH_SOURCES})
add_executable(searchGGNN ${SEARCH_GGNN_APP} ${SEARCH_SOURCES})
add_executable(searchDiskANN ${SEARCH_DISKANN_APP} ${SEARCH_SOURCES})

add_executable(testTaskBroker tests/test_TaskBroker.cpp)
target_link_libraries(testTaskBroker PRIVATE scheduler)

add_executable(testSimulation tests/test_Simulation.cpp)
target_link_libraries(testSimulation PRIVATE scheduler)

set(MEMORY_MERGE_SOURCES 
    merge/merge.cpp merge/merge.cu
    utils/indexIO.cpp utils/datasetIO.cpp utils/distance.cpp)
list(TRANSFORM MEMORY_MERGE_SOURCES  PREPEND ${SCALEGANN_SRC_PATH}/)

add_executable(executeMemoryMerge apps/executeMemoryMerge.cpp ${MEMORY_MERGE_SOURCES} ${DISKANN_SOURCES})
target_link_libraries(executeMemoryMerge PRIVATE scheduler)

set(ALL_TARGETS 
    diskannPartition diskannBuild diskannMerge diskannComputeGT diskannMemoryIndex
    executeDiskPartition executeDiskDirectPartition executeDiskMerge executeMemoryMerge
    searchScaleGANN searchCAGRA searchGGNN searchDiskANN
    testTaskBroker testSimulation
)

target_link_libraries(executeDiskPartition
    PRIVATE
    cudart
    dl
    benchmark
    pthread
    fmt
    mkl_rt
    m
    Boost::program_options
)
target_link_libraries(executeDiskDirectPartition
    PRIVATE
    cudart
    dl
    benchmark
    pthread
    fmt
    mkl_rt
    m
    Boost::program_options
)
target_link_libraries(executeDiskMerge
    PRIVATE
    cudart
    dl
    benchmark
    pthread
    fmt
    mkl_rt
    m
    Boost::program_options
)
target_link_libraries(executeDiskPipelinedMerge
    PRIVATE
    cudart
    dl
    benchmark
    pthread
    fmt
    mkl_rt
    m
    Boost::program_options
)
# target_link_libraries(executeMerge
#     PRIVATE
#     cudart
#     dl
#     benchmark
#     pthread
#     fmt
#     mkl_rt
#     m
# )
target_link_libraries(searchScaleGANN
    PRIVATE
    cudart
    dl
    benchmark
    pthread
    fmt
    mkl_rt
    m
    Boost::program_options
)
target_link_libraries(searchCAGRA
    PRIVATE
    cudart
    dl
    benchmark
    pthread
    fmt
    mkl_rt
    m
    Boost::program_options
)
target_link_libraries(searchGGNN
    PRIVATE
    cudart
    dl
    benchmark
    pthread
    fmt
    mkl_rt
    m
    Boost::program_options
)
target_link_libraries(searchDiskANN
    PRIVATE
    cudart
    dl
    benchmark
    pthread
    fmt
    mkl_rt
    m
    Boost::program_options
)
# set(COMMON_LIBS
#     cudart
#     dl
#     benchmark
#     pthread
#     fmt
#     mkl_rt
#     m
# )
# foreach(target IN LISTS executeDiskPartition executeMerge searchScaleGANN)
#     target_link_libraries(${target} PRIVATE ${COMMON_LIBS})
# endforeach()
