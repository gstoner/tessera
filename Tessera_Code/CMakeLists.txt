cmake_minimum_required(VERSION 3.20)
project(Tessera LANGUAGES C CXX)

option(TESSERA_ENABLE_CUDA "Build CUDA backend" OFF)
option(TESSERA_ENABLE_ROCM "Build ROCm backend" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(tessera_runtime
    runtime/src/api.cpp
    runtime/src/device.cpp
    runtime/src/memory.cpp
    runtime/src/stream.cpp
    runtime/src/logging.cpp
    runtime/src/collectives/nccl_stub.cpp
    runtime/src/collectives/nvshmem_stub.cpp
    runtime/src/backends/cuda_backend.cpp
    runtime/src/backends/rocm_backend.cpp
)
target_include_directories(tessera_runtime PUBLIC runtime/include)
target_compile_definitions(tessera_runtime PRIVATE
    $<$<BOOL:${TESSERA_ENABLE_CUDA}>:TESSERA_ENABLE_CUDA>
    $<$<BOOL:${TESSERA_ENABLE_ROCM}>:TESSERA_ENABLE_ROCM>
)
