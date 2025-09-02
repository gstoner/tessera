cmake_minimum_required(VERSION 3.20)
project(tessera_gpu_backend LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(TESSERA_BUILD_TESTS "Build GPU backend tests" ON)
option(TESSERA_USE_NVRTC "Build with NVRTC JIT support" ON)

# CUDA arch (user can override, default sm_90)
if (NOT DEFINED TESSERA_CUDA_ARCH)
  set(TESSERA_CUDA_ARCH "sm_90")
endif()
message(STATUS "TESSERA_CUDA_ARCH = ${TESSERA_CUDA_ARCH}")
add_compile_definitions(TESSERA_CUDA_ARCH_STR="${TESSERA_CUDA_ARCH}")

# Common CUDA flags
set(CMAKE_CUDA_ARCHITECTURES OFF) # we'll pass -arch manually for portability
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fPIC>)

# Library
add_library(tessera_gpu_backend
  src/backend_gpu.cpp
  src/runtime/cuda_driver.cpp
  src/kernels/wmma_gemm_fp16.cu
  src/kernels/wmma_gemm_bf16.cu
  src/kernels/wmma_bf16_pipeline.cu
)

target_include_directories(tessera_gpu_backend PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_compile_options(tessera_gpu_backend PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-arch=${TESSERA_CUDA_ARCH}>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
)

if (TESSERA_USE_NVRTC)
  target_sources(tessera_gpu_backend PRIVATE src/runtime/nvrtc_jit.cpp)
endif()

# Tests
if (TESSERA_BUILD_TESTS)
  add_executable(test_wmma tests/test_wmma_gemm.cu)
  target_link_libraries(test_wmma PRIVATE tessera_gpu_backend)
  target_compile_options(test_wmma PRIVATE -arch=${TESSERA_CUDA_ARCH})
endif()


option(TESSERA_ENABLE_NVTILE "Enable NVIDIA Tile IR (experimental) lowering skeleton and PTX WGMMA example" ON)

if (TESSERA_ENABLE_NVTILE)
  target_sources(tessera_gpu_backend PRIVATE
    src/lowering_nvtile_mlir.cpp
    src/kernels/ptx_wgmma_bf16.cu
  )
endif()


# WGMMA + IMMA samples
target_sources(tessera_gpu_backend PRIVATE
  src/kernels/wgmma_bf16_inline_ptx.cu
  src/kernels/imma_gemm_int8.cu
)

option(TESSERA_BUILD_GPU_BENCH "Build GPU micro-bench" ON)
if (TESSERA_BUILD_GPU_BENCH)
  add_executable(bench_gpu bench/bench_gpu.cu)
  target_link_libraries(bench_gpu PRIVATE tessera_gpu_backend)
  target_compile_options(bench_gpu PRIVATE -arch=${TESSERA_CUDA_ARCH})
endif()
