cmake_minimum_required(VERSION 3.20)

# Tessera NVIDIA backend.
#
# TESSERA_BUILD_NVIDIA_BACKEND=ON always builds the hardware-free MLIR Target
# IR contract spine. CUDA runtime kernels are compiled only when the top-level
# TESSERA_ENABLE_CUDA option enabled CUDA language support and found the toolkit.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(TESSERA_NVIDIA_BUILD_TOOLS "Build tessera-nvidia-opt" ON)
option(TESSERA_BUILD_GPU_BENCH "Build NVIDIA GPU micro-bench" ON)
option(TESSERA_ENABLE_NVTILE "Enable NVIDIA Tile IR runtime samples" ON)
option(TESSERA_USE_NVRTC "Build NVIDIA runtime with NVRTC JIT support" ON)

if(NOT DEFINED TESSERA_CUDA_ARCH)
  set(TESSERA_CUDA_ARCH "sm_90a")
endif()
message(STATUS "TESSERA_CUDA_ARCH = ${TESSERA_CUDA_ARCH}")

add_subdirectory(lib/IR)
add_subdirectory(lib/Conversion)

# Shipped NVRTC-compiled mma.sync GEMM runtime symbols (self-gates on
# TESSERA_ENABLE_CUDA). The hardware_verified backend_kernel proof lib.
add_subdirectory(runtime/cuda)

if(TESSERA_NVIDIA_BUILD_TOOLS)
  add_subdirectory(tools)
endif()

if(TESSERA_BUILD_TESTS)
  add_subdirectory(test)
endif()

if(TESSERA_ENABLE_CUDA)
  set(CMAKE_CUDA_ARCHITECTURES OFF)

  add_library(tessera_gpu_backend
    src/backend_gpu.cpp
    src/runtime/cuda_driver.cpp
    src/kernels/wmma_gemm_fp16.cu
    src/kernels/wmma_gemm_bf16.cu
    src/kernels/wmma_bf16_pipeline.cu
    src/kernels/wgmma_bf16_inline_ptx.cu
    src/kernels/imma_gemm_int8.cu
  )

  target_include_directories(tessera_gpu_backend PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_compile_definitions(tessera_gpu_backend PUBLIC
    TESSERA_CUDA_ARCH_STR="${TESSERA_CUDA_ARCH}"
  )
  target_compile_options(tessera_gpu_backend PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-arch=${TESSERA_CUDA_ARCH}>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fPIC>
  )

  if(TESSERA_USE_NVRTC)
    target_sources(tessera_gpu_backend PRIVATE src/runtime/nvrtc_jit.cpp)
  endif()

  if(TESSERA_ENABLE_NVTILE)
    target_sources(tessera_gpu_backend PRIVATE
      src/lowering_nvtile_mlir.cpp
      src/lowering_nvvm_mlir.cpp
      src/kernels/ptx_wgmma_bf16.cu
    )
  endif()

  if(TESSERA_BUILD_TESTS)
    add_executable(test_wmma tests/test_wmma_gemm.cu)
    target_link_libraries(test_wmma PRIVATE tessera_gpu_backend)
    target_compile_options(test_wmma PRIVATE -arch=${TESSERA_CUDA_ARCH})

    add_executable(test_wgmma_tile_ir tests/test_wgmma_tile_ir.cu)
    target_link_libraries(test_wgmma_tile_ir PRIVATE tessera_gpu_backend)
    target_compile_options(test_wgmma_tile_ir PRIVATE -arch=${TESSERA_CUDA_ARCH})
  endif()

  if(TESSERA_BUILD_GPU_BENCH)
    add_executable(bench_gpu bench/bench_gpu.cu)
    target_link_libraries(bench_gpu PRIVATE tessera_gpu_backend)
    target_compile_options(bench_gpu PRIVATE -arch=${TESSERA_CUDA_ARCH})
  endif()
else()
  message(STATUS "NVIDIA CUDA runtime disabled; building hardware-free Target IR tools only")
endif()
