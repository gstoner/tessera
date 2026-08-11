
cmake_minimum_required(VERSION 3.20)
project(tessera_src_optimizations C CXX)

option(TESSERA_WITH_CUDA "Build CUDA examples" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 01 and 02 include <immintrin.h> and use AVX (`__m256`, `_mm256_*`)
# unconditionally, so they compile only on x86. On any other host they fail the
# whole `ninja -C build`, which is not cosmetic: ninja stopping here leaves
# later targets (notably tessera-opt) stale, and that surfaces as phantom lit
# failures. Gate the two targets, not the directory — 03 and 07 are portable
# (07 uses `__builtin_prefetch`).
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64|i[3-6]86)$")
  add_executable(01_loop_tiling_blocking src/01_loop_tiling_blocking.cpp)
  target_compile_options(01_loop_tiling_blocking PRIVATE -O3)

  add_executable(02_vectorization_intrinsics src/02_vectorization_intrinsics.cpp)
  target_compile_options(02_vectorization_intrinsics PRIVATE -O3)
else()
  message(STATUS
    "examples/optimization: skipping 01_loop_tiling_blocking and "
    "02_vectorization_intrinsics (x86 AVX intrinsics; host is "
    "${CMAKE_SYSTEM_PROCESSOR})")
endif()

add_executable(03_cache_friendly_layout src/03_cache_friendly_layout.cpp)
target_compile_options(03_cache_friendly_layout PRIVATE -O3)

add_executable(07_software_prefetch src/07_software_prefetch.cpp)
target_compile_options(07_software_prefetch PRIVATE -O3)

if (TESSERA_WITH_CUDA)
  enable_language(CUDA)
  add_executable(04_parallel_reduction_shared_mem src/04_parallel_reduction_shared_mem.cu)
  set_target_properties(04_parallel_reduction_shared_mem PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_compile_options(04_parallel_reduction_shared_mem PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

  add_executable(06_branchless_kernels src/06_branchless_kernels.cu)
  set_target_properties(06_branchless_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_compile_options(06_branchless_kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

  add_executable(05_async_copy_tma_wgmma src/05_async_copy_tma_wgmma.cu)
  set_target_properties(05_async_copy_tma_wgmma PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_compile_options(05_async_copy_tma_wgmma PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
endif()
