cmake_minimum_required(VERSION 3.20)
project(tessera_x86_backend LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(TESSERA_ENABLE_AMX "Enable Intel AMX kernels" ON)
option(TESSERA_ENABLE_AVX512 "Enable AVX-512 kernels" ON)

# Try to detect and add compiler flags for AVX-512 / AMX
include(CheckCXXCompilerFlag)

macro(try_add_flag flag var)
  check_cxx_compiler_flag(${flag} ${var})
  if(${var})
    add_compile_options(${flag})
  endif()
endmacro()

if (TESSERA_ENABLE_AVX512)
  try_add_flag("-mavx512f" HAS_AVX512F)
  try_add_flag("-mavx512bw" HAS_AVX512BW)
  try_add_flag("-mavx512dq" HAS_AVX512DQ)
  try_add_flag("-mavx512vl" HAS_AVX512VL)
  try_add_flag("-mavx512vnni" HAS_AVX512VNNI)
  try_add_flag("-mavx512bf16" HAS_AVX512BF16)
endif()

if (TESSERA_ENABLE_AMX)
  try_add_flag("-mamx-tile" HAS_AMX_TILE)
  try_add_flag("-mamx-int8" HAS_AMX_INT8)
  try_add_flag("-mamx-bf16" HAS_AMX_BF16)
endif()

add_library(tessera_x86_backend
  src/backend_x86.cpp
  src/kernels/avx512_gemm_bf16.cpp
  src/kernels/amx_gemm_bf16.cpp
  src/kernels/amx_gemm_int8.cpp
  src/kernels/avx512_vnni_gemm_int8.cpp
  src/kernels/avx512_reduce_f32.cpp
  src/kernels/avx512_unary_f32.cpp
  src/kernels/avx512_binary_f32.cpp
  src/kernels/avx512_compare_f32.cpp
  src/kernels/avx512_logical_i8.cpp
  src/kernels/avx512_bitwise_i32.cpp
  src/kernels/avx512_scan_f32.cpp
  src/kernels/avx512_argreduce_f32.cpp
  src/kernels/avx512_where_f32.cpp
  src/kernels/avx512_transcendental_f32.cpp
  src/kernels/avx512_norm_f32.cpp
  src/kernels/avx512_gemm_f32.cpp
  src/kernels/avx512_posenc_f32.cpp
  src/kernels/avx512_loss_f32.cpp
  src/kernels/avx512_policy_loss_f32.cpp
  src/kernels/avx512_fpquant_f32.cpp
  src/kernels/avx512_fft_f32.cpp
  src/kernels/avx512_sparse_f32.cpp
  src/kernels/epilogue.cpp
  src/runtime/amx_runtime.cpp
)

target_include_directories(tessera_x86_backend PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link libraries: none required for intrinsics; pthread for tests if needed
add_executable(test_gemm tests/test_gemm.cpp)
target_link_libraries(test_gemm PRIVATE tessera_x86_backend)

add_executable(test_gemm_reference_tails tests/test_gemm_reference_tails.cpp)
target_link_libraries(test_gemm_reference_tails PRIVATE tessera_x86_backend)

add_executable(test_reduce tests/test_reduce.cpp)
target_link_libraries(test_reduce PRIVATE tessera_x86_backend)

add_executable(test_unary tests/test_unary.cpp)
target_link_libraries(test_unary PRIVATE tessera_x86_backend)

add_executable(test_binary tests/test_binary.cpp)
target_link_libraries(test_binary PRIVATE tessera_x86_backend)

add_executable(test_compare tests/test_compare.cpp)
target_link_libraries(test_compare PRIVATE tessera_x86_backend)

add_executable(test_logical tests/test_logical.cpp)
target_link_libraries(test_logical PRIVATE tessera_x86_backend)

add_executable(test_bitwise tests/test_bitwise.cpp)
target_link_libraries(test_bitwise PRIVATE tessera_x86_backend)

add_executable(test_scan tests/test_scan.cpp)
target_link_libraries(test_scan PRIVATE tessera_x86_backend)

add_executable(test_argreduce tests/test_argreduce.cpp)
target_link_libraries(test_argreduce PRIVATE tessera_x86_backend)

add_executable(test_where tests/test_where.cpp)
target_link_libraries(test_where PRIVATE tessera_x86_backend)

add_executable(test_transcendental tests/test_transcendental.cpp)
target_link_libraries(test_transcendental PRIVATE tessera_x86_backend)

add_executable(test_binary_math tests/test_binary_math.cpp)
target_link_libraries(test_binary_math PRIVATE tessera_x86_backend)

add_executable(test_norm tests/test_norm.cpp)
target_link_libraries(test_norm PRIVATE tessera_x86_backend)

add_executable(test_gemm_f32 tests/test_gemm_f32.cpp)
target_link_libraries(test_gemm_f32 PRIVATE tessera_x86_backend)

add_executable(test_posenc tests/test_posenc.cpp)
target_link_libraries(test_posenc PRIVATE tessera_x86_backend)

add_executable(test_loss tests/test_loss.cpp)
target_link_libraries(test_loss PRIVATE tessera_x86_backend)

add_executable(test_policy_loss tests/test_policy_loss.cpp)
target_link_libraries(test_policy_loss PRIVATE tessera_x86_backend)

add_executable(test_fpquant tests/test_fpquant.cpp)
target_link_libraries(test_fpquant PRIVATE tessera_x86_backend)

add_executable(test_fft tests/test_fft.cpp)
target_link_libraries(test_fft PRIVATE tessera_x86_backend)

# ── Runtime-loadable elementwise kernel lane ────────────────────────────────
# A SHARED library of just the AVX-512 elementwise kernels (reduce / unary /
# binary / compare / logical / bitwise) — the C-ABI symbols the Python runtime
# ctypes-loads (`runtime.launch()` → `x86_*_compiled`), the CPU analog of the
# ROCm compiled lanes. Deliberately EXCLUDES the AMX/GEMM sources so it builds
# on toolchains without AMX intrinsics (e.g. gcc on this box); these kernels are
# pure AVX-512 + scalar reference.
add_library(tessera_x86_elementwise SHARED
  src/kernels/avx512_reduce_f32.cpp
  src/kernels/avx512_unary_f32.cpp
  src/kernels/avx512_binary_f32.cpp
  src/kernels/avx512_compare_f32.cpp
  src/kernels/avx512_logical_i8.cpp
  src/kernels/avx512_bitwise_i32.cpp
  src/kernels/avx512_scan_f32.cpp
  src/kernels/avx512_argreduce_f32.cpp
  src/kernels/avx512_where_f32.cpp
  src/kernels/avx512_transcendental_f32.cpp
  src/kernels/avx512_norm_f32.cpp
  src/kernels/avx512_gemm_f32.cpp
  src/kernels/avx512_posenc_f32.cpp
  src/kernels/avx512_loss_f32.cpp
  src/kernels/avx512_policy_loss_f32.cpp
  src/kernels/avx512_fpquant_f32.cpp
  src/kernels/avx512_fft_f32.cpp
  src/kernels/avx512_sparse_f32.cpp
)
