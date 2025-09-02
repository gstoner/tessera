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
  src/kernels/epilogue.cpp
  src/runtime/amx_runtime.cpp
)

target_include_directories(tessera_x86_backend PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link libraries: none required for intrinsics; pthread for tests if needed
add_executable(test_gemm tests/test_gemm.cpp)
target_link_libraries(test_gemm PRIVATE tessera_x86_backend)
