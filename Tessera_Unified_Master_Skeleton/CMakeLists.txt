cmake_minimum_required(VERSION 3.22)
project(TesseraUnified LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
enable_testing()

# --- Global options ---
option(TESSERA_ENABLE_CUDA "Enable CUDA backend" ON)
option(TESSERA_ENABLE_VERTICAL_GEMM_BN "Build GEMM+BatchNorm vertical slice" ON)

add_subdirectory(tessera/runtime)

if(TESSERA_ENABLE_VERTICAL_GEMM_BN)
  add_subdirectory(tessera/verticals/gemm_bn)
endif()

# Hook: top-level test aggregator (if verticals add tests they appear in ctest)
