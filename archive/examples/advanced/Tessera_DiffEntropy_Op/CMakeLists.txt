cmake_minimum_required(VERSION 3.20)
project(Tessera_DiffEntropy LANGUAGES C CXX)

# Expect TESSERA_DIR to point to your Tessera/MLIR toolchain
if(NOT DEFINED TESSERA_DIR)
  message(FATAL_ERROR "Set -DTESSERA_DIR=... to your Tessera root")
endif()

add_library(tessera_diffentropy STATIC
  lib/DiffEntropy/DiffEntropyOps.cpp
  lib/DiffEntropy/DiffEntropyLowering.cpp
  lib/DiffEntropy/kernels/dentropy_ref.cpp
)
target_include_directories(tessera_diffentropy PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${TESSERA_DIR}/include
)

add_executable(tessera-dentropy-opt tools/dentropy_opt_main.cpp)
target_link_libraries(tessera-dentropy-opt PRIVATE tessera_diffentropy)
