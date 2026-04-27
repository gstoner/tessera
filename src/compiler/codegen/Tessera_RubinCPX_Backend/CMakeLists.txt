cmake_minimum_required(VERSION 3.20)

# ─── Standalone vs. sub-directory detection ──────────────────────────────────
# When built as part of the main Tessera tree, LLVM/MLIR are already found and
# CMake helpers are already included.  When built standalone (for CPX-only
# development), we find them ourselves.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(TesseraRubinCPXBackend LANGUAGES C CXX)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)

  option(TESSERA_BUILD_CPX_OPT         "Build tessera-cpx-opt driver"  ON)
  option(TESSERA_RUBINCPX_ENABLE_TESTS "Enable Rubin CPX lit tests"     ON)

  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
  find_package(LLVM REQUIRED CONFIG)
  find_package(MLIR REQUIRED CONFIG)
  message(STATUS "[RubinCPX] Found MLIR: ${MLIR_DIR}")
  list(APPEND CMAKE_MODULE_PATH ${MLIR_CMAKE_DIR} ${LLVM_CMAKE_DIR})
  include(TableGen)
  include(AddMLIR)
  include(AddLLVM)
  include(HandleLLVMOptions)
  include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
  link_directories(${MLIR_LIB_DIR})
  add_definitions(${LLVM_DEFINITIONS})
else()
  # Sub-directory build: parent already did find_package + includes.
  option(TESSERA_BUILD_CPX_OPT         "Build tessera-cpx-opt driver"  ON)
  option(TESSERA_RUBINCPX_ENABLE_TESTS "Enable Rubin CPX lit tests"     OFF)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(include)
add_subdirectory(lib)

if(TESSERA_BUILD_CPX_OPT)
  add_subdirectory(tools/tessera-cpx-opt)
endif()

if(TESSERA_RUBINCPX_ENABLE_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()
