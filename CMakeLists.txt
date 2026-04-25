# Tessera: Next-Generation Deep Learning Programming Model
cmake_minimum_required(VERSION 3.20)

project(Tessera
    VERSION 0.1.0
    DESCRIPTION "Next-Generation Deep Learning Programming Model"
    LANGUAGES CXX CUDA
)

# Single authoritative version variable — referenced by all sub-components
set(TESSERA_VERSION "${PROJECT_VERSION}" CACHE STRING "Tessera version string" FORCE)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --------------------------------------------------------------------------
# Build options
# --------------------------------------------------------------------------
option(TESSERA_BUILD_TESTS    "Build Tessera tests"     ON)
option(TESSERA_BUILD_EXAMPLES "Build Tessera examples"  ON)
option(TESSERA_BUILD_PYTHON   "Build Python bindings"   ON)
option(TESSERA_ENABLE_CUDA    "Enable CUDA support"     ON)
option(TESSERA_ENABLE_HIP     "Enable ROCm/HIP support" OFF)

# --------------------------------------------------------------------------
# LLVM / MLIR
# --------------------------------------------------------------------------
find_package(LLVM 18 REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

# Pull in MLIR cmake helpers (TableGen, AddMLIR, …) — required by sub-components
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# --------------------------------------------------------------------------
# CUDA / HIP
# --------------------------------------------------------------------------
if(TESSERA_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90")
endif()

if(TESSERA_ENABLE_HIP)
    find_package(hip REQUIRED CONFIG)
endif()

# --------------------------------------------------------------------------
# Global include paths
# --------------------------------------------------------------------------
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/mlir/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/src/mlir/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/include)

add_definitions(${LLVM_DEFINITIONS})

# --------------------------------------------------------------------------
# Source tree (dialects, runtime, compiler passes)
# --------------------------------------------------------------------------
add_subdirectory(src)

# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
if(TESSERA_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# --------------------------------------------------------------------------
# Examples
# --------------------------------------------------------------------------
if(TESSERA_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# --------------------------------------------------------------------------
# Python bindings (future milestone — subdir wired when ready)
# --------------------------------------------------------------------------
if(TESSERA_BUILD_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
endif()

# --------------------------------------------------------------------------
# Configuration summary
# --------------------------------------------------------------------------
message(STATUS "----------------------------------------------------")
message(STATUS "Tessera ${TESSERA_VERSION} build configuration")
message(STATUS "  LLVM/MLIR version : ${LLVM_PACKAGE_VERSION}")
message(STATUS "  CUDA support      : ${TESSERA_ENABLE_CUDA}")
message(STATUS "  HIP  support      : ${TESSERA_ENABLE_HIP}")
message(STATUS "  Build tests       : ${TESSERA_BUILD_TESTS}")
message(STATUS "  Build examples    : ${TESSERA_BUILD_EXAMPLES}")
message(STATUS "  Build Python      : ${TESSERA_BUILD_PYTHON}")
message(STATUS "----------------------------------------------------")
