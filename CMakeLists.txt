# Tessera: Next-Generation Deep Learning Programming Model
cmake_minimum_required(VERSION 3.20)

project(Tessera
    VERSION 0.1.0
    DESCRIPTION "Next-Generation Deep Learning Programming Model"
    LANGUAGES CXX
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
option(TESSERA_CPU_ONLY       "Disable GPU backends"     OFF)
option(TESSERA_ENABLE_CUDA    "Enable CUDA support"     OFF)
option(TESSERA_ENABLE_HIP     "Enable ROCm/HIP support" OFF)
option(TESSERA_BUILD_X86_BACKEND      "Build x86 compiler backend" OFF)
option(TESSERA_BUILD_NVIDIA_BACKEND   "Build NVIDIA compiler backend" OFF)
option(TESSERA_BUILD_ROCM_BACKEND     "Build ROCm compiler backend" OFF)
option(TESSERA_BUILD_TPU_BACKEND      "Build TPU compiler backend" OFF)
option(TESSERA_BUILD_METALIUM_BACKEND  "Build TT-Metalium compiler backend" OFF)
option(TESSERA_BUILD_CEREBRAS_BACKEND  "Build Cerebras compiler backend" OFF)
option(TESSERA_BUILD_RUBINCPX_BACKEND  "Build Rubin CPX compiler backend (NVRubinCPX dialect + passes)" OFF)

if(TESSERA_CPU_ONLY)
    set(TESSERA_ENABLE_CUDA OFF CACHE BOOL "Enable CUDA support" FORCE)
    set(TESSERA_ENABLE_HIP OFF CACHE BOOL "Enable ROCm/HIP support" FORCE)
endif()

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
    if(DEFINED CUDA_TOOLKIT_ROOT_DIR AND NOT DEFINED CUDAToolkit_ROOT)
        set(CUDAToolkit_ROOT "${CUDA_TOOLKIT_ROOT_DIR}" CACHE PATH "CUDA Toolkit root")
    endif()
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
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/compiler/mlir/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/src/compiler/mlir/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/include)

add_definitions(${LLVM_DEFINITIONS})

# --------------------------------------------------------------------------
# Source tree (dialects, runtime, compiler passes)
# --------------------------------------------------------------------------
add_subdirectory(src)
add_subdirectory(tools/tessera-opt)

add_custom_target(TesseraCompilerFoundation
    DEPENDS
        TesseraIRTableGen
        TesseraAttnTableGen
        TesseraQueueTableGen
        TesseraSolversTableGen
        TesseraLinalgSolverTableGen
        TesseraIR
        TesseraPasses
        tessera_solvers_passes
        tessera_linalg_solver_passes
        tessera-opt
)

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
message(STATUS "  CPU only          : ${TESSERA_CPU_ONLY}")
message(STATUS "  HIP  support      : ${TESSERA_ENABLE_HIP}")
message(STATUS "  x86 backend       : ${TESSERA_BUILD_X86_BACKEND}")
message(STATUS "  NVIDIA backend    : ${TESSERA_BUILD_NVIDIA_BACKEND}")
message(STATUS "  ROCm backend      : ${TESSERA_BUILD_ROCM_BACKEND}")
message(STATUS "  TPU backend       : ${TESSERA_BUILD_TPU_BACKEND}")
message(STATUS "  Metalium backend  : ${TESSERA_BUILD_METALIUM_BACKEND}")
message(STATUS "  Cerebras backend  : ${TESSERA_BUILD_CEREBRAS_BACKEND}")
message(STATUS "  Rubin CPX backend : ${TESSERA_BUILD_RUBINCPX_BACKEND}")
message(STATUS "  Build tests       : ${TESSERA_BUILD_TESTS}")
message(STATUS "  Build examples    : ${TESSERA_BUILD_EXAMPLES}")
message(STATUS "  Build Python      : ${TESSERA_BUILD_PYTHON}")
message(STATUS "----------------------------------------------------")
