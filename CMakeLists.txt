# Tessera: Next-Generation Deep Learning Programming Model
cmake_minimum_required(VERSION 3.20)

project(Tessera
    VERSION 0.1.0
    DESCRIPTION "Next-Generation Deep Learning Programming Model"
    LANGUAGES C CXX
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
option(TESSERA_BUILD_APPLE_BACKEND     "Build Apple Silicon compiler backend" OFF)
option(TESSERA_BUILD_CLIFFORD_BACKEND  "Build the Clifford / Geometric-Algebra dialect (GA7+)" OFF)
option(TESSERA_BUILD_EBM_BACKEND       "Build the Energy-Based-Model dialect (EBM5+)" OFF)

# --------------------------------------------------------------------------
# Sanitizers — runtime instrumentation for the C++ runtime + collectives.
#
# The findings audit (2026-05-19) noted that the highest-leverage
# static-analysis investment for Tessera's C++ surface is a sanitizer
# build: TSAN would have caught the PerfettoTraceWriter race on its
# first run, and ASAN catches the use-after-free shape the
# live-handle ratchet in tsrShutdown was added to prevent.  Two
# smoke binaries exercise the surface under each sanitizer:
#
#   tessera-collective-runtime-smoke — collective `tessera_*` C ABI
#                                       (qos, submit, shutdown, trace).
#   tessera-runtime-abi-smoke         — device runtime `tsr*` C ABI
#                                       (Init / Malloc / CreateStream /
#                                       Free / Shutdown lifecycle,
#                                       including the live-handle
#                                       refusal path and an
#                                       init-after-shutdown cycle).
#
# Both are driven by scripts/run_sanitizers.sh and locked by the
# parametrized tests in tests/unit/test_sanitizer_smoke.py.  The
# runtime smoke also runs unsanitized in every Python sweep via
# tests/unit/test_runtime_abi_smoke.py.
#
# Mutually exclusive — pick one per build directory:
#
#   off    (default)
#   address    →  -fsanitize=address,undefined  (AddressSanitizer + UBSan)
#   thread     →  -fsanitize=thread             (ThreadSanitizer)
#   undefined  →  -fsanitize=undefined          (UndefinedBehaviorSanitizer)
#
# Usage (separate build dirs per sanitizer):
#
#   cmake -B build-asan -DTESSERA_ENABLE_SANITIZERS=address
#   cmake -B build-tsan -DTESSERA_ENABLE_SANITIZERS=thread
#
# Then build and run the smoke binary under each:
#
#   cmake --build build-asan --target tessera-collective-runtime-smoke
#   build-asan/src/collectives/tools/tessera-collective-runtime-smoke/...
#
# Combined with a -DCMAKE_BUILD_TYPE=Debug build the sanitizers
# emit precise line-number diagnostics on a violation.
# --------------------------------------------------------------------------
set(TESSERA_ENABLE_SANITIZERS "off" CACHE STRING
  "C++ sanitizer build (off | address | thread | undefined)")
set_property(CACHE TESSERA_ENABLE_SANITIZERS PROPERTY
  STRINGS off address thread undefined)

if(TESSERA_ENABLE_SANITIZERS STREQUAL "address")
  set(_TESSERA_SAN_FLAGS "-fsanitize=address,undefined" "-fno-omit-frame-pointer" "-fno-sanitize-recover=all")
elseif(TESSERA_ENABLE_SANITIZERS STREQUAL "thread")
  set(_TESSERA_SAN_FLAGS "-fsanitize=thread" "-fno-omit-frame-pointer" "-fno-sanitize-recover=all")
elseif(TESSERA_ENABLE_SANITIZERS STREQUAL "undefined")
  set(_TESSERA_SAN_FLAGS "-fsanitize=undefined" "-fno-omit-frame-pointer" "-fno-sanitize-recover=all")
elseif(TESSERA_ENABLE_SANITIZERS STREQUAL "off")
  set(_TESSERA_SAN_FLAGS "")
else()
  message(FATAL_ERROR
    "TESSERA_ENABLE_SANITIZERS must be one of: off, address, thread, undefined "
    "(got: ${TESSERA_ENABLE_SANITIZERS})")
endif()

if(_TESSERA_SAN_FLAGS)
  add_compile_options(${_TESSERA_SAN_FLAGS})
  add_link_options(${_TESSERA_SAN_FLAGS})
  message(STATUS "Sanitizers ON: ${TESSERA_ENABLE_SANITIZERS} (${_TESSERA_SAN_FLAGS})")
endif()

if(TESSERA_CPU_ONLY)
    set(TESSERA_ENABLE_CUDA OFF CACHE BOOL "Enable CUDA support" FORCE)
    set(TESSERA_ENABLE_HIP OFF CACHE BOOL "Enable ROCm/HIP support" FORCE)
endif()

# --------------------------------------------------------------------------
# LLVM / MLIR
# --------------------------------------------------------------------------
# Tessera targets LLVM/MLIR 22+. We match without a version constraint so any
# 22.x patch release works (LLVM's CMake config only accepts exact major.minor
# matches via the find_package version arg), then assert the major below.
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

if(LLVM_VERSION_MAJOR LESS 22)
  message(FATAL_ERROR
    "Tessera requires LLVM/MLIR 22 or newer; found ${LLVM_PACKAGE_VERSION}")
endif()

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
add_subdirectory(tools/tessera-translate)
# Phase 0 production-lane CPU JIT (experimental). docs/spec/PRODUCTION_COMPILER_PLAN.md
add_subdirectory(tools/tessera-jit)

set(TESSERA_COMPILER_FOUNDATION_DEPS
    TesseraIRTableGen
    TesseraAttnTableGen
    TesseraQueueTableGen
    TesseraIR
    TesseraPasses
    tessera-opt
)
if(TARGET TesseraSolversTableGen)
    list(APPEND TESSERA_COMPILER_FOUNDATION_DEPS TesseraSolversTableGen)
endif()
if(TARGET TesseraLinalgSolverTableGen)
    list(APPEND TESSERA_COMPILER_FOUNDATION_DEPS TesseraLinalgSolverTableGen)
endif()
if(TARGET tessera_solvers_passes)
    list(APPEND TESSERA_COMPILER_FOUNDATION_DEPS tessera_solvers_passes)
endif()
if(TARGET tessera_linalg_solver_passes)
    list(APPEND TESSERA_COMPILER_FOUNDATION_DEPS tessera_linalg_solver_passes)
endif()

add_custom_target(TesseraCompilerFoundation
    DEPENDS ${TESSERA_COMPILER_FOUNDATION_DEPS}
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
if(TESSERA_BUILD_EXAMPLES AND
   ((TESSERA_BUILD_ROCM_BACKEND AND NOT TESSERA_ENABLE_CUDA) OR
    (TESSERA_BUILD_NVIDIA_BACKEND AND NOT TESSERA_ENABLE_CUDA)))
    message(STATUS "Skipping examples for hardware-free target build because CUDA is disabled")
elseif(TESSERA_BUILD_EXAMPLES)
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
message(STATUS "  Apple backend     : ${TESSERA_BUILD_APPLE_BACKEND}")
message(STATUS "  Clifford backend  : ${TESSERA_BUILD_CLIFFORD_BACKEND}")
message(STATUS "  EBM backend       : ${TESSERA_BUILD_EBM_BACKEND}")
message(STATUS "  Sanitizers        : ${TESSERA_ENABLE_SANITIZERS}")
message(STATUS "  Build tests       : ${TESSERA_BUILD_TESTS}")
message(STATUS "  Build examples    : ${TESSERA_BUILD_EXAMPLES}")
message(STATUS "  Build Python      : ${TESSERA_BUILD_PYTHON}")
message(STATUS "----------------------------------------------------")
