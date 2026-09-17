cmake_minimum_required(VERSION 3.20)
project(TesseraPowerAttention LANGUAGES CXX C)
option(TESSERA_ENABLE_CUDA "Build CUDA kernels" OFF)
option(TESSERA_ENABLE_HIP  "Build HIP kernels" OFF)
set(CMAKE_CXX_STANDARD 17)
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)
if(NOT LLVM_VERSION_MAJOR EQUAL 23 OR
   NOT DEFINED MLIR_VERSION_MAJOR OR NOT MLIR_VERSION_MAJOR EQUAL 23)
  message(FATAL_ERROR
    "TesseraPowerAttention requires LLVM/MLIR 23.x exactly; found LLVM "
    "${LLVM_PACKAGE_VERSION}, MLIR ${MLIR_PACKAGE_VERSION}")
endif()
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(TableGen)
include(AddMLIR)
if (TESSERA_ENABLE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  find_package(CUDAToolkit REQUIRED)
  add_definitions(-DTESSERA_POWER_USE_CUDA=1)
endif()
if (TESSERA_ENABLE_HIP)
  find_package(hip REQUIRED)
  add_definitions(-DTESSERA_POWER_USE_HIP=1)
endif()
# Generated headers go under the *binary* include tree at the path the sources
# include them from (`tessera/power/*.inc`). They used to land flat in the
# binary dir with only the source `include/` on the path, so on a fresh worktree
# of main configured with TESSERA_ENABLE_CUDA=ON -- which puts this example in
# `all` -- the dialect could not compile at all: `fatal error:
# tessera/power/TesseraPowerDialect.h.inc: No such file` (Super-Bear, 2026-09-17).
# The lived-in trees on that box never compiled this target in their default
# build, which is how a header path that cannot work survived.
include_directories(include ${CMAKE_CURRENT_BINARY_DIR}/include ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/tessera/power)
set(LLVM_TARGET_DEFINITIONS include/tessera/power/TesseraPower.td)
mlir_tablegen(include/tessera/power/TesseraPowerOps.h.inc -gen-op-decls)
mlir_tablegen(include/tessera/power/TesseraPowerOps.cpp.inc -gen-op-defs)
mlir_tablegen(include/tessera/power/TesseraPowerDialect.h.inc -gen-dialect-decls -dialect=power)
mlir_tablegen(include/tessera/power/TesseraPowerDialect.cpp.inc -gen-dialect-defs -dialect=power)
add_public_tablegen_target(TesseraPowerIncGen)
add_library(TesseraPowerDialect
  lib/Dialect/Power/IR/TesseraPowerDialect.cpp
  lib/Dialect/Power/IR/TesseraPowerOps.cpp
)
target_link_libraries(TesseraPowerDialect PUBLIC MLIRIR)
add_dependencies(TesseraPowerDialect TesseraPowerIncGen)
add_library(TesseraPowerTransforms
  lib/Dialect/Power/Transforms/LowerToTile.cpp
  lib/Dialect/Power/Transforms/LowerToTarget.cpp
  lib/Dialect/Power/Transforms/Passes.cpp
)
target_link_libraries(TesseraPowerTransforms PUBLIC
  TesseraPowerDialect MLIRIR MLIRPass MLIRTransforms MLIRFuncDialect)
# The kernel libraries and the runtime that links them are EXCLUDE_FROM_ALL:
# `src/kernels/cuda/power_attention.cu` does not compile (an undefined helper
# `compute_phi2_q_to_smem_bf16`, an undefined `s`, a malformed declaration at
# its line 226 -- found on a fresh CUDA-configured worktree of main, Super-Bear,
# 2026-09-17), and a scaffold's kernel must not decide whether the repository
# builds. The dialect and passes above stay in `all` and do compile; build the
# kernels by name when the scaffold grows a real kernel.
if (TESSERA_ENABLE_CUDA)
  add_library(tessera_power_kernels_cuda STATIC EXCLUDE_FROM_ALL src/kernels/cuda/power_attention.cu)
  target_link_libraries(tessera_power_kernels_cuda PUBLIC CUDA::cudart)
  # 120 = consumer Blackwell (RTX 50-series), the fleet's only CUDA device.
  set_target_properties(tessera_power_kernels_cuda PROPERTIES CUDA_ARCHITECTURES "80;89;90;100;120")
endif()
if (TESSERA_ENABLE_HIP)
  add_library(tessera_power_kernels_hip STATIC EXCLUDE_FROM_ALL src/kernels/hip/power_attention.hip.cu)
  target_link_libraries(tessera_power_kernels_hip PUBLIC hip::device)
endif()
add_library(tessera_power_runtime STATIC EXCLUDE_FROM_ALL lib/Runtime/Runtime.cpp lib/Runtime/Autotune.cpp)
target_link_libraries(tessera_power_runtime PUBLIC MLIRIR
  $<IF:$<BOOL:${TESSERA_ENABLE_CUDA}>,tessera_power_kernels_cuda,>
  $<IF:$<BOOL:${TESSERA_ENABLE_HIP}>,tessera_power_kernels_hip,>)
# src/extension holds a two-line pybind stub over <torch/extension.h> and no
# CMakeLists, so this add_subdirectory aborted configure outright — and only
# under -DTESSERA_ENABLE_CUDA=ON, which is why it survived: every CPU-only
# configure skips this file's CUDA branch above and never reaches here. A torch
# extension also cannot be a build target of this project (Decision #23), so
# the guard is the fix rather than writing the missing CMakeLists.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/extension/CMakeLists.txt")
  add_subdirectory(src/extension)
endif()
enable_testing()
