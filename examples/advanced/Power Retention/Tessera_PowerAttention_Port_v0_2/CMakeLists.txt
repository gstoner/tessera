cmake_minimum_required(VERSION 3.20)
project(TesseraPowerAttention LANGUAGES CXX C CUDA)
option(TESSERA_ENABLE_CUDA "Build CUDA kernels" ON)
option(TESSERA_ENABLE_HIP  "Build HIP kernels" OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(TableGen)
include(AddMLIR)
if (TESSERA_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_definitions(-DTESSERA_POWER_USE_CUDA=1)
endif()
if (TESSERA_ENABLE_HIP)
  find_package(hip REQUIRED)
  add_definitions(-DTESSERA_POWER_USE_HIP=1)
endif()
include_directories(include ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
set(LLVM_TARGET_DEFINITIONS include/tessera/power/TesseraPower.td)
mlir_tablegen(TesseraPowerOps.h.inc -gen-op-decls)
mlir_tablegen(TesseraPowerOps.cpp.inc -gen-op-defs)
mlir_tablegen(TesseraPowerDialect.h.inc -gen-dialect-decls -dialect=power)
mlir_tablegen(TesseraPowerDialect.cpp.inc -gen-dialect-defs -dialect=power)
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
if (TESSERA_ENABLE_CUDA)
  add_library(tessera_power_kernels_cuda STATIC src/kernels/cuda/power_attention.cu)
  target_link_libraries(tessera_power_kernels_cuda PUBLIC CUDA::cudart)
  set_target_properties(tessera_power_kernels_cuda PROPERTIES CUDA_ARCHITECTURES "80;89;90;100")
endif()
if (TESSERA_ENABLE_HIP)
  add_library(tessera_power_kernels_hip STATIC src/kernels/hip/power_attention.hip.cu)
  target_link_libraries(tessera_power_kernels_hip PUBLIC hip::device)
endif()
add_library(tessera_power_runtime STATIC lib/Runtime/Runtime.cpp)
target_link_libraries(tessera_power_runtime PUBLIC MLIRIR
  $<IF:$<BOOL:${TESSERA_ENABLE_CUDA}>,tessera_power_kernels_cuda,>
  $<IF:$<BOOL:${TESSERA_ENABLE_HIP}>,tessera_power_kernels_hip,>)
add_subdirectory(src/extension)
enable_testing()