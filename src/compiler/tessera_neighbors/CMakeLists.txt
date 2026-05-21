# Tessera Neighbors — neighbor-discovery, topology, and stencil lowering
# Canonical component: renamed from tessera-neighbors (hyphen → underscore)
#
# Can be built standalone or as part of the Tessera monorepo.
#
# Exported targets:
#   TesseraNeighbors  — topology passes (HaloInfer, StencilLower,
#                       PipelineOverlap, DynamicTopology)

# ------------------------------------------------------------------
# Standalone-only preamble
# ------------------------------------------------------------------
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.20)
    project(TesseraNeighbors LANGUAGES CXX)

    find_package(LLVM REQUIRED CONFIG)
    find_package(MLIR REQUIRED CONFIG)
    if(LLVM_VERSION_MAJOR LESS 21)
      message(FATAL_ERROR
        "TesseraNeighbors requires LLVM/MLIR 21+; found ${LLVM_PACKAGE_VERSION}")
    endif()
    list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
    include(TableGen)
    include(AddLLVM)
    include(AddMLIR)
    include(HandleLLVMOptions)
    include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
    add_definitions(${LLVM_DEFINITIONS})
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------------------------------------------------------------
# Library
# ------------------------------------------------------------------
add_library(TesseraNeighbors
    lib/Dialect/Neighbors/IR/TesseraNeighbors.cpp
    lib/Dialect/Neighbors/Transforms/HaloInferPass.cpp
    lib/Dialect/Neighbors/Transforms/StencilLowerPass.cpp
    lib/Dialect/Neighbors/Transforms/BoundaryConditionLowerPass.cpp
    lib/Dialect/Neighbors/Transforms/StencilLoopMaterializePass.cpp
    lib/Dialect/Neighbors/Transforms/HaloMeshIntegrationPass.cpp
    lib/Dialect/Neighbors/Transforms/PipelineOverlapPass.cpp
    lib/Dialect/Neighbors/Transforms/DynamicTopologyPass.cpp
)

target_include_directories(TesseraNeighbors PUBLIC include)

target_link_libraries(TesseraNeighbors
    PUBLIC
        MLIRIR
        MLIRPass
        MLIRSupport
        MLIRArithDialect
        MLIRSCFDialect
        MLIRTensorDialect
        MLIRFuncDialect
)

# ODS / tablegen: wire tessera_neighbors.td here when tablegen stubs are added.
# Lit tests: see test/Neighbors/*.mlir — driven by parent check-tessera target.
