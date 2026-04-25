# Scaling / Resilience dialect
# Canonical component: promoted from tessera_scaling_resilience_v1_1
# (v1_1 adds SRDialect.cpp, SROps.cpp, SROps.td, manifest.schema.json over v1)
#
# Can be built standalone or as part of the Tessera monorepo.
#
# Exported targets:
#   DialectSR   — SR dialect IR (SRDialect + SROps)
#   TesseraSR   — SR compiler passes (recompute, shard, resilience, manifest export)

# ------------------------------------------------------------------
# Standalone-only preamble (skipped when built as add_subdirectory)
# ------------------------------------------------------------------
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.20)
    project(Tessera_SR LANGUAGES CXX)

    find_package(LLVM 18 REQUIRED CONFIG)
    find_package(MLIR REQUIRED CONFIG)
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
# SR dialect (IR layer)
# ------------------------------------------------------------------
add_library(DialectSR
    lib/sr/dialect/SRDialect.cpp
    lib/sr/dialect/SROps.cpp
)
target_include_directories(DialectSR PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(DialectSR PUBLIC MLIRIR MLIRSupport)

# ------------------------------------------------------------------
# SR passes
# ------------------------------------------------------------------
add_library(TesseraSR
    lib/sr/passes/InsertRecomputePass.cpp
    lib/sr/passes/OptimizerShardPass.cpp
    lib/sr/passes/ResilienceRestartPass.cpp
    lib/sr/passes/ExportDeploymentManifestPass.cpp
    lib/sr/passes/RegisterPasses.cpp
)
target_include_directories(TesseraSR PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(TesseraSR PUBLIC MLIRIR MLIRPass DialectSR)

# Lit tests wired via parent check-tessera target
